import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent.bot_chain import (
    BotChainCancelled,
    BotChainResult,
    BotChainStep,
    BotChainRunner,
    parse_bot_chain_message,
)
from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli.bot_profiles import BotProfile


class _AsyncStore:
    def __init__(self):
        self._store = object()
        self.appended = []
        self.updated = []

    async def has_platform_message_id(self, session_id, message_id):
        return False

    async def append_to_transcript(self, session_id, message):
        self.appended.append((session_id, message))

    async def update_session(self, session_key, **kwargs):
        self.updated.append((session_key, kwargs))


def test_gateway_bot_chain_returns_one_labeled_reply_and_persists_retryable_turn(
    monkeypatch,
):
    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    first = BotProfile(
        name="deepseek",
        path=Path("/tmp/deepseek"),
        model="deepseek-v4",
        provider="deepseek",
        system_prompt="First",
    )
    second = BotProfile(
        name="default",
        path=Path("/tmp/default"),
        model="test/model",
        provider="nous",
        system_prompt="Second",
    )
    result = BotChainResult(
        prompt="do the task",
        steps=(
            BotChainStep(first, "do the task", "draft"),
            BotChainStep(second, "handoff", "final answer"),
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [first, second],
    )
    monkeypatch.setattr(BotChainRunner, "run", lambda *_args, **_kwargs: result)

    runner = object.__new__(GatewayRunner)
    async_store = _AsyncStore()
    runner.session_store = async_store._store
    runner._async_session_store = async_store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-7", thread_id="31")
    event = SimpleNamespace(
        text="$DeepSeek $Default do the task",
        message_id="telegram-42",
        internal=False,
        source=source,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    response = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert response == (
        "$deepseek:\ndraft\n\n$default (final):\nfinal answer"
    )
    assert [row[1]["role"] for row in async_store.appended] == [
        "user",
        "assistant",
    ]
    assert async_store.appended[0][1]["content"] == event.text
    assert async_store.appended[0][1]["message_id"] == "telegram-42"
    assert async_store.appended[1][1]["content"] == response
    assert async_store.updated == [
        (session.session_key, {"touch_activity": True})
    ]


def test_gateway_bot_chain_stop_suppresses_duplicate_delivery(monkeypatch):
    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    profile = BotProfile(
        name="worker",
        path=Path("/tmp/worker"),
        model="test/model",
        provider="nous",
        system_prompt="Work",
    )
    monkeypatch.setattr(
        "hermes_cli.bot_profiles.resolve_bot_chain",
        lambda _names: [profile],
    )
    def _cancelled(*_args, **_kwargs):
        raise BotChainCancelled("Bot chain stopped.")

    monkeypatch.setattr(BotChainRunner, "run", _cancelled)

    runner = object.__new__(GatewayRunner)
    async_store = _AsyncStore()
    runner.session_store = async_store._store
    runner._async_session_store = async_store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-43",
        internal=False,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    response = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert response is None
    assert async_store.appended[-1][1]["content"] == "Bot chain stopped."
