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
        self.admissions = {}
        self.settlements = []
        self.fail_user_append = False

    async def has_platform_message_id(self, session_id, message_id):
        return False

    async def admit_bot_chain_delivery(self, session_id, message_id, chain_name):
        key = (session_id, message_id)
        if key in self.admissions:
            return self.admissions[key]
        self.admissions[key] = "admitted"
        return "admitted"

    async def mark_bot_chain_delivery_running(self, session_id, message_id):
        pass

    async def settle_bot_chain_delivery(self, session_id, message_id, *, outcome, detail=""):
        self.admissions[(session_id, message_id)] = "settled"
        self.settlements.append((session_id, message_id, outcome))

    async def append_to_transcript(self, session_id, message):
        if self.fail_user_append and message.get("role") == "user":
            raise OSError("state.db write failed")
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


def _wire_runner(monkeypatch, async_store):
    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    runner = object.__new__(GatewayRunner)
    runner.session_store = async_store._store
    runner._async_session_store = async_store
    state = SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0.0))
    runner._session_state = lambda _key: state
    return runner


def test_gateway_bot_chain_redelivery_after_transcript_failure_never_reexecutes(
    monkeypatch,
):
    """Reviewer regression (#100758 blocker 3): the FIRST gateway transcript
    append fails after the chain executed; the platform then redelivers the
    same message id. The admission receipt must forbid a second
    BotChainRunner execution (and therefore a second canonical-history
    publish)."""
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
    run_calls = []

    def _run(self, profiles, prompt, **kwargs):
        run_calls.append(kwargs.get("conversation_name"))
        return BotChainResult(
            prompt=prompt,
            steps=(BotChainStep(profile, prompt, "done"),),
        )

    monkeypatch.setattr(BotChainRunner, "run", _run)

    async_store = _AsyncStore()
    async_store.fail_user_append = True  # the exact failure from the review
    runner = _wire_runner(monkeypatch, async_store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-99",
        internal=False,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    first = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )
    # The chain executed and the response is returned even though the
    # transcript write failed.
    assert first == "$worker (final):\ndone"
    assert async_store.settlements == [("session-1", "telegram-99", "completed")]

    # Platform redelivers the same message id.
    second = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert second is None
    # Exactly one execution, and the admission receipt carried the chain
    # identity that was used for the run.
    assert len(run_calls) == 1
    assert run_calls[0] is not None and run_calls[0].startswith("Bot Chain ")


def test_gateway_bot_chain_refuses_to_execute_without_durable_receipt(
    monkeypatch,
):
    """Admission write failure fails closed: no receipt, no execution."""

    class _BrokenAdmissionStore(_AsyncStore):
        async def admit_bot_chain_delivery(self, session_id, message_id, chain_name):
            raise OSError("state.db unavailable")

    run_calls = []
    monkeypatch.setattr(
        BotChainRunner, "run", lambda *a, **k: run_calls.append(k) or None
    )
    async_store = _BrokenAdmissionStore()
    runner = _wire_runner(monkeypatch, async_store)
    event = SimpleNamespace(
        text="$worker do the task",
        message_id="telegram-100",
        internal=False,
    )
    session = SimpleNamespace(session_id="session-1", session_key="telegram:chat-7:31")
    request = parse_bot_chain_message(event.text)

    response = asyncio.run(
        runner._handle_bot_chain_turn(event, session, session.session_key, request)
    )

    assert response is not None and "resend" in response
    assert run_calls == []
