import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.bot_chain import (
    BOT_CHAIN_USAGE,
    BOT_CHAIN_CONVERSATION_PREFIX,
    BotChainCancelled,
    BotChainControl,
    BotRuntimeUnavailable,
    BotChainRunner,
    BotChainSyntaxError,
    FallbackBotTurnExecutor,
    HermesProfileTurnExecutor,
    HermesSessionRpcTurnExecutor,
    BotTurnError,
    _SessionRPCClient,
    default_bot_turn_executor,
    format_bot_chain_result,
    parse_bot_chain_message,
    publish_bot_chain_history,
)
from hermes_cli.bot_profiles import BotProfile


def _profile(name: str) -> BotProfile:
    return BotProfile(
        name=name,
        path=Path("/tmp") / name,
        model=f"model-{name}",
        provider="test",
        system_prompt=f"You are {name}",
    )


def _event(session_id: str, payload: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "message.complete",
            "session_id": session_id,
            "payload": payload,
        },
    }


def _fake_rpc_server(prompt_handler):
    """Behavioral fake around the real transport binding contract."""
    from tui_gateway.transport import current_transport

    proof = object()
    calls = []
    state = {
        "transport": None,
        "closed": 0,
        "interrupted": 0,
        "running": False,
        "status_sequence": [],
    }

    def create(rid, params):
        calls.append(("session.create", dict(params)))
        state["transport"] = current_transport()
        return {"jsonrpc": "2.0", "id": rid, "result": {"session_id": "rpc-1"}}

    def submit(rid, params):
        calls.append(("prompt.submit", dict(params)))
        assert current_transport() is state["transport"]
        return prompt_handler(rid, params, state)

    def interrupt(rid, params):
        calls.append(("session.interrupt", dict(params)))
        state["interrupted"] += 1
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "interrupted"}}

    def status(rid, params):
        calls.append(("session.status", dict(params)))
        sequence = state["status_sequence"]
        running = sequence.pop(0) if sequence else state["running"]
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"running": running, "turn_settled": not running},
        }

    def compress(rid, params):
        calls.append(("session.compress", dict(params)))
        result = state.get("compression_result") or {"status": "compressed"}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def close(rid, params):
        calls.append(("session.close", dict(params)))
        state["closed"] += 1
        return {"jsonrpc": "2.0", "id": rid, "result": {"closed": True}}

    server = SimpleNamespace(
        _IN_PROCESS_SINGLE_QUERY_PROOF=proof,
        _methods={
            "session.create": create,
            "session.compress": compress,
            "prompt.submit": submit,
            "session.status": status,
            "session.interrupt": interrupt,
            "session.close": close,
        },
    )
    return server, calls, state, proof


def test_parse_leading_bot_chain_and_prompt():
    request = parse_bot_chain_message(
        "  $Writer $Reviewer complete the task\nwith all relevant details"
    )

    assert request is not None
    assert request.names == ("Writer", "Reviewer")
    assert request.prompt == "complete the task\nwith all relevant details"


def test_parse_returns_none_for_ordinary_chat():
    assert parse_bot_chain_message("Explain $PATH") is None


@pytest.mark.parametrize("message", ["$", "$DeepSeek", "$DeepSeek   ", "$bad,name task"])
def test_parse_invalid_or_empty_chain_shows_usage(message):
    with pytest.raises(BotChainSyntaxError, match="Usage") as exc:
        parse_bot_chain_message(message)
    assert str(exc.value) == BOT_CHAIN_USAGE


def test_runner_preserves_order_and_hands_output_to_next_bot():
    calls = []

    def execute(profile, prompt, control, *, conversation_name):
        calls.append((profile.name, prompt, conversation_name))
        return f"output-{profile.name}"

    profiles = [_profile("first"), _profile("second"), _profile("third")]
    result = BotChainRunner(turn_executor=execute).run(profiles, "original task")

    assert [name for name, _prompt, _conversation in calls] == [
        "first",
        "second",
        "third",
    ]
    assert calls[0][1] == "original task"
    assert "Original user request:\noriginal task" in calls[1][1]
    assert "Previous bot ($first) output:\noutput-first" in calls[1][1]
    assert "Previous bot ($second) output:\noutput-second" in calls[2][1]
    assert len({conversation for _name, _prompt, conversation in calls}) == 1
    assert calls[0][2].startswith(BOT_CHAIN_CONVERSATION_PREFIX)
    assert calls[0][2] != "Bot Chat"
    assert result.final_output == "output-third"
    assert format_bot_chain_result(result).endswith("$third (final):\noutput-third")


def test_runner_stops_before_downstream_bot_after_failure():
    calls = []

    def execute(profile, prompt, control, *, conversation_name):
        calls.append(profile.name)
        if profile.name == "second":
            raise BotTurnError(profile.name, "provider unavailable", reason="server")
        return f"output-{profile.name}"

    profiles = [_profile("first"), _profile("second"), _profile("third")]

    with pytest.raises(BotTurnError, match=r"\$second failed"):
        BotChainRunner(turn_executor=execute).run(profiles, "task")

    assert calls == ["first", "second"]


def test_runner_honors_preexisting_cancellation():
    control = BotChainControl()
    control.interrupt("stop")

    with pytest.raises(BotChainCancelled, match="stopped"):
        BotChainRunner(turn_executor=lambda *_args, **_kwargs: "unused").run(
            [_profile("first")], "task", control=control
        )


def test_parallel_chain_runs_receive_distinct_conversations():
    conversations = []
    conversations_lock = threading.Lock()
    both_running = threading.Barrier(2)

    def execute(profile, prompt, control, *, conversation_name):
        with conversations_lock:
            conversations.append(conversation_name)
        both_running.wait(timeout=5)
        return f"output-{profile.name}"

    runner = BotChainRunner(turn_executor=execute)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda prompt: runner.run([_profile("worker")], prompt),
                ("first task", "second task"),
            )
        )

    assert [result.final_output for result in results] == [
        "output-worker",
        "output-worker",
    ]
    assert len(conversations) == 2
    assert len(set(conversations)) == 2
    assert all(
        conversation.startswith(BOT_CHAIN_CONVERSATION_PREFIX)
        for conversation in conversations
    )


def test_rpc_executor_uses_warm_session_transport_and_closes_runtime(monkeypatch):
    def submit(rid, params, state):
        state["transport"].write(
            _event(params["session_id"], {"status": "complete", "text": "warm result"})
        )
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, state, proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))
    monkeypatch.setattr(
        "tools.bot_relay.acquire_turn_lock",
        lambda *_args, **_kwargs: pytest.fail("RPC turns must not take the profile lock"),
    )

    output = executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain exact",
    )

    assert output == "warm result"
    assert [method for method, _params in calls] == [
        "session.create",
        "prompt.submit",
        "session.status",
        "session.close",
    ]
    create_params = calls[0][1]
    assert create_params["profile"] == "worker"
    assert create_params["title"] == "Bot Chain exact"
    assert create_params["hidden"] is True
    assert create_params["follow_profile_config"] is True
    assert create_params["_single_query_proof"] is proof
    assert state["closed"] == 1


def test_rpc_executor_retries_transient_error_in_same_runtime_session():
    attempts = []

    def submit(rid, params, state):
        attempts.append(params["session_id"])
        payload = (
            {
                "status": "error",
                "text": "",
                "error": "Error code: 429 - rate limit exceeded",
                "error_surface": {"code": "rate_limit"},
            }
            if len(attempts) == 1
            else {"status": "complete", "text": "recovered"}
        )
        state["transport"].write(_event(params["session_id"], payload))
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, _calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain retry",
    ) == "recovered"
    assert attempts == ["rpc-1", "rpc-1"]


def test_rpc_executor_compresses_context_before_same_session_retry():
    attempts = []

    def submit(rid, params, state):
        attempts.append(params["session_id"])
        payload = (
            {
                "status": "error",
                "error": "maximum context length exceeded",
                "error_surface": {"code": "context_overflow"},
            }
            if len(attempts) == 1
            else {"status": "complete", "text": "compressed result"}
        )
        state["transport"].write(_event(params["session_id"], payload))
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain context",
    ) == "compressed result"
    assert attempts == ["rpc-1", "rpc-1"]
    methods = [method for method, _params in calls]
    assert methods.index("session.compress") < methods.index("prompt.submit", 2)


def test_rpc_executor_does_not_retry_unchanged_context_after_aborted_compression():
    attempts = 0

    def submit(rid, params, state):
        nonlocal attempts
        attempts += 1
        state["compression_result"] = {
            "status": "aborted",
            "message": "not enough history to compress",
        }
        state["transport"].write(
            _event(
                params["session_id"],
                {
                    "status": "error",
                    "error": "maximum context length exceeded",
                    "error_surface": {"code": "context_overflow"},
                },
            )
        )
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    with pytest.raises(BotTurnError) as exc:
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain context-abort",
        )

    assert exc.value.reason == "context_overflow"
    assert attempts == 1
    assert [method for method, _params in calls].count("session.compress") == 1


def test_rpc_executor_waits_for_terminal_turn_to_settle_before_close():
    def submit(rid, params, state):
        state["status_sequence"][:] = [True, False]
        state["transport"].write(
            _event(params["session_id"], {"status": "complete", "text": "settled"})
        )
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, calls, _state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain settle",
    ) == "settled"
    assert [method for method, _params in calls].count("session.status") == 2
    assert [method for method, _params in calls][-1] == "session.close"


def test_rpc_session_owner_refusal_is_typed_and_never_replayed():
    def submit(rid, _params, _state):
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {
                "code": 4090,
                "message": "Session already has a live owner",
                "data": {"reason": "SESSION_NOT_OWNED"},
            },
        }

    server, calls, state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))

    with pytest.raises(BotTurnError) as exc:
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain busy",
        )

    assert exc.value.reason == "session_busy"
    assert [method for method, _params in calls].count("prompt.submit") == 1
    assert state["closed"] == 1


def test_rpc_executor_interrupts_and_closes_cancelled_runtime():
    started = threading.Event()

    def submit(rid, _params, _state):
        started.set()
        return {"jsonrpc": "2.0", "id": rid, "result": {"status": "streaming"}}

    server, _calls, state, _proof = _fake_rpc_server(submit)
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))
    control = BotChainControl()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor,
            _profile("worker"),
            "do it",
            control,
            conversation_name="Bot Chain cancel",
        )
        assert started.wait(timeout=2)
        control.interrupt("stop")
        with pytest.raises(BotChainCancelled):
            future.result(timeout=3)

    assert state["interrupted"] == 1
    assert state["closed"] == 1


def test_fallback_runs_only_for_pre_admission_runtime_unavailable():
    calls = []

    def primary(*_args, **_kwargs):
        calls.append("primary")
        raise BotRuntimeUnavailable("RPC missing")

    def fallback(*_args, **_kwargs):
        calls.append("fallback")
        return "legacy result"

    executor = FallbackBotTurnExecutor(primary, fallback)
    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain fallback",
    ) == "legacy result"
    assert calls == ["primary", "fallback"]


def test_fallback_never_replays_an_admitted_turn_failure():
    fallback_called = False

    def primary(*_args, **_kwargs):
        raise BotTurnError("worker", "provider failed")

    def fallback(*_args, **_kwargs):
        nonlocal fallback_called
        fallback_called = True
        return "must not run"

    executor = FallbackBotTurnExecutor(primary, fallback)
    with pytest.raises(BotTurnError):
        executor(
            _profile("worker"),
            "do it",
            BotChainControl(),
            conversation_name="Bot Chain admitted",
        )
    assert fallback_called is False


def test_default_executor_is_process_wide_and_rpc_first():
    first = default_bot_turn_executor()
    second = default_bot_turn_executor()

    assert first is second
    assert isinstance(first, FallbackBotTurnExecutor)
    assert isinstance(first.primary, HermesSessionRpcTurnExecutor)
    assert isinstance(first.fallback, HermesProfileTurnExecutor)
    assert first.history_publisher is publish_bot_chain_history


def test_history_projection_promotes_first_isolated_turn_to_bot_chat(tmp_path):
    from hermes_state import SessionDB

    profile_dir = tmp_path / "hermes" / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db = SessionDB(profile_dir / "state.db")
    try:
        db.create_session(
            "chain-first",
            source="cli",
            model="test/model",
            model_config={"follow_profile_config": True},
            profile_name="worker",
        )
        db.set_session_title("chain-first", "Bot Chain first")
        db.set_session_hidden("chain-first", True)
        db.append_messages_batch(
            "chain-first",
            [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "first answer"},
            ],
        )
    finally:
        db.close()

    canonical_id = publish_bot_chain_history(
        BotProfile(
            name="worker",
            path=profile_dir,
            model="test/model",
            provider="test",
            system_prompt="Work",
        ),
        "Bot Chain first",
    )

    db = SessionDB(profile_dir / "state.db")
    try:
        canonical = db.get_session_by_title("Bot Chat")
        assert canonical is not None
        assert canonical["id"] == "chain-first"
        assert canonical["hidden"] == 1
        assert canonical_id == "chain-first"
        assert [
            (message["role"], message["content"])
            for message in db.get_messages_as_conversation("chain-first")
        ] == [
            ("user", "first prompt"),
            ("assistant", "first answer"),
        ]
        assert db.get_session_by_title("Bot Chain first") is None
    finally:
        db.close()


def test_history_projection_appends_to_existing_bot_chat_despite_desktop_owner(
    tmp_path,
):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB

    profile_dir = tmp_path / "hermes" / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    db = SessionDB(profile_dir / "state.db")
    try:
        for session_id, title in (
            ("canonical", "Bot Chat"),
            ("chain-next", "Bot Chain next"),
        ):
            db.create_session(
                session_id,
                source="desktop" if session_id == "canonical" else "cli",
                model="test/model",
                model_config={"follow_profile_config": True},
                profile_name="worker",
            )
            db.set_session_title(session_id, title)
            db.set_session_hidden(session_id, True)
        db.append_messages_batch(
            "canonical",
            [
                {"role": "user", "content": "older prompt"},
                {"role": "assistant", "content": "older answer"},
            ],
        )
        db.append_messages_batch(
            "chain-next",
            [
                {"role": "user", "content": "new prompt"},
                {"role": "assistant", "content": "new answer"},
            ],
        )
    finally:
        db.close()

    desktop_lease, refusal = try_acquire_active_session(
        session_id="canonical",
        surface="desktop",
        config={},
        metadata={"live_session_id": "desktop-owner"},
        registry_home=profile_dir,
    )
    assert desktop_lease is not None and refusal is None
    try:
        canonical_id = publish_bot_chain_history(
            BotProfile(
                name="worker",
                path=profile_dir,
                model="test/model",
                provider="test",
                system_prompt="Work",
            ),
            "Bot Chain next",
        )
    finally:
        desktop_lease.release()

    db = SessionDB(profile_dir / "state.db")
    try:
        assert canonical_id == "canonical"
        assert [
            (message["role"], message["content"])
            for message in db.get_messages_as_conversation("canonical")
        ] == [
            ("user", "older prompt"),
            ("assistant", "older answer"),
            ("user", "new prompt"),
            ("assistant", "new answer"),
        ]
        assert db.get_session_by_title("Bot Chain next")["id"] == "chain-next"
    finally:
        db.close()


def test_history_projection_failure_does_not_replay_completed_turn():
    published = []

    def primary(*_args, **_kwargs):
        return "completed once"

    def fallback(*_args, **_kwargs):
        raise AssertionError("completed primary turn must not be replayed")

    def publisher(*_args, **_kwargs):
        published.append("attempted")
        raise RuntimeError("projection unavailable")

    executor = FallbackBotTurnExecutor(
        primary,
        fallback,
        history_publisher=publisher,
    )

    assert executor(
        _profile("worker"),
        "do it",
        BotChainControl(),
        conversation_name="Bot Chain completed",
    ) == "completed once"
    assert published == ["attempted"]


def test_live_bot_chat_owner_does_not_block_isolated_chain_turn(
    tmp_path,
    monkeypatch,
):
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_cli.main import (
        _create_titled_session,
        _resolve_session_by_name_or_id,
    )

    root = tmp_path / "hermes"
    profile_dir = root / "profiles" / "test1"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    bot_chat_session_id = _create_titled_session("Bot Chat")
    assert bot_chat_session_id
    desktop_lease, error = try_acquire_active_session(
        session_id=bot_chat_session_id,
        surface="desktop",
        config={},
        metadata={"live_session_id": "desktop-owner"},
    )
    assert desktop_lease is not None and error is None

    attempted_conversations = []

    def fake_run_once(self, argv, control):
        conversation_name = argv[argv.index("-c") + 1]
        attempted_conversations.append(conversation_name)
        session_id = _resolve_session_by_name_or_id(conversation_name)
        if session_id is None and "--create-if-missing" in argv:
            session_id = _create_titled_session(conversation_name)
        assert session_id
        chain_lease, refusal = try_acquire_active_session(
            session_id=session_id,
            surface="cli",
            config={},
            metadata={"live_session_id": "chain-owner"},
        )
        if chain_lease is None:
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr=str(refusal),
            )
        chain_lease.release()
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="isolated result",
            stderr="",
        )

    monkeypatch.setattr(HermesProfileTurnExecutor, "_run_once", fake_run_once)
    profile = BotProfile(
        name="test1",
        path=profile_dir,
        model="test/model",
        provider="test",
        system_prompt="Work",
    )

    try:
        result = BotChainRunner(turn_executor=HermesProfileTurnExecutor()).run(
            [profile], "task"
        )
    finally:
        desktop_lease.release()

    assert result.final_output == "isolated result"
    assert len(attempted_conversations) == 1
    assert attempted_conversations[0].startswith(BOT_CHAIN_CONVERSATION_PREFIX)
    assert attempted_conversations[0] != "Bot Chat"


def test_local_delivery_default_remains_canonical_bot_chat():
    from tools.bot_relay import local_delivery_command

    argv = local_delivery_command("test1", "/tmp/query.txt")

    assert argv[argv.index("-c") + 1] == "Bot Chat"
