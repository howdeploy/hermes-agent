"""Topic-bound bots: a Telegram topic titled ``$Name`` routes to that bot.

Coverage for ``agent.bot_chain.bind_topic_bot`` (pure chain-request shaping)
and ``gateway.run.GatewayRunner._telegram_topic_bound_bot`` (topic title ->
bot profile resolution).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent.bot_chain import BotChainRequest, bind_topic_bot, parse_bot_chain_message
from gateway.config import Platform
from gateway.run import GatewayRunner


# ---------------------------------------------------------------- bind_topic_bot


def test_bind_topic_bot_plain_message_becomes_single_bot_chain():
    request = bind_topic_bot(None, "writer", "Draft a concise release note.")
    assert request is not None
    assert request.names == ("writer",)
    assert request.prompt == "Draft a concise release note."


def test_bind_topic_bot_prepends_bound_bot_to_explicit_chain():
    explicit = parse_bot_chain_message("$reviewer check the proposal")
    assert explicit is not None
    request = bind_topic_bot(explicit, "writer", "$reviewer check the proposal")
    assert request.names == ("writer", "reviewer")
    assert request.prompt == "check the proposal"


def test_bind_topic_bot_dedupes_bound_bot_case_insensitively():
    explicit = parse_bot_chain_message("$WRITER $reviewer improve this draft")
    assert explicit is not None
    request = bind_topic_bot(
        explicit, "writer", "$WRITER $reviewer improve this draft"
    )
    assert request.names == ("WRITER", "reviewer")
    assert request.prompt == "improve this draft"


def test_bind_topic_bot_no_bound_bot_returns_request_unchanged():
    assert bind_topic_bot(None, None, "ordinary message") is None
    explicit = parse_bot_chain_message("$writer draft a summary")
    assert bind_topic_bot(explicit, None, "$writer draft a summary") is explicit


# ---------------------------------------------- _telegram_topic_bound_bot


@pytest.fixture()
def bot_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: default/model\n",
        encoding="utf-8",
    )
    (home / "SOUL.md").write_text("Default system prompt\n", encoding="utf-8")

    def _make_profile(name, *, model="kimi-k3", provider="kimi-coding", enabled=True):
        profile_dir = home / "profiles" / name
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {"model": {"provider": provider, "default": model}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        if not enabled:
            (profile_dir / "profile.yaml").write_text(
                yaml.safe_dump({"bot": {"enabled": False}}),
                encoding="utf-8",
            )

    _make_profile("writer")
    _make_profile("reviewer", provider="deepseek", model="deepseek-v4-flash")
    _make_profile("offline", enabled=False)
    return home


def _runner():
    return object.__new__(GatewayRunner)


def test_topic_bound_bot_resolves_dollar_topic_to_bot(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$writer"
    )
    assert _runner()._telegram_topic_bound_bot(source) == "writer"


def test_topic_bound_bot_ignores_plain_topic_titles(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="General"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None


def test_topic_bound_bot_ignores_non_telegram_platforms(bot_home):
    source = SimpleNamespace(
        platform=Platform.SLACK, chat_type="dm", chat_topic="$writer"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None


def test_topic_bound_bot_rejects_unknown_bot(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$nobody"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None


def test_topic_bound_bot_rejects_disabled_bot(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$offline"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None


def test_topic_bound_bot_rejects_bare_dollar_topic(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None
