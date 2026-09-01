from pathlib import Path

import pytest
import yaml

from hermes_cli import profiles as profiles_mod
from hermes_cli.bot_profiles import (
    configure_bot_profile,
    create_bot_profile,
    get_bot_profile,
    list_bot_profiles,
    remove_bot_profile,
    rename_bot_profile,
    resolve_bot_chain,
)


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
    (home / ".env").write_text("TEST_API_KEY=secret\n", encoding="utf-8")
    (home / "auth.json").write_text('{"token": "test"}\n', encoding="utf-8")

    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _path: False)
    monkeypatch.setattr(profiles_mod, "_cleanup_gateway_service", lambda *_args: None)
    monkeypatch.setattr(profiles_mod, "_maybe_unregister_gateway_service", lambda *_args: None)
    monkeypatch.setattr(profiles_mod, "_stop_profile_backends", lambda *_args: None)
    monkeypatch.setattr(profiles_mod, "remove_wrapper_script", lambda *_args: False)
    monkeypatch.setattr(profiles_mod, "_migrate_honcho_profile_host", lambda *_args: None)
    monkeypatch.setattr(
        profiles_mod,
        "check_alias_collision",
        lambda _name: "wrapper disabled in test",
    )
    return home


def test_bot_profile_crud_persists_model_prompt_enabled_and_credentials(bot_home):
    created = create_bot_profile(
        "DeepSeek",
        provider="deepseek",
        model="deepseek-v4-flash",
        system_prompt="Solve carefully.",
        seed_skills=False,
    )

    assert created.name == "deepseek"
    assert created.provider == "deepseek"
    assert created.model == "deepseek-v4-flash"
    assert created.system_prompt == "Solve carefully."
    assert created.enabled is True
    assert (created.path / ".env").read_text() == "TEST_API_KEY=secret\n"
    assert (created.path / "auth.json").is_file()

    config = yaml.safe_load((created.path / "config.yaml").read_text())
    assert config["model"]["provider"] == "deepseek"
    assert config["model"]["default"] == "deepseek-v4-flash"
    metadata = yaml.safe_load((created.path / "profile.yaml").read_text())
    assert metadata["bot"]["enabled"] is True

    configured = configure_bot_profile(
        "deepseek",
        model="deepseek-v4",
        system_prompt="Reason before answering.",
        enabled=False,
    )
    assert configured.model == "deepseek-v4"
    assert configured.system_prompt == "Reason before answering."
    assert configured.enabled is False

    renamed = rename_bot_profile("deepseek", "researcher")
    assert renamed.name == "researcher"
    assert renamed.enabled is False
    assert not (bot_home / "profiles" / "deepseek").exists()

    removed = remove_bot_profile("researcher", confirmed=True)
    assert removed.name == "researcher"
    assert not removed.exists()


def test_list_and_resolution_include_default_and_reject_unknown_or_disabled(bot_home):
    create_bot_profile(
        "worker",
        provider="openrouter",
        model="vendor/model",
        system_prompt="Work.",
        seed_skills=False,
    )

    assert [profile.name for profile in list_bot_profiles()] == ["default", "worker"]
    assert [profile.name for profile in resolve_bot_chain(["Worker", "Default"])] == [
        "worker",
        "default",
    ]

    with pytest.raises(ValueError, match=r"Unknown bot '\$missing'.*\$default.*\$worker"):
        resolve_bot_chain(["missing"])

    configure_bot_profile("worker", enabled=False)
    with pytest.raises(ValueError, match=r"Bot '\$worker' is disabled"):
        resolve_bot_chain(["worker"])


def test_remove_requires_explicit_confirmation(bot_home):
    create_bot_profile(
        "worker",
        provider="nous",
        model="test/model",
        system_prompt="Work.",
        seed_skills=False,
    )

    with pytest.raises(ValueError, match="without --yes"):
        remove_bot_profile("worker")
    assert get_bot_profile("worker").name == "worker"
