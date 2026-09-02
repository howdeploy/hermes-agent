"""Tests for SessionDB bot-chain delivery admission (#100758 review).

A ``$Bot`` chain triggered by a platform message performs model turns with
durable side effects. At-least-once platform delivery must therefore meet an
idempotent recipient: the ``bot_chain_deliveries`` admission row is written
BEFORE any model execution and decides whether a (re)delivered message may
start a chain. Redelivery of a settled or never-settled event never starts a
second execution.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-chain", source="cli")
    yield d
    d.close()


class TestBotChainDeliveryAdmission:
    def test_first_delivery_is_admitted_and_records_chain_identity(self, db):
        status = db.admit_bot_chain_delivery("sess-chain", "tg-1", "Bot Chain abc")
        assert status == "admitted"
        row = db.get_bot_chain_delivery("sess-chain", "tg-1")
        assert row["chain_name"] == "Bot Chain abc"
        assert row["state"] == "admitted"
        assert row["outcome"] is None

    def test_settled_delivery_is_never_readmitted(self, db):
        assert db.admit_bot_chain_delivery("sess-chain", "tg-2", "Bot Chain a") == "admitted"
        db.mark_bot_chain_delivery_running("sess-chain", "tg-2")
        db.settle_bot_chain_delivery("sess-chain", "tg-2", outcome="completed")
        assert db.admit_bot_chain_delivery("sess-chain", "tg-2", "Bot Chain b") == "settled"
        row = db.get_bot_chain_delivery("sess-chain", "tg-2")
        assert row["outcome"] == "completed"
        # The original chain identity stays bound to the receipt.
        assert row["chain_name"] == "Bot Chain a"

    def test_never_settled_delivery_is_reconciled_without_reexecution(self, db):
        """Crash between admission and settlement: redelivery must not rerun."""
        assert db.admit_bot_chain_delivery("sess-chain", "tg-3", "Bot Chain a") == "admitted"
        db.mark_bot_chain_delivery_running("sess-chain", "tg-3")
        # Process dies here; the platform redelivers the same message.
        assert db.admit_bot_chain_delivery("sess-chain", "tg-3", "Bot Chain b") == "reconciled"
        row = db.get_bot_chain_delivery("sess-chain", "tg-3")
        assert row["state"] == "settled"
        assert row["outcome"] == "abandoned"
        # And it stays deduped afterwards.
        assert db.admit_bot_chain_delivery("sess-chain", "tg-3", "Bot Chain c") == "settled"

    def test_settlement_overwrites_a_reconciled_abandoned_row(self, db):
        """A redelivery reconciled while the real execution still runs: the
        truthful outcome wins when the in-flight chain finally settles."""
        assert db.admit_bot_chain_delivery("sess-chain", "tg-4", "Bot Chain a") == "admitted"
        assert db.admit_bot_chain_delivery("sess-chain", "tg-4", "Bot Chain b") == "reconciled"
        db.settle_bot_chain_delivery("sess-chain", "tg-4", outcome="completed")
        row = db.get_bot_chain_delivery("sess-chain", "tg-4")
        assert row["state"] == "settled"
        assert row["outcome"] == "completed"

    def test_running_marker_is_admitted_only(self, db):
        assert db.admit_bot_chain_delivery("sess-chain", "tg-5", "Bot Chain a") == "admitted"
        db.mark_bot_chain_delivery_running("sess-chain", "tg-5")
        assert db.get_bot_chain_delivery("sess-chain", "tg-5")["state"] == "running"
        # A second marker transition is a no-op once settled.
        db.settle_bot_chain_delivery("sess-chain", "tg-5", outcome="failed", detail="boom")
        db.mark_bot_chain_delivery_running("sess-chain", "tg-5")
        assert db.get_bot_chain_delivery("sess-chain", "tg-5")["state"] == "settled"

    def test_admissions_are_scoped_per_session(self, db):
        db.create_session("sess-other", source="cli")
        assert db.admit_bot_chain_delivery("sess-chain", "tg-6", "Bot Chain a") == "admitted"
        assert db.admit_bot_chain_delivery("sess-other", "tg-6", "Bot Chain b") == "admitted"

    def test_get_delivery_without_table_returns_none(self, db):
        assert db.get_bot_chain_delivery("sess-chain", "never-admitted") is None
