"""Tests for SessionDB bot-chain delivery admission (#100758 review).

A ``$Bot`` chain triggered by a platform message performs model turns with
durable side effects. At-least-once platform delivery must therefore meet an
idempotent recipient: admission reserves one stable chain identity, while an
atomic running claim decides which live attempt may execute it. A dead owner
can be reclaimed under that same identity; a live owner and a settled receipt
can never be claimed twice.
"""

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

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
        token = db.mark_bot_chain_delivery_running("sess-chain", "tg-2")
        assert token
        assert db.settle_bot_chain_delivery(
            "sess-chain", "tg-2", outcome="completed", owner_token=token
        )
        assert db.admit_bot_chain_delivery("sess-chain", "tg-2", "Bot Chain b") == "settled"
        row = db.get_bot_chain_delivery("sess-chain", "tg-2")
        assert row["outcome"] == "completed"
        # The original chain identity stays bound to the receipt.
        assert row["chain_name"] == "Bot Chain a"

    def test_admitted_delivery_stays_retryable_and_keeps_chain_identity(self, db):
        """A crash before the running claim must not silently discard work."""
        assert db.admit_bot_chain_delivery("sess-chain", "tg-3", "Bot Chain a") == "admitted"
        assert db.admit_bot_chain_delivery("sess-chain", "tg-3", "Bot Chain b") == "admitted"
        row = db.get_bot_chain_delivery("sess-chain", "tg-3")
        assert row["state"] == "admitted"
        assert row["outcome"] is None
        assert row["chain_name"] == "Bot Chain a"

    def test_live_running_delivery_is_not_abandoned_or_reclaimed(self, db):
        """A concurrent redelivery must leave the original execution alone."""
        assert db.admit_bot_chain_delivery("sess-chain", "tg-4", "Bot Chain a") == "admitted"
        assert db.mark_bot_chain_delivery_running("sess-chain", "tg-4")
        assert db.admit_bot_chain_delivery("sess-chain", "tg-4", "Bot Chain b") == "running"
        row = db.get_bot_chain_delivery("sess-chain", "tg-4")
        assert row["state"] == "running"
        assert row["outcome"] is None
        assert row["chain_name"] == "Bot Chain a"

    def test_concurrent_running_claim_has_one_live_winner(self, db):
        """Concurrent redeliveries elect exactly one side-effect owner."""
        assert db.admit_bot_chain_delivery("sess-chain", "tg-claim", "Bot Chain a") == "admitted"
        barrier = Barrier(2)

        def claim():
            contender = SessionDB(db.db_path)
            try:
                barrier.wait()
                return contender.mark_bot_chain_delivery_running(
                    "sess-chain", "tg-claim"
                )
            finally:
                contender.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda _index: claim(), range(2)))

        assert sorted(bool(result) for result in claims) == [False, True]
        assert db.get_bot_chain_delivery("sess-chain", "tg-claim")["state"] == "running"

    def test_dead_running_owner_is_reclaimed_under_original_chain_name(self, db):
        """A process death after claiming execution must remain recoverable."""
        script = """
import sys
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(Path(sys.argv[1]))
try:
    assert db.admit_bot_chain_delivery(sys.argv[2], sys.argv[3], sys.argv[4]) == 'admitted'
    assert db.mark_bot_chain_delivery_running(sys.argv[2], sys.argv[3])
finally:
    db.close()
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(db.db_path),
                "sess-chain",
                "tg-dead",
                "Bot Chain original",
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )

        assert (
            db.admit_bot_chain_delivery(
                "sess-chain", "tg-dead", "Bot Chain replacement"
            )
            == "admitted"
        )
        row = db.get_bot_chain_delivery("sess-chain", "tg-dead")
        assert row["state"] == "admitted"
        assert row["outcome"] is None
        assert row["chain_name"] == "Bot Chain original"

    def test_running_marker_is_admitted_only(self, db):
        assert db.admit_bot_chain_delivery("sess-chain", "tg-5", "Bot Chain a") == "admitted"
        token = db.mark_bot_chain_delivery_running("sess-chain", "tg-5")
        assert token
        assert db.get_bot_chain_delivery("sess-chain", "tg-5")["state"] == "running"
        # A second marker transition is a no-op once settled.
        assert db.settle_bot_chain_delivery(
            "sess-chain", "tg-5", outcome="failed", detail="boom", owner_token=token
        )
        assert db.mark_bot_chain_delivery_running("sess-chain", "tg-5") is None
        assert db.get_bot_chain_delivery("sess-chain", "tg-5")["state"] == "settled"

    def test_admissions_are_scoped_per_session(self, db):
        db.create_session("sess-other", source="cli")
        assert db.admit_bot_chain_delivery("sess-chain", "tg-6", "Bot Chain a") == "admitted"
        assert db.admit_bot_chain_delivery("sess-other", "tg-6", "Bot Chain b") == "admitted"

    def test_get_delivery_without_table_returns_none(self, db):
        assert db.get_bot_chain_delivery("sess-chain", "never-admitted") is None

    def test_release_returns_own_running_claim_to_admitted(self, db):
        """Settlement-write failure recovery: the owner releases its claim so
        a redelivery resumes the admission instead of standing down until
        the lease lapses."""
        assert db.admit_bot_chain_delivery("sess-chain", "tg-r1", "Bot Chain a") == "admitted"
        token = db.mark_bot_chain_delivery_running("sess-chain", "tg-r1")
        assert token
        assert db.release_bot_chain_delivery_claim("sess-chain", "tg-r1", token)
        row = db.get_bot_chain_delivery("sess-chain", "tg-r1")
        assert row["state"] == "admitted"
        # The chain identity survives the release for the resumed redelivery.
        assert row["chain_name"] == "Bot Chain a"
        # A released admission is claimable again by the resumed attempt.
        assert db.mark_bot_chain_delivery_running("sess-chain", "tg-r1")

    def test_release_never_touches_admitted_or_settled_rows(self, db):
        # admitted row: nothing to release
        assert db.admit_bot_chain_delivery("sess-chain", "tg-r2", "Bot Chain a") == "admitted"
        assert not db.release_bot_chain_delivery_claim("sess-chain", "tg-r2", "no-such-token")
        assert db.get_bot_chain_delivery("sess-chain", "tg-r2")["state"] == "admitted"

        # settled row: terminal, untouched
        token = db.mark_bot_chain_delivery_running("sess-chain", "tg-r2")
        assert token
        assert db.settle_bot_chain_delivery(
            "sess-chain", "tg-r2", outcome="completed", owner_token=token
        )
        assert not db.release_bot_chain_delivery_claim("sess-chain", "tg-r2", token)
        row = db.get_bot_chain_delivery("sess-chain", "tg-r2")
        assert row["state"] == "settled"
        assert row["outcome"] == "completed"

    def test_release_cannot_revoke_a_foreign_owner_claim(self, db):
        """Owner-scoped release: another live process's claim is untouchable."""
        script = """
import sys
from pathlib import Path
from hermes_state import SessionDB

db = SessionDB(Path(sys.argv[1]))
try:
    assert db.admit_bot_chain_delivery(sys.argv[2], sys.argv[3], sys.argv[4]) == 'admitted'
    assert db.mark_bot_chain_delivery_running(sys.argv[2], sys.argv[3])
    print("claimed", flush=True)
    import time
    time.sleep(60)
finally:
    db.close()
"""
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(db.db_path),
                "sess-chain",
                "tg-foreign",
                "Bot Chain foreign",
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            # Bounded wait for the child's claim: a bare readline() could
            # block this test forever if the child dies before printing.
            claimed_line = []

            def _read_claim():
                claimed_line.append(child.stdout.readline())

            reader = threading.Thread(target=_read_claim, daemon=True)
            reader.start()
            reader.join(timeout=30)
            assert claimed_line, "foreign-owner child did not claim within 30s"
            assert claimed_line[0].strip() == "claimed"
            assert not db.release_bot_chain_delivery_claim(
                "sess-chain", "tg-foreign", "not-the-childs-token"
            )
            row = db.get_bot_chain_delivery("sess-chain", "tg-foreign")
            assert row["state"] == "running"
            assert row["chain_name"] == "Bot Chain foreign"
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_expired_lease_from_prior_runtime_generation_reclaims_exactly_once(
        self, db
    ):
        """A running claim whose runtime generation is gone (container/host
        identity changed, state directory restored — anything a pid+host pair
        can never disprove) becomes reclaimable when its lease expires, and
        exactly one redelivery wins the resumed claim under the ORIGINAL
        chain identity."""
        assert (
            db.admit_bot_chain_delivery("sess-chain", "tg-stale", "Bot Chain original")
            == "admitted"
        )
        old_token = db.mark_bot_chain_delivery_running(
            "sess-chain", "tg-stale", lease_seconds=-1
        )
        assert old_token
        row = db.get_bot_chain_delivery("sess-chain", "tg-stale")
        assert row["state"] == "running"
        assert row["lease_expires_at"] is not None

        # The expired lease makes the prior generation's claim reclaimable
        # even though its pid/host can never be probed again.
        assert (
            db.admit_bot_chain_delivery("sess-chain", "tg-stale", "Bot Chain replacement")
            == "admitted"
        )
        row = db.get_bot_chain_delivery("sess-chain", "tg-stale")
        assert row["state"] == "admitted"
        assert row["chain_name"] == "Bot Chain original"

        # Exactly one resumed attempt wins the new claim.
        new_token = db.mark_bot_chain_delivery_running("sess-chain", "tg-stale")
        assert new_token
        assert new_token != old_token
        assert db.mark_bot_chain_delivery_running("sess-chain", "tg-stale") is None

        # The stale generation can neither settle nor release the new claim.
        assert not db.settle_bot_chain_delivery(
            "sess-chain", "tg-stale", outcome="completed", owner_token=old_token
        )
        assert not db.release_bot_chain_delivery_claim(
            "sess-chain", "tg-stale", old_token
        )
        # The resumed owner settles terminally.
        assert db.settle_bot_chain_delivery(
            "sess-chain", "tg-stale", outcome="completed", owner_token=new_token
        )
        row = db.get_bot_chain_delivery("sess-chain", "tg-stale")
        assert row["state"] == "settled"
        assert row["outcome"] == "completed"
        assert row["chain_name"] == "Bot Chain original"

    def test_unexpired_lease_from_live_foreign_runtime_stands_down(self, db):
        """The lease also protects a live owner this host cannot probe: an
        unexpired claim recorded under a different host identity is NOT
        reclaimed before its lease lapses."""
        assert (
            db.admit_bot_chain_delivery("sess-chain", "tg-live", "Bot Chain original")
            == "admitted"
        )
        token = db.mark_bot_chain_delivery_running("sess-chain", "tg-live")
        assert token
        # Simulate the same durable DB opened by another host identity.
        import sqlite3 as _sqlite3

        with _sqlite3.connect(db.db_path) as conn:
            conn.execute(
                "UPDATE bot_chain_deliveries SET owner_host = 'other-host' "
                "WHERE session_id = 'sess-chain' AND platform_message_id = 'tg-live'"
            )
        assert (
            db.admit_bot_chain_delivery("sess-chain", "tg-live", "Bot Chain replacement")
            == "running"
        )
        row = db.get_bot_chain_delivery("sess-chain", "tg-live")
        assert row["state"] == "running"
        assert row["chain_name"] == "Bot Chain original"

    def test_renew_extends_only_own_claim_lease(self, db):
        """Heartbeat: the owner extends its lease; a stale token renews nothing."""
        assert (
            db.admit_bot_chain_delivery("sess-chain", "tg-renew", "Bot Chain a")
            == "admitted"
        )
        token = db.mark_bot_chain_delivery_running(
            "sess-chain", "tg-renew", lease_seconds=30
        )
        assert token
        before = db.get_bot_chain_delivery("sess-chain", "tg-renew")[
            "lease_expires_at"
        ]
        assert not db.renew_bot_chain_delivery_claim(
            "sess-chain", "tg-renew", "stale-token", lease_seconds=120
        )
        assert db.renew_bot_chain_delivery_claim(
            "sess-chain", "tg-renew", token, lease_seconds=120
        )
        after = db.get_bot_chain_delivery("sess-chain", "tg-renew")[
            "lease_expires_at"
        ]
        assert after > before
        # A renewed live claim is not reclaimed.
        assert (
            db.admit_bot_chain_delivery("sess-chain", "tg-renew", "Bot Chain b")
            == "running"
        )
