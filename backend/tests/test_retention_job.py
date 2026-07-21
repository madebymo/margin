"""Bounded scheduler entry point for anonymous-session retention."""

from __future__ import annotations

from tutor.api.v2_persistence import RetentionBatch
from tutor.db.retention_job import main, run_retention_batches


class FakePersistence:
    def __init__(self, batches: list[RetentionBatch]) -> None:
        self._batches = iter(batches)
        self.calls: list[tuple[int, str | None]] = []

    def purge_expired_anonymous_sessions_batch(
        self,
        *,
        limit: int,
        after_session_id: str | None,
    ) -> RetentionBatch:
        self.calls.append((limit, after_session_id))
        return next(self._batches)


def test_scheduler_follows_cursor_and_stops_at_complete_page():
    persistence = FakePersistence(
        [
            RetentionBatch(
                purged=2,
                scanned=2,
                complete=False,
                next_cursor="internal-session-cursor",
            ),
            RetentionBatch(purged=1, scanned=1, complete=True),
        ]
    )

    result = run_retention_batches(
        persistence,
        batch_size=2,
        max_batches=5,
    )

    assert result == (3, 2, True)
    assert persistence.calls == [
        (2, None),
        (2, "internal-session-cursor"),
    ]


def test_scheduler_stops_at_hard_batch_limit():
    persistence = FakePersistence(
        [
            RetentionBatch(
                purged=1,
                scanned=1,
                complete=False,
                next_cursor=f"cursor-{index}",
            )
            for index in range(2)
        ]
    )

    assert run_retention_batches(
        persistence,
        batch_size=1,
        max_batches=2,
    ) == (2, 2, False)
    assert len(persistence.calls) == 2


def test_cli_requires_explicit_database_url(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "retention job INVALID: DATABASE_URL is required\n"


def test_retention_batch_repr_excludes_internal_cursor():
    batch = RetentionBatch(
        purged=1,
        scanned=1,
        complete=False,
        next_cursor="private-session-id",
    )

    assert "private-session-id" not in repr(batch)
    assert batch.next_cursor == "private-session-id"
