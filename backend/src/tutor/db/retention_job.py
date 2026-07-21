"""Run bounded anonymous-session retention from an external scheduler."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Protocol

from sqlalchemy.engine import make_url

from tutor.api.v2_persistence import RetentionBatch, V2PersistenceService
from tutor.db.session import get_engine


class RetentionPersistence(Protocol):
    def purge_expired_anonymous_sessions_batch(
        self,
        *,
        limit: int,
        after_session_id: str | None,
    ) -> RetentionBatch: ...


def run_retention_batches(
    persistence: RetentionPersistence,
    *,
    batch_size: int = 100,
    max_batches: int = 10,
) -> tuple[int, int, bool]:
    """Run at most ``max_batches`` cursor pages and return sanitized totals."""

    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    if type(max_batches) is not int or not 1 <= max_batches <= 100:
        raise ValueError("max_batches must be between 1 and 100")
    cursor = None
    total_purged = 0
    for batch_number in range(1, max_batches + 1):
        result = persistence.purge_expired_anonymous_sessions_batch(
            limit=batch_size,
            after_session_id=cursor,
        )
        total_purged += result.purged
        if result.complete:
            return total_purged, batch_number, True
        cursor = result.next_cursor
        if cursor is None:  # defensive contract guard
            raise RuntimeError("an incomplete retention page has no cursor")
    return total_purged, max_batches, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args(argv)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("retention job INVALID: DATABASE_URL is required", file=sys.stderr)
        return 2
    try:
        if make_url(database_url).get_backend_name() != "postgresql":
            raise ValueError("retention job requires PostgreSQL")
        engine = get_engine(database_url)
        try:
            persistence = V2PersistenceService(engine)
            purged, batches, complete = run_retention_batches(
                persistence,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
            )
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001 - scheduled-job boundary
        print(
            f"retention job FAILED: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    status = "complete" if complete else "batch_limit_reached"
    print(
        f"retention job {status}: purged={purged} batches={batches}"
    )
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
