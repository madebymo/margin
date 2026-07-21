"""Add bounded replay, retention, token, and receipt lookup indexes.

Revision ID: 20260721_0004
Revises: 20260721_0003
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260721_0004"
down_revision: str | None = "20260721_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_index(
    table_name: str,
    index_name: str,
    columns: list[str],
) -> None:
    existing = {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    }
    if index_name not in existing:
        op.create_index(index_name, table_name, columns, unique=False)


def upgrade() -> None:
    _create_index(
        "evidence_events",
        "ix_evidence_learner_time",
        ["learner_id", "t", "id"],
    )
    _create_index("evidence_events", "ix_evidence_episode", ["episode_id", "id"])
    _create_index(
        "resume_tokens",
        "ix_resume_tokens_expiry_revoked",
        ["expires_at", "revoked"],
    )
    _create_index("resume_tokens", "ix_resume_tokens_session", ["session_id"])
    _create_index(
        "session_checkpoints",
        "ix_session_checkpoint_learner_started",
        ["learner_id", "started_at"],
    )
    _create_index(
        "session_checkpoints", "ix_session_checkpoint_updated", ["updated_at"]
    )
    _create_index(
        "session_mutation_receipts",
        "ix_session_receipt_request",
        ["request_id"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "the production schema is forward-only; restore a verified backup to downgrade"
    )
