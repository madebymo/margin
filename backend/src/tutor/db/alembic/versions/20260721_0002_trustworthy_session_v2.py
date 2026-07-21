"""Add atomic session-v2 ledgers and legacy-labelled evidence fields.

Revision ID: 20260721_0002
Revises: 20260721_0001
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260721_0002"
down_revision: str | None = "20260721_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_type() -> sa.TypeEngine:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _columns(table_name: str) -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _add_column_if_missing(table_name: str, column: sa.Column) -> bool:
    if column.name in _columns(table_name):
        return False
    op.add_column(table_name, column)
    return True


def _create_or_validate(table_name: str, *elements: sa.SchemaItem) -> None:
    inspector = sa.inspect(op.get_bind())
    if table_name not in inspector.get_table_names():
        op.create_table(table_name, *elements)
        return
    expected = {
        element.name for element in elements if isinstance(element, sa.Column)
    }
    missing = sorted(expected - _columns(table_name))
    if missing:
        raise RuntimeError(
            f"cannot adopt session-v2 table {table_name!r}; missing columns: {missing}"
        )


def upgrade() -> None:
    for column in (
        sa.Column("episode_id", sa.String(length=36), nullable=True),
        sa.Column("family_id", sa.String(length=128), nullable=True),
        sa.Column(
            "surface",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'legacy'"),
        ),
        sa.Column(
            "item_revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column(
            "attempt_number", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column(
            "policy_version",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'legacy'"),
        ),
        sa.Column(
            "learner_params_version",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'v1'"),
        ),
        sa.Column(
            "content_provenance",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("'legacy'"),
        ),
        sa.Column(
            "learning_opportunity",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    ):
        _add_column_if_missing("evidence_events", column)

    _create_or_validate(
        "session_checkpoints",
        sa.Column("session_id", sa.String(length=36), primary_key=True),
        sa.Column("learner_id", sa.String(length=36), nullable=False),
        sa.Column("goal_id", sa.String(length=128), nullable=False),
        sa.Column("target_kc", sa.String(length=128), nullable=False),
        sa.Column("profile", _json_type(), nullable=False),
        sa.Column("requested_content_mode", sa.String(length=32), nullable=False),
        sa.Column("effective_content_mode", sa.String(length=32), nullable=False),
        sa.Column("fallback_reason", sa.Text(), nullable=True),
        sa.Column(
            "revision", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("phase", sa.String(length=32), nullable=False),
        sa.Column("checkpoint", _json_type(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["learner_id"],
            ["learners.learner_id"],
            name="fk_session_checkpoints_learner",
        ),
    )
    _create_or_validate(
        "session_mutation_receipts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("request_payload", _json_type(), nullable=False),
        sa.Column("response_payload", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session_checkpoints.session_id"],
            name="fk_session_receipts_checkpoint",
        ),
        sa.UniqueConstraint(
            "session_id", "request_id", name="uq_session_receipt_request"
        ),
    )
    _create_or_validate(
        "transcript_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("entry", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session_checkpoints.session_id"],
            name="fk_transcript_checkpoint",
        ),
        sa.UniqueConstraint(
            "session_id", "sequence", name="uq_transcript_session_sequence"
        ),
    )
    _create_or_validate(
        "item_exposures",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("item_id", sa.String(length=128), nullable=False),
        sa.Column(
            "item_revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column(
            "variant_id",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("'base'"),
        ),
        sa.Column("family_id", sa.String(length=128), nullable=False),
        sa.Column("surface", sa.String(length=32), nullable=False),
        sa.Column("exposure_sequence", sa.Integer(), nullable=False),
        sa.Column(
            "solution_exposed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "hint_level", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "answer_revealed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session_checkpoints.session_id"],
            name="fk_item_exposures_checkpoint",
        ),
        sa.UniqueConstraint(
            "session_id",
            "exposure_sequence",
            name="uq_exposure_session_sequence",
        ),
    )
    _create_or_validate(
        "widget_attempts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("interaction_key", sa.String(length=128), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("response", _json_type(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session_checkpoints.session_id"],
            name="fk_widget_attempts_checkpoint",
        ),
        sa.UniqueConstraint(
            "session_id",
            "interaction_key",
            "attempt_number",
            name="uq_widget_attempt_interaction_attempt",
        ),
    )

    if "session_id" not in _columns("resume_tokens"):
        if op.get_bind().dialect.name == "sqlite":
            # SQLite cannot add a foreign-key constraint in place. Batch mode
            # copies the table transactionally and retains every legacy row.
            with op.batch_alter_table("resume_tokens") as batch_op:
                batch_op.add_column(
                    sa.Column("session_id", sa.String(length=36), nullable=True)
                )
                batch_op.create_foreign_key(
                    "fk_resume_tokens_session",
                    "session_checkpoints",
                    ["session_id"],
                    ["session_id"],
                )
        else:
            op.add_column(
                "resume_tokens",
                sa.Column("session_id", sa.String(length=36), nullable=True),
            )
            op.create_foreign_key(
                "fk_resume_tokens_session",
                "resume_tokens",
                "session_checkpoints",
                ["session_id"],
                ["session_id"],
            )
    # Bind old resumable tokens once to the latest checkpoint available at
    # migration time. Tokens with no corresponding checkpoint remain legacy
    # and cannot be guessed into a v2 episode during restoration.
    op.execute(
        sa.text(
            "UPDATE resume_tokens SET session_id = ("
            "SELECT session_id FROM session_checkpoints "
            "WHERE session_checkpoints.learner_id = resume_tokens.learner_id "
            "ORDER BY session_checkpoints.updated_at DESC LIMIT 1"
            ") WHERE session_id IS NULL"
        )
    )

    _add_column_if_missing(
        "widget_attempts",
        sa.Column(
            "verification_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'incorrect'"),
        ),
    )
    _add_column_if_missing(
        "widget_attempts",
        sa.Column(
            "counted", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
    )


def downgrade() -> None:
    raise RuntimeError(
        "the production schema is forward-only; restore a verified backup to downgrade"
    )
