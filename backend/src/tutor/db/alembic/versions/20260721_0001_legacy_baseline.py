"""Create or adopt the pre-v2 application schema.

Revision ID: 20260721_0001
Revises: None
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260721_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_type() -> sa.TypeEngine:
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _create_or_adopt(table_name: str, *elements: sa.SchemaItem) -> None:
    """Create a missing legacy table or validate the existing table's shape."""

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        op.create_table(table_name, *elements)
        return
    expected = {
        element.name for element in elements if isinstance(element, sa.Column)
    }
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    missing = sorted(expected - existing)
    if missing:
        raise RuntimeError(
            f"cannot adopt legacy table {table_name!r}; missing columns: {missing}"
        )


def upgrade() -> None:
    """Create the immutable legacy baseline while preserving existing rows."""

    _create_or_adopt(
        "graph_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("version", name="uq_graph_versions_version"),
    )
    _create_or_adopt(
        "kc_nodes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("graph_version_id", sa.Integer(), nullable=False),
        sa.Column("kc_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("course_level", sa.String(length=64), nullable=False),
        sa.Column("canonical_examples", _json_type(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_version_id"],
            ["graph_versions.id"],
            name="fk_kc_nodes_graph_version",
        ),
        sa.UniqueConstraint(
            "graph_version_id", "kc_id", name="uq_kc_nodes_graph_kc"
        ),
    )
    _create_or_adopt(
        "kc_edges",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("graph_version_id", sa.Integer(), nullable=False),
        sa.Column("from_kc", sa.String(length=128), nullable=False),
        sa.Column("to_kc", sa.String(length=128), nullable=False),
        sa.Column("type", sa.String(length=8), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.CheckConstraint("from_kc != to_kc", name="no_self_loop"),
        sa.ForeignKeyConstraint(
            ["graph_version_id"],
            ["graph_versions.id"],
            name="fk_kc_edges_graph_version",
        ),
        sa.UniqueConstraint(
            "graph_version_id",
            "from_kc",
            "to_kc",
            name="uq_kc_edges_graph_edge",
        ),
    )
    _create_or_adopt(
        "pedagogy_packs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("graph_version_id", sa.Integer(), nullable=False),
        sa.Column("kc_id", sa.String(length=128), nullable=False),
        sa.Column("content", _json_type(), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_version_id"],
            ["graph_versions.id"],
            name="fk_pedagogy_packs_graph_version",
        ),
        sa.UniqueConstraint(
            "graph_version_id",
            "kc_id",
            "version",
            name="uq_pedagogy_packs_graph_kc_version",
        ),
    )
    _create_or_adopt(
        "learners",
        sa.Column("learner_id", sa.String(length=36), primary_key=True),
        sa.Column("profile", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create_or_adopt(
        "resume_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("learner_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "revoked", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["learner_id"], ["learners.learner_id"], name="fk_resume_tokens_learner"
        ),
        sa.UniqueConstraint("token_hash", name="uq_resume_tokens_token_hash"),
    )
    _create_or_adopt(
        "evidence_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("learner_id", sa.String(length=36), nullable=False),
        sa.Column("t", sa.DateTime(timezone=True), nullable=False),
        sa.Column("item_id", sa.String(length=128), nullable=False),
        sa.Column("kc_ids", _json_type(), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("response_class", sa.String(length=32), nullable=False),
        sa.Column(
            "hints_used", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "assisted", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("misconception_id", sa.String(length=128), nullable=True),
        sa.Column("content_versions", _json_type(), nullable=False),
        sa.ForeignKeyConstraint(
            ["learner_id"], ["learners.learner_id"], name="fk_evidence_learner"
        ),
        sa.UniqueConstraint("event_id", name="uq_evidence_events_event_id"),
    )
    _create_or_adopt(
        "derived_mastery",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("learner_id", sa.String(length=36), nullable=False),
        sa.Column("kc_id", sa.String(length=128), nullable=False),
        sa.Column("direct", sa.Float(), nullable=False),
        sa.Column("inferred", sa.Float(), nullable=False),
        sa.Column(
            "observations", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("last_practiced", sa.DateTime(timezone=True), nullable=True),
        sa.Column("params_version", sa.Integer(), nullable=False),
        sa.Column("graph_version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["learner_id"], ["learners.learner_id"], name="fk_mastery_learner"
        ),
        sa.UniqueConstraint(
            "learner_id", "kc_id", name="uq_derived_mastery_learner_kc"
        ),
    )
    _create_or_adopt(
        "episodes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("learner_id", sa.String(length=36), nullable=False),
        sa.Column("target_kc", sa.String(length=128), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("envelope", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["learner_id"], ["learners.learner_id"], name="fk_episodes_learner"
        ),
    )
    _create_or_adopt(
        "generation_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("inputs", _json_type(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("result", _json_type(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", name="uq_generation_jobs_job_id"),
        sa.UniqueConstraint(
            "idempotency_key", name="uq_generation_jobs_idempotency_key"
        ),
    )
    _create_or_adopt(
        "mini_lessons",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kc_id", sa.String(length=128), nullable=False),
        sa.Column("applicability", _json_type(), nullable=False),
        sa.Column("versions", _json_type(), nullable=False),
        sa.Column("package", _json_type(), nullable=False),
        sa.Column("telemetry_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    raise RuntimeError(
        "the production schema is forward-only; restore a verified backup to downgrade"
    )
