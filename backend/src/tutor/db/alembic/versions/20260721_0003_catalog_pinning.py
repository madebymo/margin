"""Pin pedagogy catalog provenance while preserving legacy records.

Revision ID: 20260721_0003
Revises: 20260721_0002
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20260721_0003"
down_revision: str | None = "20260721_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _add_catalog_column(table_name: str) -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }
    if "pedagogy_catalog_version" not in columns:
        op.add_column(
            table_name,
            sa.Column(
                "pedagogy_catalog_version",
                sa.String(length=128),
                nullable=False,
                server_default=sa.text("'legacy'"),
            ),
        )


def upgrade() -> None:
    _add_catalog_column("evidence_events")
    _add_catalog_column("session_checkpoints")


def downgrade() -> None:
    raise RuntimeError(
        "the production schema is forward-only; restore a verified backup to downgrade"
    )
