"""Deterministic additive migration for trustworthy-session v2.

The project does not currently carry Alembic.  This idempotent SQLAlchemy
migration upgrades an existing pilot database without relying on ``create_all``
to alter tables.  Run it before deploying API v2:

    python -m tutor.db.migrate_session_v2
"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateColumn

from tutor.db.base import Base
from tutor.db.session import get_engine

_MIGRATION_ID = "20260720_trustworthy_session_v2_1"
_NEW_TABLES = (
    "session_checkpoints",
    "session_mutation_receipts",
    "transcript_entries",
    "item_exposures",
    "widget_attempts",
)


def _evidence_columns() -> tuple[Column, ...]:
    return (
        Column("episode_id", String(36), nullable=True),
        Column("family_id", String(128), nullable=True),
        Column(
            "surface",
            String(32),
            nullable=False,
            server_default=text("'legacy'"),
        ),
        Column(
            "item_revision", Integer, nullable=False, server_default=text("1")
        ),
        Column(
            "attempt_number", Integer, nullable=False, server_default=text("1")
        ),
        Column(
            "policy_version",
            String(64),
            nullable=False,
            server_default=text("'legacy'"),
        ),
        Column(
            "learner_params_version",
            String(64),
            nullable=False,
            server_default=text("'v1'"),
        ),
        Column(
            "content_provenance",
            String(128),
            nullable=False,
            server_default=text("'legacy'"),
        ),
        Column(
            "learning_opportunity",
            Boolean,
            nullable=False,
            server_default=text("FALSE"),
        ),
    )


def _resume_token_columns() -> tuple[Column, ...]:
    return (
        Column(
            "session_id",
            String(36),
            ForeignKey("session_checkpoints.session_id"),
            nullable=True,
        ),
    )


def _widget_attempt_columns() -> tuple[Column, ...]:
    return (
        Column(
            "verification_status",
            String(32),
            nullable=False,
            server_default=text("'incorrect'"),
        ),
        Column(
            "counted",
            Boolean,
            nullable=False,
            server_default=text("TRUE"),
        ),
    )


def migrate(engine: Engine) -> bool:
    """Apply the migration once; return whether any schema work was performed."""
    import tutor.db.models  # noqa: F401 — register all tables on Base.metadata

    changed = False
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(migration_id VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
        )
        already_applied = connection.exec_driver_sql(
            "SELECT migration_id FROM schema_migrations WHERE migration_id = :id",
            {"id": _MIGRATION_ID},
        ).first()
        if already_applied is not None:
            return False

        inspector = inspect(connection)
        if "evidence_events" not in inspector.get_table_names():
            raise RuntimeError("base schema is missing evidence_events; initialize it first")
        existing_columns = {
            column["name"] for column in inspector.get_columns("evidence_events")
        }
        for column in _evidence_columns():
            if column.name in existing_columns:
                continue
            ddl = str(CreateColumn(column).compile(dialect=engine.dialect))
            connection.exec_driver_sql(
                f"ALTER TABLE evidence_events ADD COLUMN {ddl}"
            )
            changed = True

        for table_name in _NEW_TABLES:
            table = Base.metadata.tables[table_name]
            table.create(bind=connection, checkfirst=True)
            if table_name not in inspector.get_table_names():
                changed = True

        inspector = inspect(connection)
        if "resume_tokens" not in inspector.get_table_names():
            raise RuntimeError("base schema is missing resume_tokens; initialize it first")
        resume_columns = {
            column["name"] for column in inspector.get_columns("resume_tokens")
        }
        for column in _resume_token_columns():
            if column.name in resume_columns:
                continue
            ddl = str(CreateColumn(column).compile(dialect=engine.dialect))
            connection.exec_driver_sql(
                f"ALTER TABLE resume_tokens ADD COLUMN {ddl}"
            )
            changed = True

        # Freeze the old "latest checkpoint for learner" resolution once at
        # migration time so already-issued v2 tokens resume one exact episode.
        connection.exec_driver_sql(
            "UPDATE resume_tokens SET session_id = ("
            "SELECT session_id FROM session_checkpoints "
            "WHERE session_checkpoints.learner_id = resume_tokens.learner_id "
            "ORDER BY session_checkpoints.updated_at DESC LIMIT 1"
            ") WHERE session_id IS NULL"
        )

        inspector = inspect(connection)
        widget_columns = {
            column["name"] for column in inspector.get_columns("widget_attempts")
        }
        for column in _widget_attempt_columns():
            if column.name in widget_columns:
                continue
            ddl = str(CreateColumn(column).compile(dialect=engine.dialect))
            connection.exec_driver_sql(
                f"ALTER TABLE widget_attempts ADD COLUMN {ddl}"
            )
            changed = True

        connection.exec_driver_sql(
            "INSERT INTO schema_migrations (migration_id, applied_at) "
            "VALUES (:id, CURRENT_TIMESTAMP)",
            {"id": _MIGRATION_ID},
        )
    return changed


def main() -> None:
    migrate(get_engine())


if __name__ == "__main__":
    main()
