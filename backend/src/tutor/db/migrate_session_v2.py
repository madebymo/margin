"""Deterministic additive migration for trustworthy-session v2.

The project does not currently carry Alembic.  This idempotent SQLAlchemy
migration upgrades an existing pilot database without relying on ``create_all``
to alter tables.  Run it before deploying API v2:

    python -m tutor.db.migrate_session_v2
"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.schema import CreateColumn

from tutor.db.base import Base
from tutor.db.session import get_engine

_MIGRATION_ID = "20260720_trustworthy_session_v2_1"
_CATALOG_MIGRATION_ID = "20260720_pedagogy_catalog_pin_v2_2"
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


def _catalog_evidence_columns() -> tuple[Column, ...]:
    return (
        Column(
            "pedagogy_catalog_version",
            String(128),
            nullable=False,
            server_default=text("'legacy'"),
        ),
    )


def _catalog_checkpoint_columns() -> tuple[Column, ...]:
    return (
        Column(
            "pedagogy_catalog_version",
            String(128),
            nullable=False,
            server_default=text("'legacy'"),
        ),
    )


def _migration_applied(connection: Connection, migration_id: str) -> bool:
    return (
        connection.exec_driver_sql(
            "SELECT migration_id FROM schema_migrations WHERE migration_id = :id",
            {"id": migration_id},
        ).first()
        is not None
    )


def _record_migration(connection: Connection, migration_id: str) -> None:
    connection.exec_driver_sql(
        "INSERT INTO schema_migrations (migration_id, applied_at) "
        "VALUES (:id, CURRENT_TIMESTAMP)",
        {"id": migration_id},
    )


def _add_missing_columns(
    connection: Connection,
    engine: Engine,
    table_name: str,
    columns: tuple[Column, ...],
) -> bool:
    inspector = inspect(connection)
    if table_name not in inspector.get_table_names():
        raise RuntimeError(f"base schema is missing {table_name}; initialize it first")
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    changed = False
    for column in columns:
        if column.name in existing:
            continue
        ddl = str(CreateColumn(column).compile(dialect=engine.dialect))
        connection.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")
        changed = True
    return changed


def _apply_session_v2_base(connection: Connection, engine: Engine) -> bool:
    changed = _add_missing_columns(
        connection, engine, "evidence_events", _evidence_columns()
    )
    existing_tables = set(inspect(connection).get_table_names())
    for table_name in _NEW_TABLES:
        Base.metadata.tables[table_name].create(bind=connection, checkfirst=True)
        changed = changed or table_name not in existing_tables

    changed = (
        _add_missing_columns(
            connection, engine, "resume_tokens", _resume_token_columns()
        )
        or changed
    )
    # Freeze the old "latest checkpoint for learner" resolution once at
    # migration time so already-issued v2 tokens resume one exact episode.
    connection.exec_driver_sql(
        "UPDATE resume_tokens SET session_id = ("
        "SELECT session_id FROM session_checkpoints "
        "WHERE session_checkpoints.learner_id = resume_tokens.learner_id "
        "ORDER BY session_checkpoints.updated_at DESC LIMIT 1"
        ") WHERE session_id IS NULL"
    )
    return (
        _add_missing_columns(
            connection, engine, "widget_attempts", _widget_attempt_columns()
        )
        or changed
    )


def _apply_catalog_pinning(connection: Connection, engine: Engine) -> bool:
    evidence_changed = _add_missing_columns(
        connection,
        engine,
        "evidence_events",
        _catalog_evidence_columns(),
    )
    checkpoint_changed = _add_missing_columns(
        connection,
        engine,
        "session_checkpoints",
        _catalog_checkpoint_columns(),
    )
    return evidence_changed or checkpoint_changed


def migrate(engine: Engine) -> bool:
    """Apply every unapplied additive v2 migration in order."""
    import tutor.db.models  # noqa: F401 — register all tables on Base.metadata

    changed = False
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(migration_id VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
        )
        if not _migration_applied(connection, _MIGRATION_ID):
            changed = _apply_session_v2_base(connection, engine) or changed
            _record_migration(connection, _MIGRATION_ID)
        if not _migration_applied(connection, _CATALOG_MIGRATION_ID):
            changed = _apply_catalog_pinning(connection, engine) or changed
            _record_migration(connection, _CATALOG_MIGRATION_ID)
    return changed


def main() -> None:
    migrate(get_engine())


if __name__ == "__main__":
    main()
