"""Alembic entry point and production schema-head readiness checks.

The module name is retained so existing deployment commands keep working, but
schema ownership now belongs to the Alembic revision chain in
``tutor.db.alembic``.  The revisions are deliberately additive: an existing
unversioned pilot database is adopted in place and legacy rows are preserved.

Run before starting a production worker::

    python -m tutor.db.migrate_session_v2

The standard Alembic CLI is equivalent::

    alembic -c backend/alembic.ini upgrade head
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, inspect

from tutor.db.session import get_engine

# Keep this a literal, reviewable deployment contract.  Readiness compares the
# database's Alembic revision to this exact value; it does not infer currency
# from whichever revision files happen to be present at runtime.
REQUIRED_SCHEMA_HEAD = "20260721_0004"

# Historical identifiers are retained only as documentation for databases
# upgraded by the pre-Alembic command.  Alembic leaves ``schema_migrations``
# and its rows untouched while adopting those databases at the real head.
LEGACY_CUSTOM_MIGRATIONS = (
    "20260720_trustworthy_session_v2_1",
    "20260720_pedagogy_catalog_pin_v2_2",
    "20260720_operational_indexes_v2_3",
)


def _script_location() -> Path:
    return Path(__file__).resolve().parent / "alembic"


def _database_revisions(engine: Engine) -> set[str]:
    """Return current Alembic revisions without importing Alembic at startup."""

    with engine.connect() as connection:
        if "alembic_version" not in inspect(connection).get_table_names():
            return set()
        return {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            )
        }


def schema_migration_status(engine: Engine) -> dict[str, object]:
    """Return a sanitized readiness snapshot for the explicit production head."""

    try:
        revisions = _database_revisions(engine)
    except Exception:
        return {
            "reachable": False,
            "current": False,
            "head": REQUIRED_SCHEMA_HEAD,
        }
    return {
        "reachable": True,
        "current": revisions == {REQUIRED_SCHEMA_HEAD},
        "head": REQUIRED_SCHEMA_HEAD,
    }


def _alembic_config(*, connection=None):
    """Build a location-stable Alembic config for CLI-compatible upgrades."""

    try:
        from alembic.config import Config
    except ImportError as exc:  # pragma: no cover - packaging/configuration guard
        raise RuntimeError(
            "Alembic is required for database migrations; install backend[pilot]"
        ) from exc

    config = Config()
    config.set_main_option("script_location", str(_script_location()))
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def migrate(engine: Engine) -> bool:
    """Upgrade ``engine`` to the pinned Alembic head.

    Returns ``True`` when the database revision changed and ``False`` for an
    already-current database.  Migration scripts perform their own schema
    introspection so empty, legacy-unversioned, pre-Alembic-v2, and current
    databases all converge without deleting or rewriting legacy rows.
    """

    from alembic import command

    before = _database_revisions(engine)
    if before == {REQUIRED_SCHEMA_HEAD}:
        return False
    with engine.begin() as connection:
        command.upgrade(_alembic_config(connection=connection), "head")
    after = _database_revisions(engine)
    if after != {REQUIRED_SCHEMA_HEAD}:
        raise RuntimeError(
            "database migration completed without reaching the required schema head"
        )
    return before != after


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL is required; refusing to migrate an ephemeral database"
        )
    migrate(get_engine(database_url))


if __name__ == "__main__":
    main()
