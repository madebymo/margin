"""Alembic converges fresh and legacy databases on one production head."""

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
import pytest
from sqlalchemy import inspect

from tutor.api.v2_persistence import V2PersistenceService
from tutor.db.base import Base
from tutor.db.migrate_session_v2 import (
    LEGACY_CUSTOM_MIGRATIONS,
    REQUIRED_SCHEMA_HEAD,
    _alembic_config,
    migrate,
    schema_migration_status,
)
from tutor.db.session import create_all, get_engine


def test_revision_chain_has_one_explicit_production_head():
    scripts = ScriptDirectory.from_config(_alembic_config())

    assert scripts.get_heads() == [REQUIRED_SCHEMA_HEAD]


def test_migration_builds_a_fresh_database_without_create_all():
    engine = get_engine("sqlite+pysqlite:///:memory:")

    assert migrate(engine) is True
    assert schema_migration_status(engine) == {
        "reachable": True,
        "current": True,
        "head": REQUIRED_SCHEMA_HEAD,
    }
    assert {
        "alembic_version",
        "learners",
        "evidence_events",
        "session_checkpoints",
        "session_mutation_receipts",
        "transcript_entries",
        "item_exposures",
        "widget_attempts",
    } <= set(inspect(engine).get_table_names())
    V2PersistenceService(engine)

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"compare_type": True},
        )
        assert compare_metadata(context, Base.metadata) == []


def test_current_unversioned_create_all_schema_is_adopted_in_place():
    engine = get_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)

    assert schema_migration_status(engine)["current"] is False
    assert migrate(engine) is True
    assert schema_migration_status(engine)["current"] is True
    assert migrate(engine) is False


def test_migration_adds_v2_columns_and_tables_to_legacy_schema():
    engine = get_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE learners ("
            "learner_id VARCHAR(36) PRIMARY KEY, profile JSON NOT NULL, "
            "created_at TIMESTAMP NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE resume_tokens ("
            "id INTEGER PRIMARY KEY, learner_id VARCHAR(36) NOT NULL, "
            "token_hash VARCHAR(128) NOT NULL UNIQUE, expires_at TIMESTAMP NOT NULL, "
            "revoked BOOLEAN NOT NULL DEFAULT FALSE, created_at TIMESTAMP NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE evidence_events ("
            "id INTEGER PRIMARY KEY, event_id VARCHAR(36) NOT NULL UNIQUE, "
            "learner_id VARCHAR(36) NOT NULL, t TIMESTAMP NOT NULL, "
            "item_id VARCHAR(128) NOT NULL, kc_ids JSON NOT NULL, "
            "correct BOOLEAN NOT NULL, response_class VARCHAR(32) NOT NULL, "
            "hints_used INTEGER NOT NULL DEFAULT 0, "
            "assisted BOOLEAN NOT NULL DEFAULT FALSE, "
            "misconception_id VARCHAR(128), content_versions JSON NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE schema_migrations ("
            "migration_id VARCHAR(128) PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
        )
        for migration_id in LEGACY_CUSTOM_MIGRATIONS:
            connection.exec_driver_sql(
                "INSERT INTO schema_migrations VALUES (:migration_id, CURRENT_TIMESTAMP)",
                {"migration_id": migration_id},
            )
        connection.exec_driver_sql(
            "INSERT INTO learners VALUES "
            "('learner-1', '{}', CURRENT_TIMESTAMP)"
        )
        connection.exec_driver_sql(
            "INSERT INTO evidence_events "
            "(id,event_id,learner_id,t,item_id,kc_ids,correct,response_class,"
            "hints_used,assisted,content_versions) VALUES "
            "(1,'event-1','learner-1',CURRENT_TIMESTAMP,'item','[]',TRUE,"
            "'symbolic_entry',0,FALSE,'{}')"
        )

    with pytest.raises(RuntimeError, match="migrate_session_v2"):
        V2PersistenceService(engine)
    assert migrate(engine) is True
    V2PersistenceService(engine)
    inspector = inspect(engine)
    assert {
        "session_checkpoints",
        "session_mutation_receipts",
        "transcript_entries",
        "item_exposures",
        "widget_attempts",
    } <= set(inspector.get_table_names())
    columns = {
        column["name"] for column in inspector.get_columns("evidence_events")
    }
    assert {
        "episode_id",
        "family_id",
        "surface",
        "item_revision",
        "attempt_number",
        "policy_version",
        "learner_params_version",
        "content_provenance",
        "learning_opportunity",
        "pedagogy_catalog_version",
    } <= columns
    checkpoint_columns = {
        column["name"]
        for column in inspector.get_columns("session_checkpoints")
    }
    assert "pedagogy_catalog_version" in checkpoint_columns
    resume_columns = {
        column["name"] for column in inspector.get_columns("resume_tokens")
    }
    assert "session_id" in resume_columns
    widget_columns = {
        column["name"] for column in inspector.get_columns("widget_attempts")
    }
    assert {"verification_status", "counted"} <= widget_columns
    with engine.connect() as connection:
        legacy = connection.exec_driver_sql(
            "SELECT surface, item_revision, policy_version, "
            "pedagogy_catalog_version "
            "FROM evidence_events WHERE id = 1"
        ).one()
        historical_migrations = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT migration_id FROM schema_migrations"
            )
        }
        alembic_revision = connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
    assert tuple(legacy) == ("legacy", 1, "legacy", "legacy")
    assert historical_migrations == set(LEGACY_CUSTOM_MIGRATIONS)
    assert alembic_revision == REQUIRED_SCHEMA_HEAD
    assert migrate(engine) is False
