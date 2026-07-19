"""Engine construction and schema creation helpers."""

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from tutor.db.base import Base

DEFAULT_URL = "sqlite+pysqlite:///:memory:"


def get_engine(url: str | None = None) -> Engine:
    """Create an engine from an explicit URL, ``DATABASE_URL``, or in-memory SQLite.

    In-memory SQLite uses a StaticPool so every connection shares the same
    database — otherwise each new connection would see a fresh, empty schema.
    """
    resolved = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    if resolved.startswith("sqlite") and ":memory:" in resolved:
        return create_engine(
            resolved,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    return create_engine(resolved)


def create_all(engine: Engine) -> None:
    """Create all tables. Dev/test convenience; production uses migrations."""
    import tutor.db.models  # noqa: F401  # ensure mappings are registered

    Base.metadata.create_all(engine)
