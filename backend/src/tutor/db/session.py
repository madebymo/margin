"""Engine construction and schema creation helpers.

Production PostgreSQL connections use deliberately bounded defaults.  Every
setting can be overridden through a documented environment variable, but bad
values fail startup rather than silently selecting an unbounded driver
default.  SQLite keeps its small test/development configuration.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from tutor.db.base import Base

DEFAULT_URL = "sqlite+pysqlite:///:memory:"

_APPLICATION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,63}$")


def _bounded_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class PostgresEngineSettings:
    """Bounded PostgreSQL pool and server-side timeout configuration."""

    connect_timeout_seconds: int = 5
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout_seconds: int = 5
    pool_recycle_seconds: int = 1800
    statement_timeout_ms: int = 3000
    lock_timeout_ms: int = 1000
    idle_transaction_timeout_ms: int = 5000
    application_name: str = "adaptive-math-tutor"

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "PostgresEngineSettings":
        source = os.environ if environ is None else environ
        application_name = source.get(
            "TUTOR_DB_APPLICATION_NAME", "adaptive-math-tutor"
        )
        if _APPLICATION_NAME_PATTERN.fullmatch(application_name) is None:
            raise ValueError(
                "TUTOR_DB_APPLICATION_NAME must contain 1-63 letters, digits, dots, "
                "hyphens, or underscores"
            )
        return cls(
            connect_timeout_seconds=_bounded_int(
                source,
                "TUTOR_DB_CONNECT_TIMEOUT_SECONDS",
                5,
                minimum=1,
                maximum=30,
            ),
            pool_size=_bounded_int(
                source, "TUTOR_DB_POOL_SIZE", 10, minimum=1, maximum=100
            ),
            max_overflow=_bounded_int(
                source, "TUTOR_DB_MAX_OVERFLOW", 5, minimum=0, maximum=100
            ),
            pool_timeout_seconds=_bounded_int(
                source,
                "TUTOR_DB_POOL_TIMEOUT_SECONDS",
                5,
                minimum=1,
                maximum=60,
            ),
            pool_recycle_seconds=_bounded_int(
                source,
                "TUTOR_DB_POOL_RECYCLE_SECONDS",
                1800,
                minimum=60,
                maximum=86400,
            ),
            statement_timeout_ms=_bounded_int(
                source,
                "TUTOR_DB_STATEMENT_TIMEOUT_MS",
                3000,
                minimum=250,
                maximum=60000,
            ),
            lock_timeout_ms=_bounded_int(
                source,
                "TUTOR_DB_LOCK_TIMEOUT_MS",
                1000,
                minimum=100,
                maximum=30000,
            ),
            idle_transaction_timeout_ms=_bounded_int(
                source,
                "TUTOR_DB_IDLE_TRANSACTION_TIMEOUT_MS",
                5000,
                minimum=500,
                maximum=60000,
            ),
            application_name=application_name,
        )

    @property
    def server_options(self) -> str:
        """libpq options applied to every connection before its first query."""

        return " ".join(
            (
                f"-c statement_timeout={self.statement_timeout_ms}",
                f"-c lock_timeout={self.lock_timeout_ms}",
                "-c idle_in_transaction_session_timeout="
                f"{self.idle_transaction_timeout_ms}",
            )
        )


def get_engine(
    url: str | None = None,
    *,
    postgres_settings: PostgresEngineSettings | None = None,
) -> Engine:
    """Create an engine from an explicit URL, ``DATABASE_URL``, or in-memory SQLite.

    In-memory SQLite uses a StaticPool so every connection shares the same
    database — otherwise each new connection would see a fresh, empty schema.
    """
    resolved = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    parsed = make_url(resolved)
    if parsed.get_backend_name() == "sqlite" and parsed.database in {None, "", ":memory:"}:
        return create_engine(
            resolved,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    if parsed.get_backend_name() == "postgresql":
        settings = postgres_settings or PostgresEngineSettings.from_environment()
        return create_engine(
            resolved,
            pool_pre_ping=True,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_recycle=settings.pool_recycle_seconds,
            connect_args={
                "connect_timeout": settings.connect_timeout_seconds,
                "application_name": settings.application_name,
                "options": settings.server_options,
            },
        )
    return create_engine(resolved)


def create_all(engine: Engine) -> None:
    """Create all tables. Dev/test convenience; production uses migrations."""
    import tutor.db.models  # noqa: F401  # ensure mappings are registered

    Base.metadata.create_all(engine)
