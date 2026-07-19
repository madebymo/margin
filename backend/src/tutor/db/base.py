"""Declarative base and the JSON column type used across all tables.

JSONVariant renders as JSONB on Postgres and plain JSON elsewhere (SQLite in
tests), so the same models work in both environments.
"""

from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase

JSONVariant = JSON().with_variant(postgresql.JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Base class for all ORM models."""
