"""Generation jobs: the queue-shaped seam for content generation.

v1 runs a same-process worker; a real broker can replace it later without
changing this contract. Idempotency keys prevent duplicate generation.
"""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from tutor.schemas.common import JobStatus


def _utcnow() -> datetime:
    """Timezone-aware now(), used for defaults."""
    return datetime.now(timezone.utc)


class GenerationJob(BaseModel):
    """A single content-generation request with version-pinned inputs."""

    job_id: UUID
    idempotency_key: str = Field(min_length=1)
    kind: Literal["lesson", "probe", "pack"]
    inputs: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
