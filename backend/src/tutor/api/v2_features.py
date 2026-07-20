"""Independent rollout switches for the trustworthy-session stack."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

_ALLOWED_STUDENT_ROLLOUT_PERCENTAGES = frozenset({0, 5, 25, 100})
_FLAG_ENV_NAMES = {
    "api_session_v2": "TUTOR_ENABLE_API_SESSION_V2",
    "content_allocation_v2": "TUTOR_ENABLE_CONTENT_ALLOCATION_V2",
    "diagnosis_v2": "TUTOR_ENABLE_DIAGNOSIS_V2",
    "lesson_flow_v2": "TUTOR_ENABLE_LESSON_FLOW_V2",
    "rich_widgets": "TUTOR_ENABLE_RICH_WIDGETS_V2",
}
_ROLLOUT_ENV_NAME = "TUTOR_V2_STUDENT_ROLLOUT_PERCENT"


def _flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _student_rollout_percentage() -> int:
    raw = os.environ.get(_ROLLOUT_ENV_NAME, "100")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "TUTOR_V2_STUDENT_ROLLOUT_PERCENT must be one of 0, 5, 25, or 100"
        ) from exc
    if value not in _ALLOWED_STUDENT_ROLLOUT_PERCENTAGES:
        raise ValueError(
            "TUTOR_V2_STUDENT_ROLLOUT_PERCENT must be one of 0, 5, 25, or 100"
        )
    return value


@dataclass(frozen=True)
class V2FeatureFlags:
    api_session_v2: bool = True
    content_allocation_v2: bool = True
    diagnosis_v2: bool = True
    lesson_flow_v2: bool = True
    rich_widgets: bool = True
    student_rollout_percent: int = 100

    def __post_init__(self) -> None:
        if self.student_rollout_percent not in _ALLOWED_STUDENT_ROLLOUT_PERCENTAGES:
            raise ValueError("student_rollout_percent must be one of 0, 5, 25, or 100")

    @classmethod
    def from_environment(cls) -> "V2FeatureFlags":
        pilot_production = _flag("TUTOR_PILOT_PRODUCTION", default=False)
        if pilot_production:
            missing = [
                name
                for name in (*_FLAG_ENV_NAMES.values(), _ROLLOUT_ENV_NAME)
                if name not in os.environ
            ]
            if missing:
                raise RuntimeError(
                    "TUTOR_PILOT_PRODUCTION requires explicit v2 rollout "
                    f"configuration: {', '.join(missing)}"
                )
        return cls(
            api_session_v2=_flag(_FLAG_ENV_NAMES["api_session_v2"]),
            content_allocation_v2=_flag(
                _FLAG_ENV_NAMES["content_allocation_v2"]
            ),
            diagnosis_v2=_flag(_FLAG_ENV_NAMES["diagnosis_v2"]),
            lesson_flow_v2=_flag(_FLAG_ENV_NAMES["lesson_flow_v2"]),
            rich_widgets=_flag(_FLAG_ENV_NAMES["rich_widgets"]),
            student_rollout_percent=_student_rollout_percentage(),
        )

    @property
    def student_stack_enabled(self) -> bool:
        """Whether the core v2 flow is safe to admit, independent of rich UI."""
        return all(
            (
                self.api_session_v2,
                self.content_allocation_v2,
                self.diagnosis_v2,
                self.lesson_flow_v2,
            )
        )

    @property
    def student_flow_enabled(self) -> bool:
        return self.student_stack_enabled and self.student_rollout_percent > 0

    def as_dict(self) -> dict[str, bool | int]:
        return asdict(self)
