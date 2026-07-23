"""Error and performance monitoring configuration for the API process."""

import os
from collections.abc import Mapping

import sentry_sdk

_SENTRY_DSN_ENV = "SENTRY_DSN"
_SENTRY_TRACES_SAMPLE_RATE_ENV = "SENTRY_TRACES_SAMPLE_RATE"


def configure_sentry(environ: Mapping[str, str] | None = None) -> bool:
    """Initialize Sentry when a DSN is configured.

    Returns whether monitoring was enabled. Keeping an absent DSN as a no-op
    lets local development and tests run without sending telemetry.
    """
    values = os.environ if environ is None else environ
    dsn = values.get(_SENTRY_DSN_ENV, "").strip()
    if not dsn:
        return False

    raw_sample_rate = values.get(_SENTRY_TRACES_SAMPLE_RATE_ENV, "1.0")
    try:
        traces_sample_rate = float(raw_sample_rate)
    except ValueError as exc:
        raise RuntimeError(
            f"{_SENTRY_TRACES_SAMPLE_RATE_ENV} must be a number between 0 and 1"
        ) from exc
    if not 0.0 <= traces_sample_rate <= 1.0:
        raise RuntimeError(
            f"{_SENTRY_TRACES_SAMPLE_RATE_ENV} must be between 0 and 1"
        )

    sentry_sdk.init(
        dsn=dsn,
        send_default_pii=True,
        traces_sample_rate=traces_sample_rate,
    )
    return True


__all__ = ["configure_sentry"]
