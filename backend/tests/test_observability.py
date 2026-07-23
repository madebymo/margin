from unittest.mock import patch

import pytest

from tutor.api.observability import configure_sentry


def test_sentry_is_disabled_without_a_dsn():
    with patch("tutor.api.observability.sentry_sdk.init") as init:
        assert configure_sentry({}) is False

    init.assert_not_called()


def test_sentry_uses_configured_dsn_and_sample_rate():
    with patch("tutor.api.observability.sentry_sdk.init") as init:
        assert configure_sentry(
            {
                "SENTRY_DSN": "https://public@example.ingest.sentry.io/123",
                "SENTRY_TRACES_SAMPLE_RATE": "0.25",
            }
        ) is True

    init.assert_called_once_with(
        dsn="https://public@example.ingest.sentry.io/123",
        send_default_pii=True,
        traces_sample_rate=0.25,
    )


@pytest.mark.parametrize("sample_rate", ["invalid", "-0.1", "1.1"])
def test_sentry_rejects_invalid_sample_rates(sample_rate):
    with (
        patch("tutor.api.observability.sentry_sdk.init") as init,
        pytest.raises(RuntimeError, match="SENTRY_TRACES_SAMPLE_RATE"),
    ):
        configure_sentry(
            {
                "SENTRY_DSN": "https://public@example.ingest.sentry.io/123",
                "SENTRY_TRACES_SAMPLE_RATE": sample_rate,
            }
        )

    init.assert_not_called()
