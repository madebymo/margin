"""Strict, vendor-neutral loading for operator-configured runtime adapters.

Plugin specifications use the explicit ``package.module:factory`` form. The
configured module is trusted deployment code and is imported only when this
loader is called. Factories receive no arguments; vendor-specific settings and
secrets remain owned by that external package rather than passing through the
tutor's configuration or logs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from importlib import import_module
from typing import TypeVar

T = TypeVar("T")

_MODULE_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_FACTORY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RuntimePluginError(RuntimeError):
    """A configured runtime adapter could not be safely constructed."""


def _contract_label(contract_name: str) -> str:
    label = contract_name.strip()
    if not label:
        raise ValueError("contract_name must not be empty")
    return label


def load_factory(
    spec: str,
    *,
    contract_name: str,
) -> Callable[[], object]:
    """Import one zero-argument factory from ``package.module:factory``.

    Import and attribute errors are normalized without including the original
    exception message, which may contain vendor configuration or secrets.
    Factory invocation and returned-contract validation are performed by
    :func:`build_runtime_plugin`.
    """
    label = _contract_label(contract_name)
    normalized = spec.strip()
    if normalized.count(":") != 1:
        raise RuntimePluginError(
            f"{label} factory must use package.module:factory syntax"
        )
    module_name, factory_name = normalized.split(":", 1)
    if (
        _MODULE_PATTERN.fullmatch(module_name) is None
        or _FACTORY_PATTERN.fullmatch(factory_name) is None
    ):
        raise RuntimePluginError(
            f"{label} factory must use package.module:factory syntax"
        )
    try:
        module = import_module(module_name)
    except Exception as exc:
        raise RuntimePluginError(
            f"could not import {label} factory module {module_name!r} "
            f"({type(exc).__name__})"
        ) from None
    try:
        factory = getattr(module, factory_name)
    except Exception as exc:
        raise RuntimePluginError(
            f"could not resolve {label} factory {normalized!r} "
            f"({type(exc).__name__})"
        ) from None
    if not callable(factory):
        raise RuntimePluginError(
            f"configured {label} factory {normalized!r} is not callable"
        )
    return factory


def build_runtime_plugin(
    spec: str | None,
    *,
    contract: type[T],
    contract_name: str,
    required: bool = False,
) -> T | None:
    """Construct and runtime-check one configured adapter.

    A missing or blank specification returns ``None`` unless ``required`` is
    true. The supplied contract must support ``isinstance`` checks, such as a
    normal class/ABC or a Protocol decorated with ``@runtime_checkable``.
    """
    label = _contract_label(contract_name)
    if spec is None or not spec.strip():
        if required:
            raise RuntimePluginError(f"{label} factory configuration is required")
        return None

    factory = load_factory(spec, contract_name=label)
    try:
        plugin = factory()
    except Exception as exc:
        normalized = spec.strip()
        raise RuntimePluginError(
            f"configured {label} factory {normalized!r} failed "
            f"({type(exc).__name__})"
        ) from None
    try:
        satisfies_contract = isinstance(plugin, contract)
    except TypeError:
        raise RuntimePluginError(
            f"{label} contract does not support runtime validation"
        ) from None
    if not satisfies_contract:
        raise RuntimePluginError(
            f"configured {label} factory returned an incompatible adapter"
        )
    return plugin


def build_runtime_plugin_from_environment(
    env_name: str,
    *,
    contract: type[T],
    contract_name: str,
    required: bool = False,
) -> T | None:
    """Construct an adapter from one named environment variable."""
    if not env_name or env_name.strip() != env_name:
        raise ValueError("env_name must be a non-empty, unpadded string")
    return build_runtime_plugin(
        os.environ.get(env_name),
        contract=contract,
        contract_name=contract_name,
        required=required,
    )
