"""Vendor-neutral runtime adapter factory loading."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Protocol, runtime_checkable

import pytest

from tutor.api.runtime_plugins import (
    RuntimePluginError,
    build_runtime_plugin,
    build_runtime_plugin_from_environment,
    load_factory,
)


@runtime_checkable
class SampleAdapter(Protocol):
    def emit(self, value: str) -> None: ...


class Adapter:
    def __init__(self) -> None:
        self.values: list[str] = []

    def emit(self, value: str) -> None:
        self.values.append(value)


def _module(monkeypatch, name: str = "tests.runtime_adapter") -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_load_factory_imports_exact_module_callable_and_trims_outer_padding(
    monkeypatch,
):
    module = _module(monkeypatch)
    adapter = Adapter()
    module.create_adapter = lambda: adapter

    factory = load_factory(
        "  tests.runtime_adapter:create_adapter  ",
        contract_name="SampleAdapter",
    )

    assert factory() is adapter


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "   ",
        "module_without_factory",
        ":factory",
        "module:",
        "module:factory:extra",
        ".relative:factory",
        "package..module:factory",
        "package.bad-name:factory",
        "package.module:bad-name",
        "package.module:nested.factory",
        "package module:factory",
        "package.module:factory name",
    ],
)
def test_load_factory_rejects_ambiguous_or_non_python_specs(spec):
    with pytest.raises(
        RuntimePluginError,
        match=r"must use package\.module:factory syntax",
    ):
        load_factory(spec, contract_name="SampleAdapter")


def test_load_factory_rejects_empty_contract_label_before_import(monkeypatch):
    imported = False

    def unexpected_import(name):
        nonlocal imported
        imported = True

    monkeypatch.setattr(
        "tutor.api.runtime_plugins.import_module",
        unexpected_import,
    )
    with pytest.raises(ValueError, match="contract_name must not be empty"):
        load_factory("anything:factory", contract_name="  ")
    assert imported is False


def test_import_failure_is_sanitized_and_does_not_chain_vendor_secret(monkeypatch):
    secret = "vendor_api_key=super-secret"

    def fail_import(name):
        raise RuntimeError(secret)

    monkeypatch.setattr("tutor.api.runtime_plugins.import_module", fail_import)
    with pytest.raises(RuntimePluginError) as caught:
        load_factory("vendor.metrics:create", contract_name="MetricsSink")

    message = str(caught.value)
    assert "vendor.metrics" in message
    assert "RuntimeError" in message
    assert secret not in message
    assert caught.value.__cause__ is None


def test_missing_factory_attribute_is_sanitized(monkeypatch):
    _module(monkeypatch)

    with pytest.raises(RuntimePluginError) as caught:
        load_factory(
            "tests.runtime_adapter:create_adapter",
            contract_name="SampleAdapter",
        )

    assert "could not resolve" in str(caught.value)
    assert "AttributeError" in str(caught.value)
    assert caught.value.__cause__ is None


def test_module_attribute_hook_failure_is_sanitized(monkeypatch):
    secret = "token=do-not-log"

    class DynamicModule:
        def __getattr__(self, name):
            raise RuntimeError(secret)

    monkeypatch.setattr(
        "tutor.api.runtime_plugins.import_module",
        lambda name: DynamicModule(),
    )
    with pytest.raises(RuntimePluginError) as caught:
        load_factory("vendor.dynamic:create", contract_name="SampleAdapter")

    assert "RuntimeError" in str(caught.value)
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None


def test_non_callable_factory_is_rejected(monkeypatch):
    module = _module(monkeypatch)
    module.create_adapter = Adapter()

    with pytest.raises(RuntimePluginError, match="is not callable"):
        load_factory(
            "tests.runtime_adapter:create_adapter",
            contract_name="SampleAdapter",
        )


def test_build_runtime_plugin_calls_factory_once_and_validates_protocol(monkeypatch):
    module = _module(monkeypatch)
    calls = 0

    def create_adapter():
        nonlocal calls
        calls += 1
        return Adapter()

    module.create_adapter = create_adapter
    plugin = build_runtime_plugin(
        "tests.runtime_adapter:create_adapter",
        contract=SampleAdapter,
        contract_name="SampleAdapter",
    )

    assert isinstance(plugin, Adapter)
    assert calls == 1


@pytest.mark.parametrize("spec", [None, "", "  "])
def test_optional_missing_plugin_returns_none_without_import(monkeypatch, spec):
    def unexpected_import(name):
        raise AssertionError("missing optional plugin must not import")

    monkeypatch.setattr(
        "tutor.api.runtime_plugins.import_module",
        unexpected_import,
    )
    assert (
        build_runtime_plugin(
            spec,
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )
        is None
    )


@pytest.mark.parametrize("spec", [None, "", "  "])
def test_required_missing_plugin_fails_without_import(monkeypatch, spec):
    def unexpected_import(name):
        raise AssertionError("missing required plugin must not import")

    monkeypatch.setattr(
        "tutor.api.runtime_plugins.import_module",
        unexpected_import,
    )
    with pytest.raises(RuntimePluginError, match="configuration is required"):
        build_runtime_plugin(
            spec,
            contract=SampleAdapter,
            contract_name="SampleAdapter",
            required=True,
        )


def test_factory_failure_is_sanitized_without_exception_chain(monkeypatch):
    module = _module(monkeypatch)
    secret = "redis_password=do-not-log"

    def create_adapter():
        raise ValueError(secret)

    module.create_adapter = create_adapter
    with pytest.raises(RuntimePluginError) as caught:
        build_runtime_plugin(
            "tests.runtime_adapter:create_adapter",
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )

    message = str(caught.value)
    assert "ValueError" in message
    assert secret not in message
    assert caught.value.__cause__ is None


def test_factory_requiring_arguments_fails_as_sanitized_type_error(monkeypatch):
    module = _module(monkeypatch)

    def create_adapter(required_argument):
        return Adapter()

    module.create_adapter = create_adapter
    with pytest.raises(RuntimePluginError) as caught:
        build_runtime_plugin(
            "tests.runtime_adapter:create_adapter",
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )
    assert "TypeError" in str(caught.value)
    assert "required_argument" not in str(caught.value)


def test_incompatible_factory_result_is_rejected_without_repr(monkeypatch):
    module = _module(monkeypatch)
    secret = "secret-in-repr"

    class Incompatible:
        def __repr__(self):
            return secret

    module.create_adapter = Incompatible
    with pytest.raises(RuntimePluginError) as caught:
        build_runtime_plugin(
            "tests.runtime_adapter:create_adapter",
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )
    assert "incompatible adapter" in str(caught.value)
    assert secret not in str(caught.value)


def test_contract_without_runtime_validation_is_normalized(monkeypatch):
    module = _module(monkeypatch)
    module.create_adapter = Adapter

    class NonRuntimeProtocol(Protocol):
        def emit(self, value: str) -> None: ...

    with pytest.raises(RuntimePluginError, match="runtime validation"):
        build_runtime_plugin(
            "tests.runtime_adapter:create_adapter",
            contract=NonRuntimeProtocol,
            contract_name="NonRuntimeProtocol",
        )


def test_environment_loader_returns_none_when_optional_variable_is_absent(monkeypatch):
    monkeypatch.delenv("TUTOR_TEST_PLUGIN_FACTORY", raising=False)
    assert (
        build_runtime_plugin_from_environment(
            "TUTOR_TEST_PLUGIN_FACTORY",
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )
        is None
    )


def test_environment_loader_requires_config_when_requested(monkeypatch):
    monkeypatch.delenv("TUTOR_TEST_PLUGIN_FACTORY", raising=False)
    with pytest.raises(RuntimePluginError, match="configuration is required"):
        build_runtime_plugin_from_environment(
            "TUTOR_TEST_PLUGIN_FACTORY",
            contract=SampleAdapter,
            contract_name="SampleAdapter",
            required=True,
        )


def test_environment_loader_builds_configured_adapter(monkeypatch):
    module = _module(monkeypatch)
    module.create_adapter = Adapter
    monkeypatch.setenv(
        "TUTOR_TEST_PLUGIN_FACTORY",
        "tests.runtime_adapter:create_adapter",
    )

    plugin = build_runtime_plugin_from_environment(
        "TUTOR_TEST_PLUGIN_FACTORY",
        contract=SampleAdapter,
        contract_name="SampleAdapter",
        required=True,
    )

    assert isinstance(plugin, Adapter)


@pytest.mark.parametrize("env_name", ["", " PADDED", "PADDED "])
def test_environment_loader_rejects_invalid_environment_name(env_name):
    with pytest.raises(ValueError, match="env_name must be"):
        build_runtime_plugin_from_environment(
            env_name,
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )


def test_system_exit_from_trusted_factory_is_not_swallowed(monkeypatch):
    module = _module(monkeypatch)

    def stop_process():
        raise SystemExit(7)

    module.create_adapter = stop_process
    with pytest.raises(SystemExit, match="7"):
        build_runtime_plugin(
            "tests.runtime_adapter:create_adapter",
            contract=SampleAdapter,
            contract_name="SampleAdapter",
        )
