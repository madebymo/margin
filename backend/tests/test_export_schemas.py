"""JSON Schema export: files are written and structurally valid."""

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_json_schemas.py"
_spec = importlib.util.spec_from_file_location("export_json_schemas", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_export_writes_valid_schema_files(tmp_path):
    written = _module.export_schemas(tmp_path)
    assert len(written) == 10
    assert {path.name for path in written} >= {
        "item_blueprint_document.json",
        "content_review_manifest.json",
    }
    for path in written:
        data = json.loads(path.read_text())
        assert any(key in data for key in ("properties", "$defs", "oneOf")), path.name
