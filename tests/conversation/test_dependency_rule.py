"""The layering the design fixes (doc/conversation-memory-design.md §1): ``conversation → core +
contracts`` and nothing else; and no engine package knows conversations exist. Checked on the source's
import statements, so a stray import fails here rather than quietly coupling the layers."""

import ast
from pathlib import Path

import tarnrag

ROOT = Path(tarnrag.__file__).resolve().parent


def _imports(package: str) -> set[str]:
    """Every ``tarnrag.<x>`` sub-package imported anywhere under ``tarnrag/<package>/``."""
    found: set[str] = set()
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                parts = name.split(".")
                if parts[0] == "tarnrag" and len(parts) > 1:
                    found.add(parts[1])
    return found


def test_conversation_depends_only_on_core_and_contracts():
    assert _imports("conversation") <= {"conversation", "core", "contracts"}


def test_no_engine_package_imports_conversation():
    for package in ("ingestion", "retrieval", "generation", "storage", "contracts", "core"):
        assert "conversation" not in _imports(package), package
