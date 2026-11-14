"""The layering the design fixes (doc/conversation-memory-design.md §1): ``conversation → core +
contracts`` and nothing else; and no engine package knows conversations exist. Checked on the source's
import statements, absolute and relative alike, so a stray import fails here rather than quietly
coupling the layers."""

import ast
from pathlib import Path

import tarnrag

ROOT = Path(tarnrag.__file__).resolve().parent

# Every package that must not know conversations exist: the engines, the store, the shared kernel, and
# the eval harnesses built on the engines.
NOT_CONVERSATION_AWARE = ("ingestion", "retrieval", "generation", "storage", "contracts", "core", "eval")


def _imported_modules(path: Path, root: Path) -> list[str]:
    """The absolute dotted name of every module ``path`` imports — a relative import (``from ..x import
    y``) is resolved against the importing module's own package."""
    # The containing package: drop the module name (for an __init__.py, that leaves the package itself).
    package = list(path.relative_to(root.parent).with_suffix("").parts)[:-1]
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: climb level-1 packages up from this module's package
                base = package[: len(package) - (node.level - 1)]
                if node.module:
                    found.append(".".join([*base, node.module]))
                else:  # `from .. import retrieval` imports each name as a module of base
                    found += [".".join([*base, alias.name]) for alias in node.names]
            elif node.module:
                found.append(node.module)
    return found


def _imports(package: str, root: Path = ROOT) -> set[str]:
    """Every ``tarnrag.<x>`` sub-package imported anywhere under ``<root>/<package>/``."""
    found: set[str] = set()
    for path in (root / package).rglob("*.py"):
        for name in _imported_modules(path, root):
            parts = name.split(".")
            if parts[0] == root.name and len(parts) > 1:
                found.add(parts[1])
    return found


def test_conversation_depends_only_on_core_and_contracts():
    assert _imports("conversation") <= {"conversation", "core", "contracts"}


def test_no_engine_package_imports_conversation():
    for package in NOT_CONVERSATION_AWARE:
        assert "conversation" not in _imports(package), package


def test_relative_imports_are_caught_too(tmp_path):
    """The guard itself: a relative import of a forbidden package counts, from a module or a sub-package."""
    root = tmp_path / "tarnrag"
    (root / "conversation" / "components").mkdir(parents=True)
    (root / "conversation" / "memory.py").write_text("from ..retrieval.types import Query\n")
    (root / "conversation" / "components" / "x.py").write_text("from ... import generation\nfrom . import y\n")
    (root / "conversation" / "__init__.py").write_text("from .memory import Query\n")
    assert _imports("conversation", root) == {"retrieval", "generation", "conversation"}
