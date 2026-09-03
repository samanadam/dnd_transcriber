"""Every `config.<name>` in the package must actually exist on Config.

The recorder half shipped a reference to a setting a refactor had removed, and
no test noticed because none of them drove that code path - the bot connected to
voice and then raised. This package is clean today; the check is here so it
stays that way as settings come and go.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from dnd_transcriber.config import Config

PACKAGE = Path(__file__).resolve().parent.parent / "dnd_transcriber"


def config_attribute_names() -> set[str]:
    """Anything reachable on a Config: fields, properties and methods."""
    fields = {f.name for f in dataclasses.fields(Config)}
    return fields | {name for name in dir(Config) if not name.startswith("__")}


def referenced_attributes(path: Path) -> set[str]:
    """Every `X.config.<attr>` and `config.<attr>` read in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr == "config":
            found.add(node.attr)
        elif isinstance(base, ast.Name) and base.id == "config":
            found.add(node.attr)
    return found


def python_modules() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if p.name != "__init__.py")


def test_the_scanner_finds_something_at_all():
    """Guard the guard: a silently empty scan would pass every check below."""
    seen: set[str] = set()
    for module in python_modules():
        seen |= referenced_attributes(module)
    assert "incoming_dir" in seen or "workspace" in seen
    assert len(seen) > 5


@pytest.mark.parametrize("module", python_modules(), ids=lambda p: p.name)
def test_every_config_attribute_referenced_exists(module: Path):
    unknown = sorted(referenced_attributes(module) - config_attribute_names())
    assert not unknown, (
        f"{module.name} reads config attributes that Config does not define: "
        f"{', '.join(unknown)}"
    )
