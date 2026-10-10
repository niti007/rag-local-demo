"""Guards for the chat UI theme toggle (GP-2): relies on Chainlit 2.x's built-in
Light/Dark/System toggle, and the UI module must stay import-light."""

from __future__ import annotations

import ast
import glob
import os
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import chainlit

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_chainlit_major_version_at_least_2():
    assert int(version("chainlit").split(".")[0]) >= 2


def test_bundled_frontend_has_theme_support():
    pattern = os.path.join(
        os.path.dirname(chainlit.__file__), "frontend", "dist", "assets", "index-*.js"
    )
    files = glob.glob(pattern)
    assert files, f"no bundled frontend found at {pattern}"
    js = "".join(Path(f).read_text(encoding="utf-8", errors="ignore") for f in files)
    for token in ("theme-toggle", "vite-ui-theme", "prefers-color-scheme"):
        assert token in js, f"{token!r} missing from bundled Chainlit frontend"


def test_app_module_level_imports_are_light():
    tree = ast.parse((PROJECT_ROOT / "src" / "ui" / "app.py").read_text(encoding="utf-8"))
    roots = []
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Import):
            roots += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.append(node.module)
    bad = []
    for name in roots:
        top = name.split(".")[0]
        if top in sys.stdlib_module_names or top == "chainlit" or name == "src.config":
            continue
        bad.append(name)
    assert not bad, f"unexpected module-level imports in src/ui/app.py: {bad}"


def test_no_default_theme_override():
    cfg = PROJECT_ROOT / ".chainlit" / "config.toml"
    if not cfg.exists():
        pytest.skip("no local .chainlit/config.toml")
    for line in cfg.read_text(encoding="utf-8").splitlines():
        assert not line.strip().startswith("default_theme"), line
