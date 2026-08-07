from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "runtime" / "experiments" / "cognitive_gain"
MODULES = (
    "runtime.experiments",
    "runtime.experiments.cognitive_gain",
    "runtime.experiments.cognitive_gain.contracts",
    "runtime.experiments.cognitive_gain.ledger",
)
FORBIDDEN = (
    "runtime.world",
    "runtime.organism.experience",
    "runtime.neural.runtime",
)


def test_passive_imports_have_no_connection_ddl_env_or_filesystem_side_effects(tmp_path):
    script = r'''
import json, os, pathlib, sys
sys.path.insert(0, %r)
before_env = dict(os.environ)
before_files = sorted(str(p.relative_to(pathlib.Path.cwd())) for p in pathlib.Path.cwd().rglob("*"))
for name in %r:
    __import__(name)
after_files = sorted(str(p.relative_to(pathlib.Path.cwd())) for p in pathlib.Path.cwd().rglob("*"))
print(json.dumps({
    "env_equal": before_env == dict(os.environ),
    "files_equal": before_files == after_files,
    "forbidden": [name for name in sys.modules if name.startswith(%r)],
    "postgres_imported": "runtime.experiments.cognitive_gain.postgres_store" in sys.modules,
}))
''' % (str(ROOT), MODULES, FORBIDDEN)
    env = {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "env_equal": True,
        "files_equal": True,
        "forbidden": [],
        "postgres_imported": False,
    }


def test_passive_modules_do_not_import_forbidden_domains_or_psycopg():
    imported = set()
    for filename in ("canonical.py", "contracts.py", "ledger.py", "__init__.py"):
        tree = ast.parse((MODULE_DIR / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not any(name.startswith(FORBIDDEN) for name in imported)
    assert "psycopg" not in imported


def test_postgres_backend_requires_explicit_initialize(monkeypatch):
    from runtime.experiments.cognitive_gain import postgres_store

    calls = []
    monkeypatch.setattr(postgres_store.psycopg, "connect", lambda *a, **k: calls.append((a, k)))
    postgres_store.PostgresCausalLedgerStore(
        dsn="postgresql://isolated.invalid/test",
        schema="rnfe_cg_test_no_io",
    )
    assert calls == []
