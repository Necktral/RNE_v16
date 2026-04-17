import ast
from pathlib import Path


def _python_files(root: Path):
    for path in root.rglob("*.py"):
        yield path


def _imports_in_file(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield module


def test_runtime_and_exocortex_must_not_import_archive():
    repo = Path(__file__).resolve().parents[2]
    for scope in ("runtime", "exocortex"):
        for py_file in _python_files(repo / scope):
            for imported in _imports_in_file(py_file):
                assert not imported.startswith("archive"), f"{py_file} imports archive: {imported}"


def test_runtime_must_not_import_exocortex():
    repo = Path(__file__).resolve().parents[2]
    for py_file in _python_files(repo / "runtime"):
        for imported in _imports_in_file(py_file):
            assert not imported.startswith("exocortex"), f"{py_file} imports exocortex: {imported}"

