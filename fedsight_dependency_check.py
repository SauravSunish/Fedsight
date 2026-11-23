#!/usr/bin/env python3
"""FedSight dependency diagnostic. Uses only Python's standard library.

Place this file in the FedSight project root (beside app.py and requirements.txt)
and run it with the same Python interpreter used to launch Streamlit.
It diagnoses; it does not install, modify, or delete anything.
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata as metadata
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "build", "dist",
    "node_modules", ".streamlit", ".pytest_cache", "tmp", "backups", "data",
}

# Common import-name -> pip distribution-name differences.
IMPORT_TO_DIST = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "faiss": "faiss-cpu",
    "dotenv": "python-dotenv",
    "Crypto": "pycryptodome",
    "gi": "PyGObject",
    "torchreid": "Official source: https://github.com/KaiyangZhou/deep-person-reid",
}

# Imports worth smoke-testing for this prototype. These are imported one by one
# so a failure in one package does not prevent checking the others.
SMOKE_IMPORTS = [
    "streamlit",
    "numpy",
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "torchreid",
    "PIL",
    "gdown",
    "tensorboard",
]


def norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distributions() -> dict[str, str]:
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            found[norm_dist(name)] = dist.version
    return found


def parse_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("-", "http://", "https://")):
            continue
        # Handles common requirements such as package, package[extra]>=1.2,
        # and package @ URL. Complex pip options are intentionally skipped.
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            names.append(match.group(1))
    return names


def local_module_names(root: Path) -> set[str]:
    names: set[str] = set()
    for p in root.iterdir():
        if p.is_file() and p.suffix == ".py":
            names.add(p.stem)
        elif p.is_dir() and p.name not in SKIP_DIRS and (p / "__init__.py").exists():
            names.add(p.name)
    return names


def scan_project_imports(root: Path) -> tuple[set[str], list[str]]:
    imports: set[str] = set()
    parse_errors: list[str] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except (SyntaxError, OSError) as exc:
            parse_errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
    return imports, parse_errors


def check_spec(name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return False, "not found"
        origin = spec.origin or "namespace/built-in"
        return True, origin
    except Exception as exc:  # Some broken packages throw while resolving submodules.
        return False, f"lookup failed: {type(exc).__name__}: {exc}"


def main() -> int:
    print("=" * 72)
    print("FedSight dependency diagnostic")
    print("=" * 72)
    print(f"Project folder : {ROOT}")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Python exe     : {sys.executable}")
    print()

    dists = installed_distributions()
    req_names = parse_requirements(ROOT / "requirements.txt")
    if req_names:
        print("1) requirements.txt distribution check")
        missing_req = []
        for name in req_names:
            version = dists.get(norm_dist(name))
            if version:
                print(f"  OK       {name} ({version})")
            else:
                print(f"  MISSING  {name}")
                missing_req.append(name)
        if not missing_req:
            print("  All parsed requirements are installed.")
    else:
        print("1) requirements.txt: not found or no simple package entries found.")
        missing_req = []
    print()

    print("2) Project import scan")
    project_imports, parse_errors = scan_project_imports(ROOT)
    local_names = local_module_names(ROOT)
    stdlib = getattr(sys, "stdlib_module_names", set())
    missing_imports: list[tuple[str, str]] = []
    for name in sorted(project_imports):
        if name in stdlib or name in local_names:
            continue
        ok, detail = check_spec(name)
        if ok:
            print(f"  OK       {name}")
        else:
            suggested = IMPORT_TO_DIST.get(name, name)
            print(f"  MISSING  {name}  | likely pip package: {suggested} | {detail}")
            missing_imports.append((name, suggested))
    if not project_imports:
        print("  No project imports found. Is this the project root?")
    for err in parse_errors:
        print(f"  PARSE    {err}")
    print()

    print("3) Important runtime import tests")
    smoke_failures: list[tuple[str, str]] = []
    for name in SMOKE_IMPORTS:
        try:
            importlib.import_module(name)
            print(f"  OK       import {name}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED   import {name} -> {message}")
            smoke_failures.append((name, message))
    print()

    print("4) Summary")
    print(f"  Missing requirements entries : {len(missing_req)}")
    print(f"  Missing project top-level imports : {len(missing_imports)}")
    print(f"  Failed runtime import tests : {len(smoke_failures)}")
    if smoke_failures:
        print("\n  Runtime failures to resolve:")
        for name, message in smoke_failures:
            if name.startswith("torchreid"):
                print("   - Torchreid: use the official KaiyangZhou/deep-person-reid source;")
                print("     a similarly named pip package may not contain torchreid.utils.")
            print(f"   - {name}: {message}")
    if missing_imports:
        print("\n  Likely pip installs (review before running):")
        for name, package in missing_imports:
            print(f"   - {name}: {package}")
    print("\nThis is a diagnostic, not a complete proof that every code path will run.")
    print("It does not install packages or change Windows security settings.")
    return 1 if (missing_req or missing_imports or smoke_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
