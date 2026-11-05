"""Offline source and credential-hygiene checks, also used by GitHub Actions."""
import ast
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
errors = []
parsed = 0
patterns = [
    re.compile(rb"Bearer\s+[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKID[A-Za-z0-9]{25,}"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{25,}"),
]
for relative in filter(None, paths):
    path = ROOT / relative
    if not path.is_file():
        continue
    if path.name in {"credential.txt", "brain_credentials.txt"} or path.suffix == ".pyc":
        errors.append(f"Forbidden tracked credential/generated file: {relative}")
    data = path.read_bytes()
    if path.suffix == ".py":
        try:
            ast.parse(data, filename=relative)
            parsed += 1
        except SyntaxError as error:
            errors.append(f"{relative}:{error.lineno}: {error.msg}")
    if any(pattern.search(data) for pattern in patterns):
        errors.append(f"Potential hardcoded provider credential: {relative}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Parsed {parsed} Python files; tracked-file credential checks passed.")
