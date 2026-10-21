#!/usr/bin/env python3
"""
check_no_hardcoding.py — the single audit oracle for the "no hardcoded disease
values" invariant.

Scans .py files and the tracked pipeline .ipynb notebooks for functional
hardcoded disease-group labels, outcome-column names, network->disease
mappings, output directories, and expected counts (len(x) != 18). Exits non-zero (with a file:line report) if any violation is found,
so an audit is one deterministic command instead of eyeballed greps.

Ignored: comment lines, triple-quoted docstrings, argparse help/description text,
parser.error messages, the resolver module `disease_focus.py` (which legitimately
defines the canonical labels), and any line carrying `# noqa: hardcode`.

Usage:
    python check_no_hardcoding.py            # scan the repo, exit 1 on violations
    python check_no_hardcoding.py --root .   # scan a specific root
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

# The resolver is the one place canonical labels are allowed to appear.
ALLOWLIST_FILES = {"disease_focus.py", "check_no_hardcoding.py"}

# Functional patterns that must never appear in executable code.
FORBIDDEN = [
    ("network->disease ternary",
     re.compile(r"""['"](diarrheal|hypertension|infectious|ncd)['"]\s+if\b""", re.I)),
    ("hardcoded outcome-column literal",
     re.compile(r"""\b(diarrheal_count|hypertension_count|diarrheal_count_adjusted|hypertension_count_adjusted)\b""")),
    ("hardcoded weekly-file disease label",
     re.compile(r"""_weekly_(infectious|ncd|hypertension|diarrheal)_adjusted""")),
    ("hardcoded disease-group comparison",
     re.compile(r"""==\s*['"](Diarrheal Diseases|Hypertension|diarrheal|hypertension)['"]""", re.I)),
    ("hardcoded disease-category dict entry",
     re.compile(r"""['"](INF|NCD|SYNINF|SYNNCD)['"]\s*:\s*['"](Diarrheal Diseases|Hypertension|hypertension|diarrheal)['"]""")),
    # Each network/mode run writes to its own directory. A literal "out" that is
    # not an environment fallback sends one run's results into another run's
    # directory, or reads a stale file from it, without any error. Legitimate
    # uses read HSA_OUT_DIR / PIPELINE_OUT_DIR first, so those lines are exempt
    # by the guard below.
    ("hardcoded output directory",
     re.compile(r"""(?:["']out/|Path\(\s*["']out["']\s*\)|,\s*["']out["']\s*\)|/\s*["']out["'])""")),
    # The number of HSAs, anchors or facilities is an output of the run. Asserting
    # it equals a literal (len(x) != 18) rejects every run whose delineation
    # differs, e.g. a sensitivity setting that adds an anchor. Compare against
    # the run's own allocation or GeoJSON instead. Literals 0 and 1 are emptiness
    # and uniqueness tests, not expected sizes, and stay allowed.
    ("hardcoded expected count",
     re.compile(r"""(?:\blen\((?:[^()]|\([^()]*\))*\)|\.nunique\(\)|\.shape\[0\])\s*[!=]=\s*(?:[2-9]|[1-9]\d+)(?![\d.])"""
                r"""|(?<![\d.])(?:[2-9]|[1-9]\d+)\s*[!=]=\s*(?:len\(|\S+\.nunique\(\)|\S+\.shape\[0\])""")),
]

# A line that consults the environment is resolving the run directory properly;
# the "out" it names is only the fallback.
ENV_AWARE = re.compile(r"(environ|getenv|HSA_OUT_DIR|PIPELINE_OUT_DIR|args\.out_dir|out_dir)")

# Lines that look like documentation/help rather than executable hardcoding.
IGNORE_LINE = re.compile(r"(help\s*=|description\s*=|parser\.error|#\s*noqa:\s*hardcode|->|e\.g\.)")


def _exempt(label: str, line: str) -> bool:
    """Rule-specific exemptions beyond the generic documentation filter."""
    if label == "hardcoded output directory":
        return bool(ENV_AWARE.search(line))
    return False


def _iter_code_lines(src: str):
    """Yield (lineno, line) for executable lines, skipping comments and
    triple-quoted docstrings/strings (a crude but effective toggle)."""
    in_doc = False
    quote = None
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        # Toggle docstring state on lines that open/close a triple quote.
        triples = re.findall(r'"""|\'\'\'', line)
        if in_doc:
            # still inside a docstring; check if it closes here
            for t in triples:
                if t == quote:
                    in_doc = False
                    quote = None
                    break
            continue
        if triples:
            # count how many triple-quotes; odd => opens an unclosed docstring
            opens = [t for t in triples]
            # if the line has a single triple-quote (or odd count), enter docstring
            if len(opens) % 2 == 1:
                in_doc = True
                quote = opens[0]
            # skip the line itself (likely docstring content/definition)
            continue
        if stripped.startswith("#"):
            continue
        yield i, line


def _scan_text(path: str, src: str, violations: list):
    if Path(path).name in ALLOWLIST_FILES:
        return
    for lineno, line in _iter_code_lines(src):
        if IGNORE_LINE.search(line):
            continue
        for label, rx in FORBIDDEN:
            if rx.search(line) and not _exempt(label, line):
                violations.append((path, lineno, label, line.strip()[:100]))


def _notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text())
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            cells.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(cells)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit for hardcoded disease values.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)

    violations: list = []
    # Runtime artefacts are copies the runner writes beside the sources it ran;
    # scanning them reports the same line twice and can flag a superseded copy.
    def _generated(name: str) -> bool:
        return name.startswith("_in_") or name.endswith("_executed.ipynb")

    for py in sorted(root.glob("*.py")):
        _scan_text(py.name, py.read_text(errors="ignore"), violations)
    for nb in sorted(n for n in root.glob("*.ipynb") if not _generated(n.name)):
        try:
            _scan_text(nb.name, _notebook_code(nb), violations)
        except Exception as e:  # malformed notebook shouldn't crash the audit
            print(f"  WARN: could not scan {nb.name}: {e}")

    if not violations:
        print("check_no_hardcoding: PASS (no functional hardcoded values found)")
        return 0

    print(f"check_no_hardcoding: FAIL ({len(violations)} violation(s))")
    for path, lineno, label, text in violations:
        print(f"  {path}:{lineno}: [{label}] {text}")
    print("\nRoute the value through disease_focus.py (canonical_group / "
          "weekly_outcome_col / daily_outcome_col), derive counts from the run's "
          "own outputs, or add '# noqa: hardcode - <reason>' if the value is "
          "genuinely fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
