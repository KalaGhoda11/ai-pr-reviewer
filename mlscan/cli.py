"""Command-line interface for the standalone vulnerability scanner.

Examples:
    python -m mlscan path/to/file.py
    python -m mlscan --code "eval(user_input)"
    cat file.py | python -m mlscan -
"""

from __future__ import annotations

import argparse
import json
import sys

from mlscan.scanner import ModelNotTrained, scan


def _read_input(args) -> tuple[str, str]:
    if args.code is not None:
        return "<--code>", args.code
    if args.target == "-":
        return "<stdin>", sys.stdin.read()
    with open(args.target, encoding="utf-8", errors="replace") as fh:
        return args.target, fh.read()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="mlscan", description="Offline ML scanner for common code vulnerabilities.")
    p.add_argument("target", nargs="?", help="path to a source file, or '-' for stdin")
    p.add_argument("--code", help="scan this literal code string instead of a file")
    p.add_argument("--json", action="store_true", help="output raw JSON")
    p.add_argument("--threshold", type=float, default=0.50, help="min confidence to flag")
    args = p.parse_args(argv)

    if args.target is None and args.code is None:
        p.error("provide a file path, '-' for stdin, or --code")

    name, code = _read_input(args)
    try:
        result = scan(code, threshold=args.threshold)
    except ModelNotTrained as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
        return 1 if result["is_vulnerable"] else 0

    if not result["is_vulnerable"]:
        print(f"[OK] {name}: no known vulnerability detected "
              f"(top: {result['top_prediction']} {result['top_confidence']:.0%})")
        return 0

    print(f"[!] {name}: {len(result['findings'])} potential vulnerability(ies):")
    for f in result["findings"]:
        print(f"  - {f['cwe']}  {f['name']}  ({f['confidence']:.0%} confidence)")
        print(f"      OWASP {f['owasp']} — {f['description']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
