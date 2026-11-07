#!/usr/bin/env python3
"""Validate exported files against products.yaml.

Checks exact pixel size, format, colour mode, size caps and the naming pattern
for every file, and reports missing products. Exit code 0 = all runs pass.

    python scripts/validate_exports.py --config config.yaml [--run-id 20261106_run01]
(`pipeline validate` runs the same checks.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import ConfigError, load_config
from pipeline.validate import validate_output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--products", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-errors", type=int, default=50)
    args = ap.parse_args(argv)
    overrides = {"runtime.work_dir": args.work_dir} if args.work_dir else {}
    try:
        cfg = load_config(args.config, args.products, overrides)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    ok = True
    for rep in validate_output(cfg, args.run_id):
        print(rep.summary())
        for err in rep.errors[: args.max_errors]:
            print(f"  ERROR {err}")
        for warn in rep.warnings[:10]:
            print(f"  warn  {warn}")
        ok = ok and rep.ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
