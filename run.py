#!/usr/bin/env python3
"""Run parts of the ASR error analysis, as set in config.yaml.

  python run.py                      # the parts listed under run.parts in config.yaml
  python run.py --parts 4,5          # override the parts
  python run.py --parts 4 --scope test
  python run.py -c other_config.yaml
"""
import argparse
import sys
import time

from dotenv import load_dotenv

from asr_ea import judge, labels, part1_build, part3_split, part5_agreement, part6_results
from asr_ea.config import load_config
from asr_ea.context import Context
from asr_ea.sheets import connect

PARTS = {1: ("build/refresh tabs", part1_build.run), 2: ("label check", labels.run),
         3: ("dev/test split", part3_split.run), 4: ("LLM judge", judge.run),
         5: ("agreement", part5_agreement.run), 6: ("results tables", part6_results.run)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.yaml", help="config file (default: config.yaml)")
    ap.add_argument("--parts", help="comma-separated parts to run, e.g. 4,5 (overrides run.parts)")
    ap.add_argument("--scope", choices=["dev", "test", "all"], help="judge scope for Part 4 (overrides judge.scope)")
    args = ap.parse_args()

    load_dotenv()  # API keys from .env
    parts = [int(p) for p in args.parts.split(",")] if args.parts else None
    try:
        cfg = load_config(args.config, parts=parts, scope=args.scope)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Config error: {e}")
    run_parts = sorted(cfg["run"]["parts"])
    print(f"Parts to run: {', '.join(f'{p} ({PARTS[p][0]})' for p in run_parts)}")
    if 4 in run_parts:
        j = cfg["judge"]
        print(f"Judge: {j['provider']} / {j['model']}, scope = {j['scope']}")

    ctx = Context(cfg, connect(cfg))
    for p in run_parts:
        t0 = time.time()
        PARTS[p][1](ctx)
        print(f"[Part {p} finished in {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
