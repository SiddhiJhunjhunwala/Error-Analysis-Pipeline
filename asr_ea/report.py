"""Printing tables to the terminal and saving CSV copies."""
from pathlib import Path

import pandas as pd


def show(ctx, title, df, csv=None, max_rows=60):
    """Print a titled table; with `csv`, also save it to the output folder."""
    print(f"\n{title}")
    if df is None or len(df) == 0:
        print("  (empty)")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 60):
            print(df.head(max_rows).to_string())
        if len(df) > max_rows:
            print(f"  ... {len(df) - max_rows} more rows" + (f" (all in {csv})" if csv else ""))
    if csv is not None and df is not None:
        path = Path(ctx.out_dir) / csv
        df.to_csv(path, index=False)


def section(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)
