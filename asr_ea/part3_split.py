"""Part 3: split the labeled sample into dev (tune the judge) and test (measure it once).

Stratified by subtype; saved in the `split` column of llm_labels and reused afterwards.
"""
import numpy as np
import pandas as pd

from .report import section, show
from .sheets import read_tab_df, write_columns


def run(ctx):
    section("Part 3: dev/test split")
    p3 = ctx.cfg["part3"]
    L = ctx.labels()
    man = L.man_labels
    llm_tab = read_tab_df(ctx.sh, "llm_labels")
    saved = llm_tab.set_index("pair_id")["split"] if "split" in llm_tab.columns else pd.Series(dtype=object)
    saved = saved[saved.isin(["dev", "test"])]
    labeled = man[man.labeled]

    if len(saved) and not p3["resplit"]:
        split = saved.reindex(labeled.pair_id).fillna("")
        late = (split == "").sum()
        print("Using the saved split (set part3.resplit: true to redraw it)."
              + (f" {late} rows labeled after the split have none and are left out." if late else ""))
    else:
        rng = np.random.default_rng(p3["seed"])
        frac = min(1, p3["dev_size"] / max(len(labeled), 1))
        dev, remainders = [], []
        for _, g in labeled.groupby("primary"):
            ids = list(rng.permutation(g.pair_id.tolist()))
            k = int(np.floor(len(ids) * frac))
            dev += ids[:k]
            remainders.append((len(ids) * frac - k, ids[k:]))
        for _, ids in sorted(remainders, key=lambda x: -x[0]):   # largest remainder fills up to dev_size
            if len(dev) < p3["dev_size"] and ids:
                dev.append(ids[0])
        split = pd.Series("test", index=labeled.pair_id.values)
        split[split.index.isin(dev)] = "dev"
        col = pd.Series("", index=llm_tab.pair_id[llm_tab.in_manual_sample == "Y"].values)
        col.update(split)
        write_columns(ctx.sh, "llm_labels", col.rename_axis("pair_id").to_frame("split"))
        print(f"Saved a new split: {(split == 'dev').sum()} dev, {(split == 'test').sum()} test.")

    man["split"] = man.pair_id.map(split).fillna("")
    lab = man[man.labeled]
    show(ctx, "Rows per category and split", pd.crosstab(lab.primary, lab.split, margins=True).reset_index(),
         csv="part3_split_counts.csv")
