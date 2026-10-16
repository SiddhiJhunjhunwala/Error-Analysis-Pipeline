"""Part 5: judge vs manual agreement on dev and test (broader, subtype, meaning changed).

Writes the tables to the agreement tab from L11 down; dev disagreements go to outputs/ for reading.
Test disagreements stay hidden unless part5.show_test_disagreements is true, so they can't leak into tuning.
"""
import datetime

import numpy as np
import pandas as pd

from .report import section, show
from .sheets import read_tab_df, write_blocks


def cohen_kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    if len(a) == 0:
        return np.nan
    po = (a == b).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(a) | set(b))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def run(ctx):
    section("Part 5: agreement")
    p5 = ctx.cfg["part5"]
    L = ctx.labels()
    llm_tab = read_tab_df(ctx.sh, "llm_labels")
    llm_res, _ = L.resolve(llm_tab[llm_tab.in_manual_sample == "Y"],
                           {"broader": "llm_broader", "primary": "llm_cat_primary",
                            "secondary": "llm_cat_secondary", "meaning": "llm_meaning_changed"})
    llm_res["rationale"] = llm_tab.set_index("pair_id").reindex(llm_res.pair_id)["llm_rationale"].values
    df = L.man_labels.merge(llm_res, on="pair_id", suffixes=("_m", "_j"))
    df = df[df.labeled & df.split.isin(["dev", "test"])]
    judged = llm_tab.set_index("pair_id").reindex(df.pair_id)["judge_prompt_version"].fillna("") != ""
    n_invalid = int((judged.values & (df.primary_j == "").values).sum())
    if n_invalid:
        print(f"{n_invalid} rows have an invalid judge label and are left out (llm_label_valid = N).")
    df = df[df.primary_j != ""]
    if df.empty:
        raise RuntimeError("No rows have both a manual and a judge label yet. Run Part 4 first.")

    levels = [("broader", "broader_m", "broader_j"), ("subtype", "primary_m", "primary_j")]
    if L.has_manual_meaning:
        levels.append(("meaning changed", "meaning_m", "meaning_j"))
    overall = []
    for sp in ["dev", "test"]:
        d = df[df.split == sp]
        for name, a, b in levels:
            dd = d[(d[a] != "") & (d[b] != "")]
            overall.append({"split": sp, "level": name, "rows": len(dd),
                            "agreement": round((dd[a] == dd[b]).mean(), 3) if len(dd) else None,
                            "kappa": round(cohen_kappa(dd[a], dd[b]), 3) if len(dd) else None})
    overall = pd.DataFrame(overall)
    show(ctx, "Overall", overall, csv="part5_overall.csv")

    report = "test" if (df.split == "test").any() else "dev"
    d = df[df.split == report]
    m, j = d.primary_m, d.primary_j
    per_cat = pd.DataFrame([{"subtype": k, "manual": int((m == k).sum()), "judge": int((j == k).sum()),
                             "both": int(((m == k) & (j == k)).sum())} for k in sorted(set(m) | set(j))])
    per_cat["precision"] = (per_cat.both / per_cat.judge.replace(0, np.nan)).round(3)
    per_cat["recall"] = (per_cat.both / per_cat.manual.replace(0, np.nan)).round(3)
    per_cat["f1"] = (2 * per_cat.precision * per_cat.recall / (per_cat.precision + per_cat.recall)).round(3)
    confusion = pd.crosstab(m.rename("manual \\ judge"), j.rename("judge"), margins=True,
                            margins_name="total").reset_index()
    confusion.columns.name = None
    by_run = (d.assign(match=m == j).groupby("run_key").agg(rows=("match", "size"), agreement=("match", "mean"))
              .round(3).reset_index())
    show(ctx, f"Per subtype, {report} set (manual = reference)", per_cat, csv=f"part5_per_subtype_{report}.csv")
    show(ctx, f"Confusion matrix, {report} set (rows: manual, columns: judge)", confusion,
         csv=f"part5_confusion_{report}.csv")
    show(ctx, f"Subtype agreement by run, {report} set", by_run, csv=f"part5_by_run_{report}.csv")

    tk = overall[(overall.split == "test") & (overall.level == "subtype")].kappa
    if len(tk) and pd.notna(tk.iloc[0]):
        print(f"\nTest kappa (subtype) = {tk.iloc[0]:.3f}: "
              + ("meets" if tk.iloc[0] >= p5["target_kappa"] else "below") + f" the target of {p5['target_kappa']}.")

    cols = ["pair_id", "reference_transcript", "asr_output", "primary_m", "primary_j", "rationale", "notes"]
    names = {"primary_m": "manual", "primary_j": "judge", "rationale": "judge_reason", "notes": "your_note"}
    for sp in ["dev"] + (["test"] if p5["show_test_disagreements"] else []):
        dis = df[(df.split == sp) & (df.primary_m != df.primary_j)][cols].rename(columns=names)
        show(ctx, f"{sp} disagreements on subtype: {len(dis)}", dis, csv=f"part5_{sp}_disagreements.csv",
             max_rows=25)

    write_blocks(ctx.sh, ctx.sh.worksheet("agreement"),
                 [("Overall (dev and test)", overall),
                  (f"Per subtype, {report} set (manual = reference)", per_cat),
                  (f"Confusion matrix, {report} set (rows: manual, columns: judge)", confusion),
                  (f"Subtype agreement by run, {report} set", by_run)],
                 start_row=11, start_col=12,
                 stamp=f"Computed by Part 5 on {datetime.datetime.now():%Y-%m-%d %H:%M}. "
                       "Snapshot values: re-run after labels change.")
    print(f"\nWritten to the agreement tab. CSV copies are in {ctx.out_dir}")
