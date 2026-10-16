"""Part 5: judge vs manual agreement on dev and test (broader, subtype, meaning changed).

Writes the tables to the agreement tab from L11 down; dev disagreements go to outputs/ for reading.
Test disagreements stay hidden unless part5.show_test_disagreements is true, so they can't leak into tuning.

Taxonomy rules agreed 2026-10-16:
- Headline kappa = broader category on rows whose manual label is one of the 7 ASR error categories
  (the data issues "no real error", "reference issue", "uncertain manual transcript" are left out; D4).
- "Background speech" needs the audio, so it is merged with "Unrelated fluent speech" before comparing.
- Toward-correct rows (Y_toward_correct) are counted per run from the manual labels (Dilek's check, A3).
"""
import datetime

import numpy as np
import pandas as pd

from .report import section, show
from .sheets import read_tab_df, write_blocks

DATA_ISSUES = {"no real error", "reference issue", "uncertain manual transcript"}
MERGED_SUBTYPES = {"background speech": "Unrelated fluent speech"}   # judge can't hear the audio
HEADLINE = "broader, ASR errors only"


def cohen_kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    if len(a) == 0:
        return np.nan
    po = (a == b).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in set(a) | set(b))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def merge_subtypes(s):
    return s.map(lambda v: MERGED_SUBTYPES.get(str(v).strip().lower(), v))


def toward_correct_tables(L):
    """Counts and rows of manual labels with meaning_changed = Y_toward_correct, by run."""
    man = L.man_labels[L.man_labels.labeled]
    tc = man.meaning == "Y_toward_correct"
    by_run = (man.assign(toward_correct=tc).groupby("run_key")
              .agg(labeled_rows=("toward_correct", "size"), toward_correct=("toward_correct", "sum"))
              .reset_index())
    by_run["share"] = (by_run.toward_correct / by_run.labeled_rows).round(3)
    rows = man[tc][["pair_id", "run_key", "question_text", "reference_transcript", "asr_output",
                    "broader", "primary", "notes"]]
    return by_run, rows


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
    df = df[df.primary_j != ""].copy()
    if df.empty:
        raise RuntimeError("No rows have both a manual and a judge label yet. Run Part 4 first.")
    df["primary_m"], df["primary_j"] = merge_subtypes(df.primary_m), merge_subtypes(df.primary_j)
    is_error = ~df.broader_m.str.strip().str.lower().isin(DATA_ISSUES)

    levels = [(HEADLINE, "broader_m", "broader_j", True),
              ("broader, all categories", "broader_m", "broader_j", False),
              ("subtype", "primary_m", "primary_j", False)]
    if L.has_manual_meaning:
        levels.append(("meaning changed", "meaning_m", "meaning_j", False))
    overall = []
    for sp in ["dev", "test"]:
        d = df[df.split == sp]
        for name, a, b, errors_only in levels:
            dd = d[(d[a] != "") & (d[b] != "")]
            if errors_only:
                dd = dd[is_error.reindex(dd.index)]
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
    bm, bj = d.broader_m, d.broader_j
    confusion_broader = pd.crosstab(bm.rename("manual \\ judge"), bj.rename("judge"), margins=True,
                                    margins_name="total").reset_index()
    confusion_broader.columns.name = None
    by_run = (d.assign(match=bm == bj, match_sub=m == j).groupby("run_key")
              .agg(rows=("match", "size"), broader_agreement=("match", "mean"),
                   subtype_agreement=("match_sub", "mean"))
              .round(3).reset_index())
    show(ctx, f"Per subtype, {report} set (manual = reference)", per_cat, csv=f"part5_per_subtype_{report}.csv")
    show(ctx, f"Confusion matrix, broader category, {report} set (rows: manual, columns: judge)",
         confusion_broader, csv=f"part5_confusion_broader_{report}.csv")
    show(ctx, f"Confusion matrix, subtype, {report} set (rows: manual, columns: judge)", confusion,
         csv=f"part5_confusion_{report}.csv")
    show(ctx, f"Agreement by run, {report} set", by_run, csv=f"part5_by_run_{report}.csv")

    tk = overall[(overall.split == report) & (overall.level == HEADLINE)].kappa
    if len(tk) and pd.notna(tk.iloc[0]):
        print(f"\n{report.capitalize()} kappa ({HEADLINE}) = {tk.iloc[0]:.3f}: "
              + ("meets" if tk.iloc[0] >= p5["target_kappa"] else "below") + f" the target of {p5['target_kappa']}.")

    blocks = [("Overall (dev and test)", overall),
              (f"Per subtype, {report} set (manual = reference)", per_cat),
              (f"Confusion matrix, broader category, {report} set (rows: manual, columns: judge)", confusion_broader),
              (f"Confusion matrix, subtype, {report} set (rows: manual, columns: judge)", confusion),
              (f"Agreement by run, {report} set", by_run)]
    if L.has_manual_meaning:
        tc_by_run, tc_rows = toward_correct_tables(L)
        show(ctx, f"Toward-correct errors by run, manual labels ({int(tc_by_run.toward_correct.sum())} rows)",
             tc_by_run, csv="part5_toward_correct_by_run.csv")
        show(ctx, "Toward-correct rows (manual labels)", tc_rows, csv="part5_toward_correct_rows.csv", max_rows=25)
        blocks.append(("Toward-correct errors by run, manual labels", tc_by_run))

    cols = ["pair_id", "reference_transcript", "asr_output", "primary_m", "primary_j", "rationale", "notes"]
    names = {"primary_m": "manual", "primary_j": "judge", "rationale": "judge_reason", "notes": "your_note"}
    for sp in ["dev"] + (["test"] if p5["show_test_disagreements"] else []):
        dis = df[(df.split == sp) & (df.primary_m != df.primary_j)][cols].rename(columns=names)
        show(ctx, f"{sp} disagreements on subtype: {len(dis)}", dis, csv=f"part5_{sp}_disagreements.csv",
             max_rows=25)

    write_blocks(ctx.sh, ctx.sh.worksheet("agreement"), blocks,
                 start_row=11, start_col=12,
                 stamp=f"Computed by Part 5 on {datetime.datetime.now():%Y-%m-%d %H:%M}. "
                       "Snapshot values: re-run after labels change.")
    print(f"\nWritten to the agreement tab. CSV copies are in {ctx.out_dir}")
