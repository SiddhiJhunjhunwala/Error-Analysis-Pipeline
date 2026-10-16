"""Part 6: results tables from the judge's labels, written to the results tab.

Run Part 4 with judge.scope: all first, so every run is labeled the same way.
"""
import datetime

import pandas as pd

from .report import section, show
from .sheets import read_tab_df, with_retry, write_blocks


def meaning_stats(g):
    return pd.Series({"labeled_errors": len(g),
                      "meaning_changed": round(g.meaning.isin(["Y", "Y_toward_correct"]).mean(), 3),
                      "toward_correct": round((g.meaning == "Y_toward_correct").mean(), 3),
                      "toward_correct_rows": int((g.meaning == "Y_toward_correct").sum())})


def run(ctx):
    section("Part 6: results tables")
    L = ctx.labels()
    results_tab = ctx.cfg["part6"]["results_tab"]
    pairs_tab = read_tab_df(ctx.sh, "pairs")
    llm_tab = read_tab_df(ctx.sh, "llm_labels")
    lab, _ = L.resolve(llm_tab, {"broader": "llm_broader", "primary": "llm_cat_primary",
                                 "secondary": "llm_cat_secondary", "meaning": "llm_meaning_changed"})
    d = pairs_tab.merge(lab, on="pair_id", how="left").fillna("")
    d["wer_num"] = pd.to_numeric(d.wer, errors="coerce")
    err = d[d.has_error == "Y"]
    coverage = (err.primary != "").mean() if len(err) else 0
    if coverage < 1:
        lt = llm_tab.set_index("pair_id").reindex(err.pair_id)
        not_run = (lt["judge_prompt_version"].fillna("") == "").sum() if "judge_prompt_version" in lt else len(err)
        invalid = (err.primary == "").sum() - not_run
        print(f"Warning: {coverage:.0%} of error rows have a valid judge label. Not labeled yet: {not_run} "
              f"(run Part 4 with --scope all). Invalid labels: {invalid}.")
    el = err[err.primary != ""]
    if el.empty:
        raise RuntimeError("No judge labels on error rows yet. Run Part 4 with --scope all.")
    order = list(dict.fromkeys(pairs_tab.run_key))

    per_run = d.groupby("run_key").agg(model=("model", "first"), prompt=("prompt_condition", "first"),
                                       sees_story=("sees_story", "first"), pairs=("pair_id", "size"),
                                       error_rows=("has_error", lambda s: (s == "Y").sum()),
                                       mean_wer=("wer_num", "mean")).reindex(order)
    per_run["error_rate"] = (per_run.error_rows / per_run.pairs).round(3)
    per_run["mean_wer"] = per_run.mean_wer.round(3)
    per_run["judge_labeled"] = el.groupby("run_key").size().reindex(order).fillna(0).astype(int)
    per_run = per_run.reset_index()

    share = lambda col: (pd.crosstab(el.run_key, el[col], normalize="index").reindex(order).round(3)
                         .rename_axis(None, axis=1).reset_index())
    tables = [("Per run (WER and error rate)", per_run, "part6_per_run.csv"),
              ("Broader category: share of labeled error rows", share("broader"), "part6_broader_share.csv"),
              ("Subtype: share of labeled error rows", share("primary"), "part6_subtype_share.csv")]
    if (el.meaning != "").any():
        by_run = pd.DataFrame({k: meaning_stats(g) for k, g in el.groupby("run_key")}).T.reindex(order)
        by_story = pd.DataFrame({k: meaning_stats(g) for k, g in el.groupby("sees_story")}).T
        tables += [("Meaning changes by run (share of labeled error rows)",
                    by_run.rename_axis("run_key").reset_index(), "part6_meaning_by_run.csv"),
                   ("Meaning changes by whether the ASR saw the story",
                    by_story.rename_axis("sees_story").reset_index(), "part6_meaning_by_story.csv")]
    else:
        print("The judge gave no meaning_changed labels, so the meaning tables are skipped.")

    for title, t, csv in tables:
        show(ctx, title, t, csv=csv)

    existing = {w.title: w for w in ctx.sh.worksheets()}
    ws = existing.get(results_tab) or with_retry(ctx.sh.add_worksheet, title=results_tab, rows=200, cols=30)
    write_blocks(ctx.sh, ws, [(t, df) for t, df, _ in tables], start_row=1, start_col=1,
                 stamp=f"Built by Part 6 on {datetime.datetime.now():%Y-%m-%d %H:%M} from judge labels "
                       f"({coverage:.0%} of error rows labeled).")
    print(f"\nWritten to the '{results_tab}' tab. CSV copies are in {ctx.out_dir}")
