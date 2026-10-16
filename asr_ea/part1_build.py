"""Part 1: one row per answer x ASR output (pairs), the manual sample, and the 4 tabs.

First run: creates pairs, manual_annotation, llm_labels and agreement.
Later runs: rewrites pairs and llm_labels only (labels in llm_labels are kept); the manual tab is never touched.
part1.overwrite = true rebuilds all four tabs, including the manual labels.
"""
import datetime
import hashlib
import re
from collections import Counter

import numpy as np
import pandas as pd

from .report import show
from .sheets import (CELL_LIMIT, NEW_TABS, col_width, df_rows, fmt, grid, read_tab_df, table_requests,
                     with_retry, write_rows)
from .text import clean_text, col_letter, colkey, slug, to_number, word_edit_ops, yn

BASE_RULES = {
    "file_id":           lambda k: k == "fileid",
    "story_title":       lambda k: k == "storytitle",
    "full_story":        lambda k: k == "fullstory",
    "type":              lambda k: k == "type",
    "repeat_no":         lambda k: k.startswith("repeat"),
    "question_text":     lambda k: k == "questiontext",
    "start_s":           lambda k: k.startswith("starttime"),
    "end_s":             lambda k: k.startswith("endtime"),
    "manual_transcript": lambda k: "transcription" in k or "tanscription" in k,
}
SETUP_COLS = ["model", "finetuned", "prompt_condition", "sees_story", "sees_question", "is_probe"]
LLM_COLS = ["pair_id", "run_key", "in_manual_sample", "llm_cat_primary", "llm_cat_secondary", "llm_rationale",
            "judge_model", "judge_prompt_version", "run_timestamp",
            "llm_broader", "llm_meaning_changed", "llm_label_valid", "split"]
PAIRS_FMT = dict(
    widths={"pair_id": 13, "file_id": 12, "story_title": 18, "question_text": 40, "run_key": 40,
            "model": 22, "prompt_condition": 18, "manual_transcript": 45, "reference_text": 45,
            "asr_output": 45, "wer": 8},
    hidden={"full_story"},
    notes={"reference_text": "What the ASR should have output: the manual transcript, or blank for empty-audio runs.",
           "wer": "After normalization (see part1 in config.yaml). Blank when the reference is empty.",
           "has_error": "Y when the normalized ASR output differs from the normalized reference.",
           "ref_has_gap": "Y when the manual transcript has a gap (__) the transcriber couldn't hear.",
           "ref_had_teacher_speech": "Y when teacher speech was removed from the reference before comparing.",
           "source_row": "Row number in the original tab (the full story is there)."})
LLM_FMT = dict(
    widths={"pair_id": 13, "run_key": 40, "in_manual_sample": 11, "llm_cat_primary": 24, "llm_cat_secondary": 24,
            "llm_rationale": 50, "judge_model": 18, "judge_prompt_version": 14, "run_timestamp": 18,
            "llm_broader": 22, "llm_meaning_changed": 14, "llm_label_valid": 10, "split": 8},
    notes={"llm_cat_primary": "Filled by the judge (Part 4).",
           "split": "dev = used to tune the judge; test = held out to measure it (Part 3)."})
LEGEND = [
    "How to use this tab",
    "Fill in the yellow columns only. Everything else comes from the pairs tab.",
    "reference_transcript is what the ASR should have produced. It's the manual transcript, except on empty-audio "
    "rows, where it's blank because the input was silence.",
    "manual_cat_primary: the main error type. manual_cat_secondary: a second type if the output mixes two; "
    "otherwise leave it blank.",
    "Spell each category the same way every time. The agreement tab compares labels as text, ignoring case and "
    "extra spaces.",
    "run_key shows which ASR setup produced the output. Try not to let it steer the label: judge only the "
    "reference vs the ASR output.",
    "Rows are in random order across runs.",
]


# ---------------------------------------------------------------- reading the source tab
def find_columns(values):
    width = max(len(r) for r in values)
    raw = pd.DataFrame([r + [""] * (width - len(r)) for r in values], dtype=object)
    group_row = raw.iloc[0].map(clean_text).replace("", np.nan).ffill().fillna("").tolist()
    name_row = raw.iloc[1].map(clean_text).tolist()
    vals = raw.iloc[2:].to_numpy()

    is_asr_group = lambda g: g.lower().startswith(("asr", "run"))
    base_cols = {}
    for j, (g, n) in enumerate(zip(group_row, name_row)):
        if is_asr_group(g):
            continue
        for field, rule in BASE_RULES.items():
            if field not in base_cols and rule(colkey(n)):
                base_cols[field] = j
    missing = [f for f in BASE_RULES if f not in base_cols]
    if "file_id" in missing or "manual_transcript" in missing:
        raise ValueError(f"Couldn't find required columns: {missing}. Check sheet.source_tab.")
    if missing:
        print("Warning: columns not found (left blank):", missing)

    # one ASR output per system: `pred_text` inside "ASR ..." groups, the single column of each "RUN: ..." group
    outputs = []
    for j, (g, n) in enumerate(zip(group_row, name_row)):
        gl = g.lower()
        if gl.startswith("asr") and colkey(n) == "predtext":
            rid = re.sub(r"^asr\s*", "", g, flags=re.I).strip()
            outputs.append({"col": j, "run_id": rid, "run_label": rid})
        elif gl.startswith("run"):
            rid = re.sub(r"^run\s*:?\s*", "", g, flags=re.I).strip()
            outputs.append({"col": j, "run_id": rid, "run_label": n})

    n_per_id = Counter(o["run_id"] for o in outputs)
    for o in outputs:
        short = re.sub(r"^gpt_4o_(transcribe_)?", "", slug(o["run_label"]))
        o["run_key"] = o["run_id"] if n_per_id[o["run_id"]] == 1 else f'{o["run_id"]}__{short}'
    dupes = {k for k, c in Counter(o["run_key"] for o in outputs).items() if c > 1}
    for o in outputs:
        if o["run_key"] in dupes:
            o["run_key"] += f'__col{col_letter(o["col"] + 1)}'
    return base_cols, outputs, vals


def infer_config(run_id, run_label):
    s, lab = f"{run_id} {run_label}".lower(), run_label.lower()
    m = re.search(r"(\d{8})_(\d{6})", run_id)
    date = datetime.datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M") if m else ""
    c = dict(model="unknown", finetuned="?", prompt_condition="unknown",
             sees_story="?", sees_question="?", is_probe="N", run_date=date)
    if "parakeet" in s:
        c.update(model="parakeet-tdt-0.6b-v2", finetuned=yn("finetuned" in s),
                 prompt_condition="none", sees_story="N", sees_question="N")
    elif "qwen3" in s:
        c.update(model="qwen3", finetuned=yn("finetuned" in s),
                 prompt_condition="none", sees_story="N", sees_question="N")
    elif "whisper" in s:
        c.update(model="whisper-large-v3", finetuned="N",
                 prompt_condition="none", sees_story="N", sees_question="N")
    else:
        c.update(model="gpt-4o-transcribe", finetuned="N")
        if "empty" in s:
            c.update(prompt_condition="empty_audio", is_probe="Y")
        elif "wrong" in s and "same" in s:
            c.update(prompt_condition="wrong_q_same", sees_story="Y", sees_question="wrong", is_probe="Y")
        elif "wrong" in s and ("different" in s or "cross" in s):
            c.update(prompt_condition="wrong_q_cross", sees_question="wrong", is_probe="Y")
        elif "story-only" in s or "story only" in s:
            c.update(prompt_condition="story_only", sees_story="Y", sees_question="N", is_probe="Y")
        elif "4.1 mini" in lab or "4.1-mini" in lab:
            c.update(model="gpt-4o-transcribe + gpt-4.1-mini", prompt_condition="context_plus_gpt41mini",
                     sees_story="Y", sees_question="Y")
        elif "no_prompt" in s or "no prompt" in s:
            c.update(prompt_condition="none", sees_story="N", sees_question="N")
        elif "short prompt v1" in lab:
            c.update(prompt_condition="short_v1")
        elif "short prompt v2" in lab:
            c.update(prompt_condition="short_v2")
        elif "low_story" in s or "low story" in s:
            c.update(prompt_condition="low_story_v3")
        elif "strict" in s:
            c.update(prompt_condition="strict_v2")
        elif "context" in s:
            c.update(prompt_condition="context", sees_story="Y", sees_question="Y")
    return c


# ---------------------------------------------------------------- building pairs
def build_pairs(ctx, base_cols, outputs, run_cfg, vals):
    p1, nz = ctx.cfg["part1"], ctx.normalizer
    suspect = {clean_text(x) for x in p1["suspect_file_ids"]}
    empty_conditions = set(p1["empty_audio_conditions"])
    records, seen_ids = [], Counter()
    for r in range(len(vals)):
        b = {f: clean_text(vals[r, j]) for f, j in base_cols.items()}
        b = {f: b.get(f, "") for f in BASE_RULES}
        if not b["file_id"]:
            continue
        manual = b["manual_transcript"]
        for o in outputs:
            cfg = run_cfg[o["run_key"]]
            asr = clean_text(vals[r, o["col"]])
            ref = "" if cfg["prompt_condition"] in empty_conditions else manual
            ref_clean, had_gap, had_teacher = nz.clean_reference(ref)
            ref_w, hyp_w = nz.words(ref_clean), nz.words(asr)
            ref_w = nz.merge_compounds(ref_w, hyp_w)
            hyp_w = nz.merge_compounds(hyp_w, ref_w)
            S, D, I = word_edit_ops(ref_w, hyp_w)
            key = "|".join([b["file_id"], b["story_title"], b["question_text"], b["repeat_no"], b["start_s"],
                            o["run_key"]])
            pid = "P" + hashlib.sha1(key.encode()).hexdigest()[:10]
            seen_ids[pid] += 1
            if seen_ids[pid] > 1:
                pid += f"-{seen_ids[pid]}"
            records.append({
                "pair_id": pid, "file_id": b["file_id"], "story_title": b["story_title"], "type": b["type"],
                "repeat_no": to_number(b["repeat_no"]), "question_text": b["question_text"],
                "start_s": to_number(b["start_s"]), "end_s": to_number(b["end_s"]),
                "run_key": o["run_key"], **{k: cfg[k] for k in SETUP_COLS},
                "manual_transcript": manual, "reference_text": ref, "asr_output": asr,
                "wer": round((S + D + I) / len(ref_w), 4) if ref_w else None,
                "n_ref_words": len(ref_w), "n_sub": S, "n_del": D, "n_ins": I,
                "exact_match_normalized": yn(ref_w == hyp_w),
                "manual_empty": yn(not nz.words(nz.clean_reference(manual)[0])),
                "asr_empty": yn(not nz.words(asr)),
                "has_error": yn(ref_w != hyp_w),
                **({"ref_has_gap": yn(had_gap), "ref_had_teacher_speech": yn(had_teacher)} if nz.extra else {}),
                "flag_suspect_audio": yn(b["file_id"] in suspect),
                "in_manual_sample": "N",
                "source_row": r + 3,
                "full_story": b["full_story"],
            })
    pairs = pd.DataFrame(records)
    n_dup = sum(c - 1 for c in seen_ids.values() if c > 1)
    if n_dup:
        print(f"Warning: {n_dup} pairs share identifiers with another row (duplicate source rows?); suffixed -2, -3...")
    return pairs


def choose_sample(ctx, pairs, order):
    """Reuse the rows of the manual tab if it exists; otherwise draw a stratified sample of error rows."""
    p1, manual_tab = ctx.cfg["part1"], ctx.cfg["sheet"]["manual_tab"]
    if manual_tab in [w.title for w in ctx.sh.worksheets()] and not p1["overwrite"]:
        ids = [x for x in (clean_text(v) for v in read_tab_df(ctx.sh, manual_tab).get("pair_id", [])) if x]
        sample = pairs[pairs.pair_id.isin(ids)].copy()
        pairs.loc[pairs.pair_id.isin(ids), "in_manual_sample"] = "Y"
        lost = set(ids) - set(pairs.pair_id)
        print(f"Using the existing sample in '{manual_tab}': {len(sample)} rows.")
        if lost:
            print(f"Warning: {len(lost)} labeled pair_ids no longer exist in the source tab (e.g. {sorted(lost)[:3]}).")
        n_clean = (sample.has_error == "N").sum()
        if n_clean:
            print(f"{n_clean} sample rows have no error with the current normalization. "
                  "They stay in the sample, and the judge still labels them.")
        return sample

    eligible = pairs[(pairs.has_error == "Y") & (pairs.flag_suspect_audio == "N")]
    counts = eligible.groupby("run_key").size().to_dict()
    rng = np.random.default_rng(p1["random_seed"])
    quota = {k: 0 for k in counts}
    remaining = min(p1["sample_size"], sum(counts.values()))
    while remaining > 0:
        open_runs = sorted(k for k in counts if quota[k] < counts[k])
        share = max(1, remaining // len(open_runs))
        for k in rng.permutation(open_runs):
            take = min(share, counts[k] - quota[k], remaining)
            quota[k] += take
            remaining -= take
            if remaining == 0:
                break
    parts = [eligible[eligible.run_key == k].sample(n=q, random_state=p1["random_seed"])
             for k, q in quota.items() if q]
    sample = pd.concat(parts).sample(frac=1, random_state=p1["random_seed"]).reset_index(drop=True)
    pairs.loc[pairs.pair_id.isin(sample.pair_id), "in_manual_sample"] = "Y"
    print(f"Drew a new sample: {len(sample)} rows from {len(parts)} runs (asked for {p1['sample_size']}).")
    show(ctx, "Sample rows per run", pd.Series(quota, name="rows").reindex([k for k in order if k in quota])
         .rename_axis("run_key").reset_index())
    return sample


# ---------------------------------------------------------------- writing
def full_build(ctx, pairs_out, sample, llm, existing):
    sh, p1 = ctx.sh, ctx.cfg["part1"]
    ann = sample.rename(columns={"reference_text": "reference_transcript"})[
        ["pair_id", "run_key", "story_title", "question_text", "reference_transcript", "asr_output", "wer"]].copy()
    ann_inputs = ["manual_cat_primary", "manual_cat_secondary", "manual_notes", "annotator"]
    for c in ann_inputs:
        ann[c] = ""
    ann["full_story"] = sample["full_story"].values

    agr = sample[["pair_id", "run_key", "model", "prompt_condition"]].sort_values(["run_key", "pair_id"]) \
        .reset_index(drop=True)
    for c in ["manual_cat_primary", "llm_cat_primary", "primary_match",
              "manual_cat_secondary", "llm_cat_secondary", "label_set_match"]:
        agr[c] = ""
    sizes = {"pairs": (len(pairs_out) + 1, len(pairs_out.columns)),
             "manual_annotation": (max(len(ann) + 1, len(LEGEND)), len(ann.columns) + 2),
             "llm_labels": (len(llm) + 1, len(llm.columns)),
             "agreement": (len(agr) + 1, 30)}

    clash = [t for t in NEW_TABS if t in existing]
    current = sum(w.row_count * w.col_count for t, w in existing.items() if t not in clash)
    new = sum(r * c for r, c in sizes.values())
    if current + new > CELL_LIMIT:
        raise RuntimeError(f"This would put {current + new:,} cells in the spreadsheet; Google Sheets allows "
                           f"{CELL_LIMIT:,}. Trim empty rows/columns in other tabs, or set "
                           "part1.include_full_story_in_pairs: false.")
    for t in clash:
        print(f"Deleting existing tab '{t}' (part1.overwrite = true)")
        with_retry(sh.del_worksheet, existing[t])
    tabs = {t: with_retry(sh.add_worksheet, title=t, rows=r, cols=c) for t, (r, c) in sizes.items()}

    print("Writing pairs ...")
    write_rows(tabs["pairs"], df_rows(pairs_out))
    print("Writing manual_annotation, llm_labels, agreement ...")
    write_rows(tabs["manual_annotation"], df_rows(ann))
    legend_col = len(ann.columns) + 2
    write_rows(tabs["manual_annotation"], [[t] for t in LEGEND], start_col=legend_col)
    write_rows(tabs["llm_labels"], df_rows(llm))
    write_rows(tabs["agreement"], df_rows(agr))

    # agreement formulas (live in Sheets)
    L = lambda df, c: col_letter(list(df.columns).index(c) + 1)
    look = lambda tab, df, c, r: (f'=IFERROR(INDEX({tab}!${L(df, c)}:${L(df, c)},'
                                  f'MATCH($A{r},{tab}!$A:$A,0))&"","")')
    same = lambda a, b, r: f"LOWER(TRIM({a}{r}))=LOWER(TRIM({b}{r}))"
    formulas = []
    for r in range(2, len(agr) + 2):
        formulas.append([
            look("manual_annotation", ann, "manual_cat_primary", r),
            look("llm_labels", llm, "llm_cat_primary", r),
            f'=IF(OR(E{r}="",F{r}=""),"",IF({same("E", "F", r)},"Y","N"))',
            look("manual_annotation", ann, "manual_cat_secondary", r),
            look("llm_labels", llm, "llm_cat_secondary", r),
            f'=IF(OR(E{r}="",F{r}=""),"",IF(OR(AND({same("E", "F", r)},{same("H", "I", r)}),'
            f'AND({same("E", "I", r)},{same("H", "F", r)})),"Y","N"))',
        ])
    write_rows(tabs["agreement"], formulas, start_row=2, start_col=5, option="USER_ENTERED")
    last = len(agr) + 1
    summary = [
        ["Summary (live formulas)", ""],
        ["Manual primary labels done", f'=COUNTIF(E2:E{last},"?*")'],
        ["LLM primary labels done", f'=COUNTIF(F2:F{last},"?*")'],
        ["Rows labeled by both", f'=COUNTIF(G2:G{last},"Y")+COUNTIF(G2:G{last},"N")'],
        ["Primary label agreement", f'=IF(M4=0,"",COUNTIF(G2:G{last},"Y")/M4)'],
        ["Label-set agreement", f'=IF(M4=0,"",COUNTIF(J2:J{last},"Y")/M4)'],
        ["", ""],
        ["Cohen's kappa, the confusion matrix, per-category precision/recall and agreement by run "
         "are added below by Part 5.", ""],
    ]
    write_rows(tabs["agreement"], summary, start_col=12, option="USER_ENTERED")

    print("Formatting ...")
    req = table_requests(tabs["pairs"], pairs_out, **PAIRS_FMT)
    ws_ann = tabs["manual_annotation"]
    req += table_requests(ws_ann, ann,
                          widths={"pair_id": 13, "run_key": 40, "story_title": 16, "question_text": 32,
                                  "reference_transcript": 40, "asr_output": 40, "wer": 8, "manual_cat_primary": 24,
                                  "manual_cat_secondary": 24, "manual_notes": 30, "annotator": 12, "full_story": 20},
                          wrap={"question_text", "reference_transcript", "asr_output", "manual_notes"},
                          inputs=ann_inputs)
    req += [col_width(ws_ann.id, legend_col - 1, 90),
            fmt(grid(ws_ann.id, 0, len(LEGEND), legend_col - 1, legend_col), wrapStrategy="WRAP",
                verticalAlignment="TOP"),
            fmt(grid(ws_ann.id, 0, 1, legend_col - 1, legend_col), textFormat={"bold": True})]
    req += table_requests(tabs["llm_labels"], llm, **LLM_FMT)
    ws_agr = tabs["agreement"]
    req += table_requests(ws_agr, agr,
                          widths={"pair_id": 13, "run_key": 40, "model": 22, "prompt_condition": 18,
                                  "manual_cat_primary": 22, "llm_cat_primary": 22, "primary_match": 10,
                                  "manual_cat_secondary": 22, "llm_cat_secondary": 22, "label_set_match": 11},
                          notes={"primary_match": "Y/N once both primary labels exist (case and extra spaces ignored).",
                                 "label_set_match": "Y when {primary, secondary} match as a set, in either order."})
    req += [col_width(ws_agr.id, 11, 34), col_width(ws_agr.id, 12, 14),
            fmt(grid(ws_agr.id, 0, 1, 11, 12), textFormat={"bold": True}),
            fmt(grid(ws_agr.id, 4, 6, 12, 13), numberFormat={"type": "PERCENT", "pattern": "0.0%"})]
    with_retry(sh.batch_update, {"requests": req})
    print(f"Done: {sh.url}\n  pairs: {len(pairs_out):,} rows | manual_annotation: {len(ann)} | "
          f"llm_labels: {len(llm):,} | agreement: {len(agr)}")


def refresh(ctx, pairs_out, llm, existing):
    """Rewrite pairs and llm_labels; keep every label already in llm_labels; leave the other tabs alone."""
    sh = ctx.sh
    ws = existing.get("pairs") or with_retry(sh.add_worksheet, title="pairs",
                                             rows=len(pairs_out) + 1, cols=len(pairs_out.columns))
    with_retry(ws.clear)
    with_retry(ws.resize, rows=len(pairs_out) + 1, cols=len(pairs_out.columns))
    write_rows(ws, df_rows(pairs_out))
    req = table_requests(ws, pairs_out, **PAIRS_FMT)

    if "llm_labels" in existing:
        old = read_tab_df(sh, "llm_labels")
        order = [c for c in old.columns if not c.startswith("_col")]
        order += [c for c in LLM_COLS if c not in order]
        kept = old.drop(columns=[c for c in ["run_key", "in_manual_sample"] if c in old.columns])
        kept = kept[[c for c in kept.columns if not c.startswith("_col")]]
        new = llm[["pair_id", "run_key", "in_manual_sample"]].merge(kept, on="pair_id", how="left")
        new = new.reindex(columns=order).fillna("")
        label_cols = [c for c in ["llm_cat_primary", "llm_broader"] if c in old.columns]
        if label_cols:
            gone = set(old[(old[label_cols] != "").any(axis=1)].pair_id) - set(new.pair_id)
            if gone:
                print(f"Warning: {len(gone)} labeled rows in llm_labels no longer exist in pairs and were dropped.")
        ws2 = existing["llm_labels"]
    else:
        new = llm
        ws2 = with_retry(sh.add_worksheet, title="llm_labels", rows=len(new) + 1, cols=len(new.columns))
    with_retry(ws2.clear)
    with_retry(ws2.resize, rows=len(new) + 1, cols=len(new.columns))
    write_rows(ws2, df_rows(new))
    req += table_requests(ws2, new, **LLM_FMT)
    with_retry(sh.batch_update, {"requests": req})
    print(f"Refreshed pairs ({len(pairs_out):,} rows) and llm_labels ({len(new):,} rows). "
          f"'{ctx.cfg['sheet']['manual_tab']}' and 'agreement' were not touched.")


# ---------------------------------------------------------------- entry point
def run(ctx):
    cfg, sh = ctx.cfg, ctx.sh
    p1, manual_tab = cfg["part1"], cfg["sheet"]["manual_tab"]

    values = with_retry(sh.worksheet(cfg["sheet"]["source_tab"]).get_all_values)
    base_cols, outputs, vals = find_columns(values)
    print(f"Base columns: { {f: col_letter(j + 1) for f, j in base_cols.items()} }")
    print(f"Found {len(outputs)} ASR outputs and {len(vals)} data rows.")

    run_cfg = {}
    for o in outputs:
        c = infer_config(o["run_id"], o["run_label"])
        c.update(p1["config_overrides"].get(o["run_key"], {}))
        run_cfg[o["run_key"]] = c
    unknown = set(p1["config_overrides"]) - set(run_cfg)
    if unknown:
        print("Warning: config_overrides keys that match no run:", unknown)
    cfg_table = pd.DataFrame([{"run_key": o["run_key"], "column": col_letter(o["col"] + 1),
                               "label": o["run_label"], **run_cfg[o["run_key"]]} for o in outputs])
    show(ctx, "Run setup (check it; fix '?' via part1.config_overrides)", cfg_table, csv="part1_run_setup.csv")
    todo = cfg_table[(cfg_table == "?").any(axis=1) | (cfg_table.model == "unknown")]
    if len(todo):
        print(f"{len(todo)} runs have '?' or unknown fields: " + ", ".join(todo.run_key))

    pairs = build_pairs(ctx, base_cols, outputs, run_cfg, vals)
    print(f"\n{len(pairs):,} pairs from {pairs.file_id.nunique()} files.")
    order = [o["run_key"] for o in outputs]
    diag = pairs.groupby("run_key").agg(
        pairs=("pair_id", "size"), blank_asr=("asr_empty", lambda s: (s == "Y").sum()),
        error_rows=("has_error", lambda s: (s == "Y").sum()), mean_wer=("wer", "mean")).reindex(order)
    diag["error_rate"] = (diag.error_rows / diag.pairs).round(3)
    diag["mean_wer"] = diag.mean_wer.round(3)
    show(ctx, "Per run", diag.reset_index(), csv="part1_per_run.csv")
    mostly_blank = diag[diag.blank_asr > 0.5 * diag.pairs].index.tolist()
    if mostly_blank:
        print("Check these runs: over half their outputs are blank, so they may not have been run on all rows: "
              + ", ".join(mostly_blank))

    sample = choose_sample(ctx, pairs, order)

    pairs_out = pairs if p1["include_full_story_in_pairs"] else pairs.drop(columns="full_story")
    llm = pairs[["pair_id", "run_key", "in_manual_sample"]].copy()
    for c in LLM_COLS[3:]:
        llm[c] = ""

    existing = {w.title: w for w in sh.worksheets()}
    present = [t for t in NEW_TABS if t in existing]
    if present and not p1["overwrite"]:
        if manual_tab not in existing:
            raise RuntimeError(f"Some tabs exist ({present}) but '{manual_tab}' doesn't. Set sheet.manual_tab to the "
                               "tab with your labels, or delete the partial tabs to build from scratch.")
        refresh(ctx, pairs_out, llm, existing)
    else:
        full_build(ctx, pairs_out, sample, llm, existing)
