"""Part 2: categories from the definitions tab, and the manual labels checked against them.

Other parts use the Labels object this builds (via ctx.labels()).
"""
import difflib
import re

import pandas as pd

from .report import section, show
from .sheets import read_tab_df
from .text import clean_text, norm_label


class Labels:
    """Category names and how to map any label text onto them."""

    def __init__(self, taxonomy, label_fixes):
        self.taxonomy = taxonomy
        self.canon_primary = {norm_label(p): p for p in taxonomy.primary}
        self.canon_broader = {norm_label(b): b for b in taxonomy.broader}
        self.parent = dict(zip(taxonomy.primary, taxonomy.broader))
        self.fixes = {norm_label(k): v for k, v in label_fixes.items()}
        self.man_labels = None
        self.has_manual_meaning = False

    def canon(self, v, table):
        """Canonical label, '' for blank, None if unknown."""
        n = norm_label(v)
        if not n:
            return ""
        return table.get(norm_label(self.fixes.get(n, n)))

    @staticmethod
    def canon_meaning(v):
        n = norm_label(v)
        if not n:
            return ""
        if "toward" in n:
            return "Y_toward_correct"
        return {"y": "Y", "yes": "Y", "n": "N", "no": "N"}.get(n)

    def resolve(self, df, cols):
        """Canonical broader/primary/secondary/meaning per row, plus a table of problems.
        `cols` maps broader/primary/secondary/meaning to column names in df (or None)."""
        out, problems = [], []
        for _, r in df.iterrows():
            raw = {k: clean_text(r[c]) if c else "" for k, c in cols.items()}
            b = self.canon(raw["broader"], self.canon_broader)
            p = self.canon(raw["primary"], self.canon_primary)
            s = self.canon(raw["secondary"], self.canon_primary)
            if raw["primary"] == "" and raw["broader"]:      # broader-only row: use it as the subtype
                p = self.canon(raw["broader"], self.canon_primary) or (b or None)
            if b == "" and p:                                  # subtype only: take its broader category
                b = self.parent.get(p, "")
            m = self.canon_meaning(raw.get("meaning", ""))
            for kind, val, got in [("broader", raw["broader"], b), ("primary", raw["primary"], p),
                                   ("secondary", raw["secondary"], s), ("meaning", raw.get("meaning", ""), m)]:
                if val and got is None:
                    problems.append({"pair_id": r["pair_id"], "column": kind, "label": val, "issue": "unknown label"})
            if b and p and self.parent.get(p) and self.parent[p] != b:
                problems.append({"pair_id": r["pair_id"], "column": "broader/primary",
                                 "label": f"{raw['broader']} / {raw['primary']}",
                                 "issue": f"subtype belongs to '{self.parent[p]}'"})
            out.append({"pair_id": r["pair_id"], "broader": b or "", "primary": p or "",
                        "secondary": s or "", "meaning": m or ""})
        return (pd.DataFrame(out, columns=["pair_id", "broader", "primary", "secondary", "meaning"]),
                pd.DataFrame(problems, columns=["pair_id", "column", "label", "issue"]))


def _find(df, *words):
    return next((c for c in df.columns
                 if all(w in c.lower() for w in words) and not c.lower().startswith("draft")), None)


def read_taxonomy(ctx):
    p2 = ctx.cfg["part2"]
    tx = read_tab_df(ctx.sh, ctx.cfg["sheet"]["taxonomy_tab"])
    c_type, c_sub, c_def = _find(tx, "error", "type"), _find(tx, "subtype"), _find(tx, "definition")
    if not (c_type and c_sub and c_def):
        raise ValueError(f"'{ctx.cfg['sheet']['taxonomy_tab']}' needs columns like 'Error Type', 'SubType', "
                         f"'Definition'; found {list(tx.columns)}")
    skip = tuple(s.lower() for s in p2["skip_taxonomy_rows"])
    rows, current = [], None
    for _, r in tx.iterrows():
        e, s, d = clean_text(r[c_type]), clean_text(r[c_sub]), clean_text(r[c_def])
        if e and e.lower().startswith(skip):
            current = None
            continue
        if e:
            current = e
        if not (e or s) or current is None:
            continue
        rows.append({"broader": current, "primary": s or current,
                     "definition": re.sub(r"^\[proposed\]\s*", "", d, flags=re.I)})
    if not rows:
        raise ValueError("No categories found in the definitions tab.")
    return pd.DataFrame(rows).drop_duplicates("primary").reset_index(drop=True)


def load_labels(ctx, verbose=False):
    cfg, p2 = ctx.cfg, ctx.cfg["part2"]
    manual_tab = cfg["sheet"]["manual_tab"]
    L = Labels(read_taxonomy(ctx), p2["label_fixes"])

    man_raw = read_tab_df(ctx.sh, manual_tab)
    if "pair_id" not in man_raw.columns:
        raise ValueError(f"'{manual_tab}' has no pair_id column. Check sheet.manual_tab.")
    man_raw = man_raw[man_raw.pair_id.map(clean_text) != ""].reset_index(drop=True)
    cols = {}
    for k, name in p2["manual_cols"].items():
        cols[k] = name if name in man_raw.columns else (_find(man_raw, k) if k in ("broader", "meaning") else None)
        if k in ("broader", "primary") and not cols[k] and verbose:
            print(f"Warning: no '{name}' column in '{manual_tab}'")
    man, problems = L.resolve(man_raw, {k: cols.get(k) for k in ["broader", "primary", "secondary", "meaning"]})
    man["notes"] = man_raw[cols["notes"]].map(clean_text).values if cols.get("notes") else ""
    for c in ["run_key", "story_title", "question_text", "reference_transcript", "asr_output"]:
        man[c] = man_raw[c].values if c in man_raw.columns else ""
    man["labeled"] = man.primary != ""

    # dev/test split saved by Part 3
    man["split"] = ""
    if "llm_labels" in [w.title for w in ctx.sh.worksheets()]:
        llm_tab = read_tab_df(ctx.sh, "llm_labels")
        if "split" in llm_tab.columns:
            man["split"] = man.pair_id.map(llm_tab.set_index("pair_id")["split"]).fillna("")
    L.man_labels = man
    L.has_manual_meaning = bool(cols.get("meaning")) and (man.meaning != "").any()

    if verbose:
        section("Part 2: categories and label check")
        show(ctx, "Categories", L.taxonomy, csv="part2_categories.csv")
        no_def = L.taxonomy[L.taxonomy.definition == ""]
        if len(no_def):
            print(f"\nNo definition yet for: {', '.join(no_def.primary)}. The judge only sees the name for these.")
        print("\nLabel columns used:", {k: v for k, v in cols.items() if v})
        print(f"{len(man)} rows | usable labels: {man.labeled.sum()} | blank or unknown subtype: "
              f"{(~man.labeled).sum()} (left out of the split and agreement until fixed)")
        all_names = sorted(set(L.canon_primary.values()) | set(L.canon_broader.values()))
        if len(problems):
            summary = problems.groupby(["column", "label", "issue"]).size().rename("rows").reset_index()
            summary["closest category"] = [", ".join(difflib.get_close_matches(l, all_names, n=2, cutoff=0.5))
                                           if i == "unknown label" else ""
                                           for l, i in zip(summary.label, summary.issue)]
            show(ctx, f"{len(problems)} label problems (fix in the sheet or via part2.label_fixes)", summary,
                 csv="part2_label_problems.csv")
            problems.to_csv(ctx.out_dir / "part2_label_problems_by_row.csv", index=False)
        else:
            print("No label problems.")
        counts = man[man.labeled].groupby(["broader", "primary"]).size().rename("rows").reset_index()
        show(ctx, "Labeled rows per category", counts, csv="part2_label_counts.csv")
        rare = counts[counts.rows < 3]
        if len(rare):
            print("Fewer than 3 labeled rows (agreement on these will be noisy; consider merging them): "
                  + ", ".join(rare.primary))
    else:
        n_prob = len(problems)
        print(f"Loaded {len(L.taxonomy)} categories and {man.labeled.sum()} usable manual labels"
              + (f" ({n_prob} label problems: run Part 2 to see them)." if n_prob else "."))
    return L


def run(ctx):
    ctx.labels(verbose=True)
