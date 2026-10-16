"""Part 4: the LLM judge.

Gets the story, question, reference transcript and ASR output (not the run name) and answers in JSON with
broader category, subtype, secondary subtype, meaning_changed and a short reason. Labels outside the
definitions tab are sent back once; anything still invalid is saved with llm_label_valid = N.

The empty-transcript rules (taxonomy D1, agreed 2026-10-16) are also checked in code, not only in the prompt:
- reference empty, ASR not empty -> broader must be "Hallucination"
- reference not empty, ASR empty -> broader must be "No output"
- both not empty                 -> "Hallucination" is not allowed
- ASR not empty                  -> "No output" is not allowed
An answer that breaks one is sent back once with the reason, like an unknown label.

Providers: openai, azure, openai_compatible (a gateway with its own URL), vllm_server, anthropic.
"""
import datetime
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .report import section, show
from .sheets import read_tab_df, with_retry, write_columns
from .text import clean_text, colkey, yn

DEFAULT_KEY_ENV = {"openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY",
                   "openai_compatible": "OPENAI_API_KEY", "vllm_server": "VLLM_API_KEY",
                   "anthropic": "ANTHROPIC_API_KEY"}


# ---------------------------------------------------------------- client
class Judge:
    def __init__(self, jc):
        self.jc = jc
        p = jc["provider"]
        key_env = jc["api_key_env"] or DEFAULT_KEY_ENV[p]
        key = os.environ.get(key_env)
        if not key and p not in ("vllm_server",):
            raise RuntimeError(f"No API key: set {key_env} in your .env file (or change judge.api_key_env).")
        if p == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=key)
        elif p == "azure":
            from openai import AzureOpenAI
            if not jc["azure_endpoint"]:
                raise ValueError("Set judge.azure_endpoint for provider azure.")
            self.client = AzureOpenAI(api_key=key, azure_endpoint=jc["azure_endpoint"],
                                      api_version=jc["azure_api_version"])
        else:
            from openai import OpenAI
            if p in ("openai_compatible", "vllm_server") and not jc["base_url"]:
                raise ValueError(f"Set judge.base_url for provider {p}.")
            self.client = OpenAI(api_key=key or "EMPTY", base_url=jc["base_url"] or None)
        self.schema = None

    def _kwargs(self):
        jc, p = self.jc, self.jc["provider"]
        kw = {} if jc["temperature"] is None else {"temperature": jc["temperature"]}
        if p == "anthropic":
            return kw
        if p == "vllm_server":
            kw["max_tokens"] = jc["max_new_tokens"]
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": jc["qwen_thinking"]}}
            if jc["guided_json"] and not jc["qwen_thinking"]:
                kw["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "asr_error_label", "schema": self.schema}}
        elif jc["guided_json"]:   # OpenAI structured outputs: only the listed names are possible
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "asr_error_label", "strict": True, "schema": self.schema}}
        else:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def call(self, system, user):
        """One request with retries; returns the raw text."""
        kw = self._kwargs()
        for attempt in range(6):
            try:
                if self.jc["provider"] == "anthropic":
                    r = self.client.messages.create(model=self.jc["model"], max_tokens=self.jc["max_new_tokens"],
                                                    system=system, messages=[{"role": "user", "content": user}], **kw)
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                r = self.client.chat.completions.create(
                    model=self.jc["model"], messages=[{"role": "system", "content": system},
                                                      {"role": "user", "content": user}], **kw)
                return r.choices[0].message.content
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404) or attempt == 5:   # not worth retrying
                    raise
                wait = min(60, 2 ** attempt) + random.random()
                print(f"  {type(e).__name__}: {str(e)[:120]} ... retrying in {wait:.0f}s")
                time.sleep(wait)

    def call_many(self, system, users):
        with ThreadPoolExecutor(max_workers=self.jc["workers"]) as ex:
            return list(ex.map(lambda u: self.call(system, u), users))


# ---------------------------------------------------------------- prompt
def show_text(v):
    return v if clean_text(v) else "(empty)"


def taxonomy_text(taxonomy):
    lines = []
    for b, g in taxonomy.groupby("broader", sort=False):
        if len(g) == 1 and g.primary.iloc[0] == b:
            lines.append(f'- "{b}": {g.definition.iloc[0] or "(no definition)"}')
            continue
        lines.append(f'- Broader category "{b}", subtypes:')
        lines += [f'    - "{p}": {d or "(no definition)"}' for p, d in zip(g.primary, g.definition)]
    return "\n".join(lines)


def build_prompt(ctx, L):
    jc = ctx.cfg["judge"]
    man = L.man_labels
    dev_rows = man[(man.split == "dev") & man.labeled]
    if jc["few_shot_ids"]:
        examples = dev_rows[dev_rows.pair_id.isin(jc["few_shot_ids"])]
        bad = set(jc["few_shot_ids"]) - set(examples.pair_id)
        if bad:
            print(f"Ignored {len(bad)} few_shot_ids that aren't labeled dev rows: {sorted(bad)[:5]}")
    else:
        examples = dev_rows.sample(frac=1, random_state=ctx.cfg["part3"]["seed"]).drop_duplicates("primary")
    examples = examples.head(jc["max_examples"])
    example_text = "\n\n".join(
        f"Example {i}\nQuestion: {r.question_text}\nReference transcript: {show_text(r.reference_transcript)}\n"
        f"ASR output: {show_text(r.asr_output)}\nLabels: broader = {r.broader}; primary = {r.primary}; "
        f"secondary = {r.secondary or 'none'}" + (f"; meaning_changed = {r.meaning}" if r.meaning else "")
        + (f"\nNote: {r.notes}" if r.notes else "")
        for i, r in enumerate(examples.itertuples(), 1)) or "(none)"

    system = f"""You label errors made by automatic speech recognition (ASR) on recordings of children answering \
reading-comprehension questions about a short story.

For each item you get the story, the question, the reference transcript (what a human transcriber heard the child say) \
and the ASR output. Decide what kind of error the ASR made.

Categories:
{taxonomy_text(L.taxonomy)}

Rules:
{jc["labeling_rules"].strip()}

Examples labeled by a person:

{example_text}

Answer with a JSON object only:
{{"broader": "...", "primary": "...", "secondary": "...", "meaning_changed": "N" | "Y" | "Y_toward_correct", "rationale": "one short sentence"}}
"primary" and "secondary" must be subtype names from the list (for a category with no subtypes, its own name); \
"secondary" is "" when there's only one error. "broader" is the broader category of "primary"."""
    version = "p" + hashlib.sha1((jc["model"] + system).encode()).hexdigest()[:8]
    schema = {"type": "object",
              "properties": {"broader": {"type": "string", "enum": sorted(set(L.taxonomy.broader))},
                             "primary": {"type": "string", "enum": list(L.taxonomy.primary)},
                             "secondary": {"type": "string", "enum": [""] + list(L.taxonomy.primary)},
                             "meaning_changed": {"type": "string", "enum": ["N", "Y", "Y_toward_correct"]},
                             "rationale": {"type": "string"}},
              "required": ["broader", "primary", "secondary", "meaning_changed", "rationale"],
              "additionalProperties": False}
    return system, version, schema, len(examples)


def load_stories(ctx):
    v = with_retry(ctx.sh.worksheet(ctx.cfg["sheet"]["source_tab"]).get_all_values)
    names = [colkey(x) for x in v[1]]
    if "storytitle" not in names or "fullstory" not in names:
        print("Warning: no 'Story title' / 'Full Story' columns in the source tab; the judge won't see the stories.")
        return {}
    ti, si = names.index("storytitle"), names.index("fullstory")
    return {clean_text(r[ti]): clean_text(r[si]) for r in v[2:] if len(r) > si and clean_text(r[ti])}


# ---------------------------------------------------------------- answers
def parse(raw):
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    t = re.sub(r"^.*?</think>", "", t, flags=re.S) if "</think>" in t else t
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    return json.loads(m.group(0) if m else t)


def rule_problem(broader, ref, asr):
    """'' if the broader category fits the empty-transcript rules, else the reason it doesn't."""
    ref_empty, asr_empty = not clean_text(ref), not clean_text(asr)
    b = clean_text(broader).lower()
    if ref_empty and not asr_empty and b != "hallucination":
        return ('the reference is empty (the child said nothing) and the ASR output is not, '
                'so the broader category must be "Hallucination"')
    if not ref_empty and asr_empty and b != "no output":
        return 'the ASR output is empty but the reference is not, so the category must be "No output"'
    if not asr_empty and b == "no output":
        return 'the ASR output is not empty, so "No output" is not allowed'
    if not ref_empty and not asr_empty and b == "hallucination":
        return ('the child spoke (the reference is not empty), so "Hallucination" is not allowed; '
                "label what the ASR did to the child's words")
    return ""


def evaluate(L, jc, version, pair_id, raw, ref="", asr=""):
    """-> (record for llm_labels, is_valid, description of the problem)"""
    try:
        out = parse(raw)
        out = out if isinstance(out, dict) else {}
    except ValueError:
        out = {}
    labels, probs = L.resolve(
        pd.DataFrame([{"pair_id": pair_id, "b": out.get("broader", ""), "p": out.get("primary", ""),
                       "s": out.get("secondary", ""), "m": out.get("meaning_changed", "")}]),
        {"broader": "b", "primary": "p", "secondary": "s", "meaning": "m"})
    lab = labels.iloc[0]
    valid = bool(lab.primary) and not len(probs)
    if valid:
        bad = rule_problem(lab.broader, ref, asr)
        valid = not bad
    else:
        bad = ("labels that aren't in the list: "
               + (", ".join(f"{p['column']}={p['label']!r}" for _, p in probs.iterrows()) or "no valid JSON/primary"))
    rec = {"pair_id": pair_id,
           "llm_broader": lab.broader or clean_text(out.get("broader", "")),
           "llm_cat_primary": lab.primary or clean_text(out.get("primary", "")),
           "llm_cat_secondary": lab.secondary or clean_text(out.get("secondary", "")),
           "llm_meaning_changed": lab.meaning or clean_text(out.get("meaning_changed", "")),
           "llm_rationale": clean_text(out.get("rationale", "")) or ("" if out else clean_text(raw)[:300]),
           "llm_label_valid": yn(valid), "judge_model": jc["model"], "judge_prompt_version": version,
           "run_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    return rec, valid, bad


# ---------------------------------------------------------------- entry point
def run(ctx):
    jc = ctx.cfg["judge"]
    scope = jc["scope"]
    section(f"Part 4: LLM judge ({jc['provider']} / {jc['model']}, scope = {scope})")
    L = ctx.labels()
    system, version, schema, n_examples = build_prompt(ctx, L)
    prompt_file = ctx.out_dir / f"judge_prompt_{version}.txt"
    prompt_file.write_text(system)
    print(f"Prompt version {version} | {len(system):,} characters | {n_examples} examples from dev | "
          f"saved to {prompt_file}")

    pairs_tab = read_tab_df(ctx.sh, "pairs")
    llm_tab = read_tab_df(ctx.sh, "llm_labels")
    for c in ["split", "judge_prompt_version"]:
        if c not in llm_tab.columns:
            llm_tab[c] = ""
    split_of = llm_tab.set_index("pair_id")["split"]
    done_version = llm_tab.set_index("pair_id")["judge_prompt_version"]

    if scope in ("dev", "test"):
        ids = list(split_of[split_of == scope].index)
        if not ids:
            raise RuntimeError(f"No '{scope}' rows yet. Run Part 3 first.")
    else:
        ids = list(pairs_tab.pair_id[(pairs_tab.has_error == "Y") | (pairs_tab.in_manual_sample == "Y")])
    if scope == "test":
        earlier = set(done_version.reindex(ids).fillna("")) - {"", version}
        if earlier and not jc["allow_test_rerun"]:
            raise RuntimeError(f"The test set was already labeled with prompt version(s) {sorted(earlier)}. Re-running "
                               "it after more tuning makes the test number optimistic. Set judge.allow_test_rerun: "
                               "true to do it anyway.")

    todo = [p for p in ids if jc["force_relabel"] or done_version.get(p, "") != version]
    print(f"Scope '{scope}': {len(ids)} rows, {len(todo)} to label.")
    if not todo:
        print("Nothing to do: every row is already labeled with this prompt version (judge.force_relabel: true redoes them).")
        return

    judge = Judge(jc)
    judge.schema = schema
    stories = load_stories(ctx)
    items = pairs_tab.set_index("pair_id").loc[todo].reset_index()

    def user_message(r):
        return (f"Story: {stories.get(clean_text(r['story_title']), '(not found)')}\n\n"
                f"Question: {r['question_text']}\n"
                f"Reference transcript: {show_text(r['reference_text'])}\n"
                f"ASR output: {show_text(r['asr_output'])}")

    results, n_rule_fixed = [], 0
    for start in range(0, len(items), jc["batch_size"]):
        batch = items.iloc[start:start + jc["batch_size"]]
        users = [user_message(r) for _, r in batch.iterrows()]
        refs, asrs = list(batch.reference_text), list(batch.asr_output)
        evals = [evaluate(L, jc, version, pid, raw, rf, a)
                 for pid, raw, rf, a in zip(batch.pair_id, judge.call_many(system, users), refs, asrs)]
        redo = [i for i, (_, ok, _) in enumerate(evals) if not ok]
        if redo:  # one retry, telling the model what was wrong
            retry = [users[i] + f"\n\nYour previous answer was not accepted: {evals[i][2]}. Answer again with a "
                     "JSON object that follows the rules and uses only the listed category names." for i in redo]
            for i, raw in zip(redo, judge.call_many(system, retry)):
                evals[i] = evaluate(L, jc, version, batch.pair_id.iloc[i], raw, refs[i], asrs[i])
                n_rule_fixed += evals[i][1]
        recs = [e[0] for e in evals]
        write_columns(ctx.sh, "llm_labels", pd.DataFrame(recs).set_index("pair_id"))
        results += recs
        print(f"  {min(start + jc['batch_size'], len(items))}/{len(items)} labeled and saved")

    res = pd.DataFrame(results)
    res.to_csv(ctx.out_dir / f"judge_labels_{scope}_{version}.csv", index=False)
    print(f"Done. Fixed on retry: {n_rule_fixed} | still invalid (labels or empty-transcript rules): "
          f"{(res.llm_label_valid == 'N').sum()}")
    show(ctx, "Judge labels (subtype)", res.llm_cat_primary.value_counts().rename("rows")
         .rename_axis("subtype").reset_index())
