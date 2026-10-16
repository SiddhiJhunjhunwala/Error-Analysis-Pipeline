"""Load config.yaml, fill in defaults, and catch typos."""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "run": {"parts": [2], "output_dir": "outputs"},
    "sheet": {"url": "", "source_tab": "", "manual_tab": "manual_annotation", "taxonomy_tab": "error definitions"},
    "google_auth": {"method": "oauth", "oauth_client_file": "credentials.json",
                    "token_file": "authorized_user.json", "service_account_file": "service_account.json"},
    "part1": {"overwrite": False, "sample_size": 250, "random_seed": 42, "include_full_story_in_pairs": False,
              "suspect_file_ids": [], "empty_audio_conditions": ["empty_audio"], "config_overrides": {},
              "extra_normalization": False, "drop_fillers": True,
              "fillers": ["um", "umm", "uhm", "uhmm", "uh", "uhh", "er", "erm", "ah", "hm", "hmm", "hmmm",
                          "mm", "mmm", "mhm", "uhum", "uhuh", "huh", "mmhmm"],
              "spelling_map": {"cause": "because", "cuz": "because", "coz": "because", "cos": "because",
                               "ok": "okay"},
              "merge_compounds": True},
    "part2": {"manual_cols": {"broader": "Broader category", "primary": "manual_cat_primary",
                              "secondary": "manual_cat_secondary", "meaning": "meaning_changed",
                              "notes": "manual_notes"},
              "label_fixes": {}, "skip_taxonomy_rows": ["proposed additions", "how the drafts were labeled"]},
    "part3": {"dev_size": 50, "seed": 7, "resplit": False},
    "judge": {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "", "base_url": "",
              "azure_endpoint": "", "azure_api_version": "2024-10-21", "temperature": 0, "max_new_tokens": 400,
              "guided_json": True, "qwen_thinking": False, "scope": "dev", "workers": 8, "batch_size": 100,
              "force_relabel": False, "allow_test_rerun": False, "max_examples": 10, "few_shot_ids": [],
              "labeling_rules": ""},
    "part5": {"target_kappa": 0.70, "show_test_disagreements": False},
    "part6": {"results_tab": "results"},
}
# sections whose keys are free-form (not checked for typos)
FREE_FORM = {("part1", "config_overrides"), ("part1", "spelling_map"), ("part2", "label_fixes"),
             ("part2", "manual_cols")}
PROVIDERS = {"openai", "azure", "openai_compatible", "vllm_server", "anthropic"}
SCOPES = {"dev", "test", "all"}


def _merge(base, over, path, unknown):
    for k, v in (over or {}).items():
        if k not in base:
            unknown.append(".".join(path + [k]))
            continue
        if isinstance(base[k], dict) and isinstance(v, dict) and tuple(path + [k]) not in FREE_FORM:
            _merge(base[k], v, path + [k], unknown)
        elif isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v} if k == "manual_cols" else v
        else:
            base[k] = v


def load_config(path, parts=None, scope=None):
    path = Path(path).resolve()
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg, unknown = copy.deepcopy(DEFAULTS), []
    _merge(cfg, user, [], unknown)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {path.name}: {', '.join(unknown)} (typo?)")
    if parts is not None:
        cfg["run"]["parts"] = parts
    if scope is not None:
        cfg["judge"]["scope"] = scope

    cfg["run"]["parts"] = [int(p) for p in cfg["run"]["parts"]]
    bad = [p for p in cfg["run"]["parts"] if p not in range(1, 7)]
    if bad:
        raise ValueError(f"run.parts must be numbers 1-6; got {bad}")
    if not cfg["sheet"]["url"] or not cfg["sheet"]["source_tab"]:
        raise ValueError("Set sheet.url and sheet.source_tab in the config.")
    if cfg["judge"]["provider"] not in PROVIDERS:
        raise ValueError(f"judge.provider must be one of {sorted(PROVIDERS)}")
    if cfg["judge"]["scope"] not in SCOPES:
        raise ValueError(f"judge.scope must be one of {sorted(SCOPES)}")
    if cfg["google_auth"]["method"] not in ("oauth", "service_account"):
        raise ValueError("google_auth.method must be 'oauth' or 'service_account'")

    cfg["_base_dir"] = path.parent
    cfg["_path"] = path
    return cfg


def resolve_path(cfg, p):
    """Paths in the config are relative to the config file's folder."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else cfg["_base_dir"] / p
