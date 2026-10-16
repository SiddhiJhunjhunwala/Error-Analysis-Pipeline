# ASR error analysis

Reads the ASR results from the Google Sheet, prepares the error-analysis tabs, runs an LLM judge, checks it against your manual labels, and builds the results tables. Everything is written back into the same spreadsheet; CSV copies of every printed table go to `outputs/`.

You choose what to run, and with which model, in `config.yaml`. The default judge is OpenAI `gpt-4o-mini`.

| Part | What it does | Writes |
|---|---|---|
| 1 | One row per answer × ASR output with WER and `has_error`. Safe to re-run: only `pairs` and `llm_labels` are refreshed; your labels are never touched | `pairs`, `llm_labels` (+ `manual_annotation`, `agreement` on the very first run) |
| 2 | Checks your manual labels against the `error definitions` tab | nothing |
| 3 | Splits the labeled rows into dev (tune the judge) and test (measure it once) | `llm_labels` → `split` |
| 4 | LLM judge on dev, test or all error rows | `llm_labels` |
| 5 | Judge vs manual agreement | `agreement` |
| 6 | Results tables | `results` |

## Quick start (no setup needed)

Double-click **`run.command`** in Finder. It does the whole setup on the first run, then asks what to run:

1. finds Python (or installs it with Homebrew if you have it),
2. creates a private environment in `.venv` and installs the packages,
3. asks you to paste your API key once (saved in `.env`),
4. walks you through the one-time Google sign-in file and picks it up from your Downloads folder,
5. shows a menu with the steps below.

Later runs skip straight to the menu. From Terminal you can also pass options directly: `./run.command --parts 4,5 --scope dev`.

If macOS says it can't open `run.command` because it's from an unidentified developer, right-click it, choose **Open**, then **Open** again (only needed the first time). If double-clicking opens a text editor instead, run it from Terminal: `cd` into the folder and type `./run.command`.

The sections below describe the same setup done by hand.

## Manual setup on a Mac


**1. Python and packages.** Python 3.10 or newer (`python3 --version`; install from python.org or `brew install python` if needed).

```bash
cd asr-error-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in every new terminal before using the project.

**2. API key.** Copy `.env.example` to `.env` and add your key:

```
OPENAI_API_KEY=sk-...
```

**3. Google sign-in.** The project signs in as you, so it can edit any sheet you can edit. You need an OAuth client file once:

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and create a project (any name).
2. *APIs & Services → Library*: enable **Google Sheets API** and **Google Drive API**.
3. *OAuth consent screen* (also called *Google Auth Platform*): choose **External**, fill in the app name and your email, and add your own Google address under **Test users**.
4. *Credentials → Create credentials → OAuth client ID*, application type **Desktop app**. Download the JSON and save it in this folder as `credentials.json`.

The first run opens a browser window to sign in; after that the login is saved in `authorized_user.json`. If your university account blocks this, use a personal Google account that has edit access to the sheet.

Keep `.env`, `credentials.json` and `authorized_user.json` private (they're in `.gitignore`).

## Running

```bash
python run.py                         # runs the parts listed under run.parts in config.yaml
python run.py --parts 1,2             # or choose the parts on the command line
python run.py --parts 4 --scope test  # and the judge scope
python run.py -c another_config.yaml
```

Parts always run in number order. Everything else (sheet, tabs, model, rules, sample size...) is set in `config.yaml`; each setting is explained there.

### Order of work

1. **Prepare**: `python run.py --parts 1,2`. Fix any label problems Part 2 lists (in the sheet, or with `part2.label_fixes`), and repeat `--parts 2` until it's clean.
2. **Split once**: `python run.py --parts 3`.
3. **Tune on dev**: `python run.py --parts 4,5 --scope dev`. Read `outputs/part5_dev_disagreements.csv`, then adjust `judge.labeling_rules` in the config or the definitions tab, and run again. Each change gives a new prompt version, so the dev rows are labeled again.
4. **Measure once on test**: `python run.py --parts 4,5 --scope test`. This is the agreement number to report. Re-running test with a changed prompt or model is blocked unless `judge.allow_test_rerun: true`.
5. **Label everything**: `python run.py --parts 4,6 --scope all`.

Re-running Part 4 only labels rows that don't have a label from the current prompt version yet, so an interrupted run picks up where it stopped.

### What ends up in `outputs/`

CSV copies of every table (run setup, label problems, split, agreement, disagreements, results), each judge run's labels, and the exact prompt used for each version (`judge_prompt_<version>.txt`), which is handy for the paper.

## Choosing the judge model

Set `judge.provider` and `judge.model` in `config.yaml`:

| Provider | Use it for | Also set | Key in `.env` |
|---|---|---|---|
| `openai` (default) | OpenAI's API, e.g. `gpt-4o-mini` | | `OPENAI_API_KEY` |
| `azure` | an Azure OpenAI key (common for university keys) | `azure_endpoint`, `azure_api_version`; `model` = deployment name | `AZURE_OPENAI_API_KEY` |
| `openai_compatible` | a university gateway or proxy with its own URL | `base_url` | `OPENAI_API_KEY` (or set `api_key_env`) |
| `vllm_server` | Qwen etc. served with `vllm serve` on a GPU machine | `base_url`, e.g. `http://gpu-box:8000/v1` | `VLLM_API_KEY` if the server needs one |
| `anthropic` | Anthropic's API | `pip install anthropic` | `ANTHROPIC_API_KEY` |

`judge.guided_json: true` restricts the answer to valid JSON with only your category names (OpenAI structured outputs, or guided decoding on vLLM). If a gateway rejects it, set it to `false`; invalid labels are then sent back once for a retry.

vLLM itself doesn't run on a Mac, so for Qwen models run `vllm serve` on a GPU machine (or keep using the Colab notebook) and point `base_url` at it.

## Troubleshooting

- **`Config error: Unknown setting(s)`**: a typo in `config.yaml`; the message names the setting.
- **Browser says "access blocked" / "app not verified"**: add your Google address as a test user on the OAuth consent screen (setup step 3).
- **Sign-in stopped working**: delete `authorized_user.json` and run again.
- **`401` / `invalid_api_key`**: the key doesn't match the provider. University keys are often Azure keys (use `provider: azure`) or gateway keys (use `openai_compatible` with the gateway URL).
- **Connection errors or timeouts with a university key**: it may only work on the campus network or VPN.
- **Rate-limit errors (429)**: they're retried automatically; if they keep coming, lower `judge.workers`.
