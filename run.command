#!/bin/bash
# Sets everything up on the first run, then runs the analysis.
#
#   Double-click this file in Finder        -> asks which step to run
#   ./run.command                           -> same, from Terminal
#   ./run.command --parts 4,5 --scope dev   -> passes the options straight to run.py
#
# Safe to run as often as you like: finished setup steps are skipped.

cd "$(dirname "$0")" || exit 1
INTERACTIVE=0
[ $# -eq 0 ] && INTERACTIVE=1

pause_and_exit() {   # keep the Terminal window open when double-clicked
    if [ $INTERACTIVE -eq 1 ]; then
        echo
        read -r -p "Press Enter to close this window." _
    fi
    exit "$1"
}
say() { printf "\n\033[1m%s\033[0m\n" "$1"; }
fail() { printf "\n\033[31m%s\033[0m\n" "$1"; pause_and_exit 1; }

# ---------------------------------------------------------------- 1. Python
say "Checking Python ..."
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    if command -v brew >/dev/null 2>&1; then
        say "Installing Python with Homebrew ..."
        brew install python || fail "Homebrew couldn't install Python."
        PY=python3
    else
        fail "Python 3.9 or newer is needed. Install it from https://www.python.org/downloads/ (the macOS installer), then run this again.
(If a window asked to install the 'command line developer tools', accept it, wait for it to finish, and run this again.)"
    fi
fi
echo "Using $($PY --version)"

# ---------------------------------------------------------------- 2. virtual environment + packages
if [ ! -x .venv/bin/python ]; then
    say "Creating the virtual environment (.venv) ..."
    "$PY" -m venv .venv || fail "Couldn't create the virtual environment."
fi
VPY=.venv/bin/python
REQ_SUM=$(cksum requirements.txt | awk '{print $1}')
if [ ! -f .venv/.requirements_done ] || [ "$(cat .venv/.requirements_done)" != "$REQ_SUM" ]; then
    say "Installing packages (first run only; takes a minute) ..."
    "$VPY" -m pip install --quiet --upgrade pip || fail "Couldn't update pip."
    "$VPY" -m pip install --quiet -r requirements.txt || fail "Couldn't install the packages in requirements.txt."
    PROVIDER=$("$VPY" -c 'import yaml; print((yaml.safe_load(open("config.yaml")) or {}).get("judge", {}).get("provider", "openai"))' 2>/dev/null)
    if [ "$PROVIDER" = "anthropic" ]; then
        "$VPY" -m pip install --quiet anthropic || fail "Couldn't install the anthropic package."
    fi
    echo "$REQ_SUM" > .venv/.requirements_done
fi

# ---------------------------------------------------------------- 3. API key
KEY_ENV=$("$VPY" - <<'EOF' 2>/dev/null
import yaml
from asr_ea.judge import DEFAULT_KEY_ENV
j = (yaml.safe_load(open("config.yaml")) or {}).get("judge", {})
p = j.get("provider", "openai")
print(j.get("api_key_env") or DEFAULT_KEY_ENV.get(p, "OPENAI_API_KEY"))
EOF
)
KEY_ENV=${KEY_ENV:-OPENAI_API_KEY}
touch .env
if ! grep -Eq "^${KEY_ENV}=.+" .env && [ -z "$(printenv "$KEY_ENV")" ]; then
    if [ "$KEY_ENV" = "VLLM_API_KEY" ]; then
        :   # vLLM servers usually don't need a key
    else
        say "API key needed: $KEY_ENV"
        echo "Paste the key and press Enter (it won't be shown). It's saved in .env in this folder."
        read -r -s -p "> " KEY
        echo
        [ -z "$KEY" ] && fail "No key entered."
        grep -v "^${KEY_ENV}=" .env > .env.tmp; mv .env.tmp .env
        echo "${KEY_ENV}=${KEY}" >> .env
        chmod 600 .env
        echo "Saved."
    fi
fi

# ---------------------------------------------------------------- 4. Google sign-in file
CLIENT_FILE=$("$VPY" -c 'import yaml; print((yaml.safe_load(open("config.yaml")) or {}).get("google_auth", {}).get("oauth_client_file", "credentials.json"))' 2>/dev/null)
CLIENT_FILE=${CLIENT_FILE:-credentials.json}
METHOD=$("$VPY" -c 'import yaml; print((yaml.safe_load(open("config.yaml")) or {}).get("google_auth", {}).get("method", "oauth"))' 2>/dev/null)
find_download() { ls -t "$HOME"/Downloads/client_secret_*.json 2>/dev/null | head -1; }
if [ "$METHOD" = "oauth" ] && [ ! -f "$CLIENT_FILE" ]; then
    FOUND=$(find_download)
    if [ -z "$FOUND" ]; then
        say "One-time Google setup (about 5 minutes)"
        cat <<'EOF'
The script needs permission to edit the Google Sheet as you. Google requires a small
"OAuth client" file for that, which only you can create:

  1. In the page that opens, create a project (any name) if you don't have one.
  2. APIs & Services > Library: enable "Google Sheets API" and "Google Drive API".
  3. OAuth consent screen: choose External, fill in an app name and your email,
     and add your own Google address under "Test users".
  4. Credentials > Create credentials > OAuth client ID > Application type "Desktop app"
     > Create > Download JSON.

Leave the downloaded file in your Downloads folder; this script picks it up.
EOF
        open "https://console.cloud.google.com/apis/credentials" 2>/dev/null
        while [ -z "$FOUND" ]; do
            read -r -p "Press Enter once the JSON is downloaded (or type q to quit): " ANSWER
            [ "$ANSWER" = "q" ] && pause_and_exit 1
            FOUND=$(find_download)
            [ -z "$FOUND" ] && echo "No client_secret_*.json in ~/Downloads yet."
        done
    fi
    cp "$FOUND" "$CLIENT_FILE" && chmod 600 "$CLIENT_FILE"
    echo "Using $FOUND as $CLIENT_FILE."
    echo "The first run opens your browser to sign in to Google. If it says the app isn't verified,"
    echo "click 'Continue' (it's your own app). Sign in with an account that can edit the sheet."
fi

# ---------------------------------------------------------------- 5. run
if [ $INTERACTIVE -eq 1 ]; then
    say "What do you want to run?"
    cat <<'EOF'
  1) Prepare: build/refresh the tabs and check the manual labels   (parts 1,2)
  2) Check the manual labels again                                  (part 2)
  3) Make the dev/test split (once)                                 (part 3)
  4) Tune the judge on dev and show agreement                       (parts 4,5 dev)
  5) Measure the judge on test (once, after tuning)                 (parts 4,5 test)
  6) Label all error rows and build the results tables              (parts 4,6 all)
  7) Whatever run.parts says in config.yaml
EOF
    read -r -p "Choose 1-7: " CHOICE
    case "$CHOICE" in
        1) ARGS=(--parts 1,2) ;;
        2) ARGS=(--parts 2) ;;
        3) ARGS=(--parts 3) ;;
        4) ARGS=(--parts 4,5 --scope dev) ;;
        5) ARGS=(--parts 4,5 --scope test) ;;
        6) ARGS=(--parts 4,6 --scope all) ;;
        7) ARGS=() ;;
        *) fail "Please choose a number from 1 to 7." ;;
    esac
else
    ARGS=("$@")
fi

say "Running: python run.py ${ARGS[*]}"
"$VPY" run.py "${ARGS[@]}"
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo
    echo "Finished. Tables are in the Google Sheet; CSV copies are in the outputs folder."
else
    echo
    echo "Something went wrong (see the message above). README.md has a troubleshooting section."
fi
pause_and_exit $STATUS
