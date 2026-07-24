#!/usr/bin/env bash
# Run the resume -> jobs match on Claude Code session start.
#
# Scrapes fresh jobs (Indeed via jobspy), then ranks Fei's resume against them
# with the free local embedding model (no Ollama needed). Prints the top matches
# to stdout so a SessionStart hook can surface them, and also caches them to
# data/today_matches.txt.
#
# Wired up via .claude/settings.json (SessionStart hook). Safe to run by hand:
#     bash swift_mytodo/run_match_on_start.sh
set -uo pipefail

REPO="/Users/feijing/github.com/swift_demo"
APP="$REPO/swift_mytodo"
PY="$REPO/swift_demo/bin/python"          # the project venv named after the repo
RESUME="$REPO/Fei_Jing_Resume_2026.pdf"
OUT="$APP/data/today_matches.txt"

# Use the allowlisted classic HuggingFace CDN (avoids the blocked Xet endpoints).
export HF_HUB_DISABLE_XET=1
# Google's jobspy path returns nothing in this environment; Indeed is allowlisted.
export JOBSPY_SITES="${JOBSPY_SITES:-indeed}"
export TOKENIZERS_PARALLELISM=false

# Bail quietly if the project isn't set up (e.g. venv/resume missing) so a broken
# environment never blocks the session from starting.
[ -x "$PY" ] || { echo "[match] venv not found at $PY — skipping."; exit 0; }
[ -f "$RESUME" ] || { echo "[match] resume not found at $RESUME — skipping."; exit 0; }
cd "$APP" || exit 0

# 1. Scrape fresh jobs (best-effort: on failure we fall back to the last scrape).
if ! "$PY" job_scraper.py >/dev/null 2>&1; then
    echo "[match] scrape failed (offline / rate-limited) — using last cached jobs."
fi

# 2. Rank the resume against the jobs by embedding similarity.
RAW="$("$PY" match_one.py "$RESUME" --no-llm --top-n 5 2>/dev/null)"

# 3. Emit just the ranked matches (and cache them).
mkdir -p "$(dirname "$OUT")"
echo "Top job matches for $(basename "$RESUME") (fresh scrape, $(date '+%Y-%m-%d %H:%M')):"
if echo "$RAW" | grep -q "TOP MATCHES"; then
    echo "$RAW" | sed -n '/TOP MATCHES/,$p' | grep -vE "=== |^$" | tee "$OUT"
else
    echo "No matches available (no jobs scraped yet). Run: $PY job_scraper.py" | tee "$OUT"
fi
