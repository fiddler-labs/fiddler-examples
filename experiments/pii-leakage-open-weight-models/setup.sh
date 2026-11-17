#!/usr/bin/env bash
# Build the patched tau2-bench this study ran on, in ./tau2-bench.
#
#   1. clone sierra-research/tau2-bench at the pinned commit
#   2. apply patches/tau2-bench.patch (scripted user, leak task set, small fixes)
#   3. copy in the leak tasks and the study scripts
#   4. install dependencies with uv
#
# Safe to re-run: it rebuilds ./tau2-bench from scratch.
set -euo pipefail

UPSTREAM=https://github.com/sierra-research/tau2-bench.git
COMMIT=96a3ef92329baf908ffe9dfe188cea34e8701d2a
ROOT="$(cd "$(dirname "$0")" && pwd)"
DIR="$ROOT/tau2-bench"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

rm -rf "$DIR"
git clone --quiet --filter=blob:none --no-checkout "$UPSTREAM" "$DIR"
git -C "$DIR" checkout --quiet "$COMMIT"
git -C "$DIR" apply "$ROOT/patches/tau2-bench.patch"

cp -R "$ROOT/study/tasks/leak" "$DIR/data/tau2/domains/banking_knowledge/leak"
cp "$ROOT"/study/scripts/*.py "$DIR/scripts/"

cd "$DIR"
uv sync --quiet --extra knowledge --extra dev
# tau2 imports its voice package at start-up, which needs websockets. The rest of
# the voice extra (pyaudio needs PortAudio) is not used here.
uv pip install --quiet "websockets>=13.0"

echo "Ready: $DIR"
echo "Check the leak suite with:  (cd tau2-bench && uv run --no-sync pytest -q tests/test_scripted_user.py tests/test_domains/test_banking_knowledge/test_leak_tasks.py)"
