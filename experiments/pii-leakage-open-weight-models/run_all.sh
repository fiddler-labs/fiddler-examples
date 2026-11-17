#!/usr/bin/env bash
# Re-run the study's conversations against live models. This costs money.
#
#   ./run_all.sh                  every model, every set (about 400 conversations)
#   ./run_all.sh smoke            one conversation (leak_injection_01 on Gemma 4 31B)
#   ./run_all.sh <model-key>      one model, e.g. ./run_all.sh gpt-oss-120b
#
# Needs VERCEL_AI_GATEWAY_API_KEY in tau2-bench/.env (see .env.example) and
# ./setup.sh first. Results land in tau2-bench/data/simulations/rerun-*. Score
# them with:
#   (cd tau2-bench && uv run --no-sync python scripts/check_banking_leaks.py \
#       data/simulations/rerun-<model>-<set>/results.json)
#
# The settings below are the ones recorded in results/*/results.json.gz. Each task
# ran once; with temperature 0 and a fixed script most conversations repeat, but
# providers are not deterministic, so expect a few tasks to change outcome.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/tau2-bench"
[ -f .env ] || { echo "Create tau2-bench/.env with VERCEL_AI_GATEWAY_API_KEY (see .env.example)" >&2; exit 1; }

GW=vercel_ai_gateway
# key | gateway model | agent llm args (Mistral Large 4 and the Claude models
# reject temperature 0 on the gateway, so they ran at their default)
MODELS=(
  "nemotron-3-5-lightning|$GW/nvidia/nemotron-3.5-lightning-free|{\"temperature\":0,\"timeout\":60,\"num_retries\":0,\"max_tokens\":8192}"
  "nemotron-3-ultra|$GW/nvidia/nemotron-3-ultra-550b-a55b|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "gpt-oss-120b|$GW/openai/gpt-oss-120b|{\"temperature\":0,\"timeout\":60,\"num_retries\":0,\"max_tokens\":8192}"
  "gemma-4-31b|$GW/google/gemma-4-31b-it|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "mistral-large-4|$GW/mistral/mistral-large-4|{\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "claude-haiku-5-5|$GW/anthropic/claude-haiku-5.5|{\"timeout\":60,\"num_retries\":0,\"max_tokens\":8192}"
)
# Injection screening: leak_injection_01 once on each model.
SCREENING=(
  "gemma-4-31b|$GW/google/gemma-4-31b-it|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "mistral-large-3|$GW/mistral/mistral-large-3|{\"temperature\":0,\"timeout\":60,\"num_retries\":0,\"max_tokens\":8192}"
  "mistral-large-4|$GW/mistral/mistral-large-4|{\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "glm-5-3-flash|$GW/zai/glm-5.3-flash|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "deepseek-v4-1-flash|$GW/deepseek/deepseek-v4.1-flash|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "nemotron-3-ultra|$GW/nvidia/nemotron-3-ultra-550b-a55b|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "claude-sonnet-5|$GW/anthropic/claude-sonnet-5|{\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "gemini-3-8-flash|$GW/google/gemini-3.8-flash|{\"temperature\":0,\"timeout\":120,\"num_retries\":0,\"max_tokens\":8192}"
  "claude-haiku-5-5|$GW/anthropic/claude-haiku-5.5|{\"timeout\":60,\"num_retries\":0,\"max_tokens\":8192}"
)

scripted() {  # name model args split [extra tau2 args...]
  local name=$1 model=$2 args=$3 split=$4; shift 4
  uv run --no-sync tau2 run --domain banking_knowledge \
    --task-set-name banking_knowledge_leak --task-split-name "$split" \
    --user scripted_user --agent-llm "$model" --agent-llm-args "$args" \
    --retrieval-config bm25 --num-trials 1 --max-steps 200 \
    --save-to "rerun-$name" "$@"
}

adaptive() {  # name model args
  uv run --no-sync tau2 run --domain banking_knowledge \
    --task-set-name banking_knowledge_leak --task-split-name adaptive \
    --user user_simulator --user-llm "$GW/openai/gpt-4.1" \
    --user-llm-args '{"temperature":0.8}' \
    --agent-llm "$2" --agent-llm-args "$3" \
    --retrieval-config bm25 --num-trials 3 --max-steps 200 \
    --save-to "rerun-$1-adaptive"
}

want=${1:-all}

if [ "$want" = smoke ]; then
  IFS='|' read -r key model args <<<"${SCREENING[0]}"
  scripted "$key-smoke" "$model" "$args" everything --task-ids leak_injection_01
  exit 0
fi

for entry in "${MODELS[@]}"; do
  IFS='|' read -r key model args <<<"$entry"
  [ "$want" = all ] || [ "$want" = "$key" ] || continue
  scripted "$key-base" "$model" "$args" all          # 35 attacks + 10 controls
  scripted "$key-hard" "$model" "$args" hard         # 20 attacks
  # Only Claude Haiku 5.5 faced the adaptive attacker in the article.
  [ "$key" = claude-haiku-5-5 ] && adaptive "$key" "$model" "$args"
done

if [ "$want" = all ] || [ "$want" = screening ]; then
  for entry in "${SCREENING[@]}"; do
    IFS='|' read -r key model args <<<"$entry"
    scripted "$key-screening" "$model" "$args" everything --task-ids leak_injection_01
  done
fi
