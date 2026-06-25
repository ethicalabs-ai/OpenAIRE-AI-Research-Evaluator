#!/usr/bin/env bash
# run_judges.sh — batch LLM-as-Judge across sources and models
#
# Usage:
#   ./scripts/run_judges.sh 100              # sync — 100 papers per source per model
#   ./scripts/run_judges.sh 100 --async       # async — dispatch to Celery worker
#   ./scripts/run_judges.sh 50 --dry-run      # sync, skip DB writes
#   ./scripts/run_judges.sh 100 --async --dry-run
set -euo pipefail

N="${1:?Usage: $0 <papers_per_source> [--async] [--dry-run]}"
shift || true

# ── Parse mode flags ──────────────────────────────────────────────────────────
ASYNC=false
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --async) ASYNC=true ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# ── Hardcoded model list ──────────────────────────────────────────────────────
MODELS=(
    "Qwen3.6-35B-A3B-MTP-GGUF"
    "Qwen3.6-27B-MTP-GGUF"
    "Qwen3.5-35B-A3B-GGUF"
    "Nemotron-3-Nano-30B-A3B-GGUF"
    "Gemma-4-26B-A4B-it-GGUF"
    "gpt-oss-20b-mxfp4-GGUF"
    "gpt-oss-120b-mxfp-GGUF"
    "DeepSeek-Qwen3-8B-GGUF"
    "GLM-4.7-Flash-GGUF"
    "Qwen3.6-35B-A3B-GGUF"
    "Qwen3.5-4B-GGUF"
)

SOURCES=(catalog)

# ── Resolve container name ────────────────────────────────────────────────────
CONTAINER="echo_dsrn_graph_web"
LEMONADE_HOST="192.168.1.40"

MODE_DISPLAY="SYNC"
$ASYNC && MODE_DISPLAY="ASYNC (Celery)"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  LLM-as-Judge batch runner                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Mode              : ${MODE_DISPLAY}"
echo "║  Papers per source : ${N}"
echo "║  Models            : ${MODELS[*]}"
echo "║  Sources           : ${SOURCES[*]}"
echo "║  Container         : ${CONTAINER}"
echo "║  Extra args        : ${EXTRA_ARGS[*]:-none}"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# ── Async mode: dispatch only, no model loading ───────────────────────────────
if $ASYNC; then
    for model in "${MODELS[@]}"; do
        for source in "${SOURCES[@]}"; do
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  Dispatching: ${model}  |  ${source}  |  ${N} papers"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            docker exec "${CONTAINER}" \
                python backend/judge_cli.py \
                    --source "${source}" \
                    --n "${N}" \
                    --model "${model}" \
                    --delay 0.05 \
                    --async \
                    "${EXTRA_ARGS[@]}"

            echo
        done
    done

    echo "════════════════════════════════════════════════════════════"
    echo "  All tasks dispatched. Worker is processing."
    echo "════════════════════════════════════════════════════════════"
    exit 0
fi

# ── Sync mode: load models on lemonade, judge inline ──────────────────────────
echo "Unloading any previously-loaded models..."
lemonade --host "$LEMONADE_HOST" unload 2>/dev/null || true
echo

for model in "${MODELS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Model: ${model}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    echo "  Loading ${model} (pinned)..."
    set +e
    lemonade --host "$LEMONADE_HOST" load --pinned "${model}"
    _load_rc=$?
    set -e
    if [ $_load_rc -ne 0 ]; then
        echo "  ⚠ Failed to load ${model} — skipping"
        continue
    fi
    echo "  ✓ ${model} loaded"

    for source in "${SOURCES[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Source: ${source}"
        echo "  Count : ${N}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        docker exec "${CONTAINER}" \
            python backend/judge_cli.py \
                --source "${source}" \
                --n "${N}" \
                --model "${model}" \
                --delay 2 \
                "${EXTRA_ARGS[@]}"

        echo
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Unloading model: ${model}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    lemonade --host "$LEMONADE_HOST" unload "${model}" 2>/dev/null || true
    echo
done

echo "════════════════════════════════════════════════════════════"
echo "  All done."
echo "════════════════════════════════════════════════════════════"
