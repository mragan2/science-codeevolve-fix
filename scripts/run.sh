#!/usr/bin/env bash
# ===--------------------------------------------------------------------------------------===#
#
# CodeEvolve Linux runner.
# Fill in your problem name (or pass it as the first argument) and this script
# will point CodeEvolve at the correct input, config, and output folders.
#
# ===--------------------------------------------------------------------------------------===#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

mapfile -t AVAILABLE_PROBLEMS < <(find "$REPO_ROOT/problems" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)

if ((${#AVAILABLE_PROBLEMS[@]} == 0)); then
    echo "✖ No problems found in $REPO_ROOT/problems" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    PROBLEM_NAME="$1"
else
    echo "Available problems:"
    for p in "${AVAILABLE_PROBLEMS[@]}"; do
        echo "  - $p"
    done
    DEFAULT_PROBLEM="${AVAILABLE_PROBLEMS[0]}"
    read -r -p "Choose problem [${DEFAULT_PROBLEM}]: " PROBLEM_NAME
    PROBLEM_NAME="${PROBLEM_NAME:-$DEFAULT_PROBLEM}"
fi

BASE_DIR="${REPO_ROOT}/problems/${PROBLEM_NAME}"
INPUT_DIR="${BASE_DIR}/input"
CONFIG_DIR="${BASE_DIR}/configs"
CONFIG_PATH=""
REQUESTED_CONFIG=${2:-${CONFIG_CHOICE:-}}
RUN_NAME="${RUN_NAME:-}" # optional env override for output folder naming

declare -A API_KEYS

DEFAULT_RUN_NAME=$(date +"%Y%m%d_%H%M%S")
if [[ -z "$RUN_NAME" ]]; then
    read -r -p "Run name under experiments/${PROBLEM_NAME} [${DEFAULT_RUN_NAME}]: " RUN_NAME
    RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
fi
OUTPUT_DIR="${REPO_ROOT}/experiments/${PROBLEM_NAME}/${RUN_NAME}"
LOAD_CKPT="${LOAD_CKPT:--1}"
CPU_LIST="${CPU_LIST:-}"

echo "\nOptional: set API keys for this run (stored only in memory)."
while true; do
    read -r -p "API key env var name (e.g., OPENAI_API_KEY) [skip]: " API_KEY_NAME
    API_KEY_NAME=${API_KEY_NAME:-}
    if [[ -z "$API_KEY_NAME" ]]; then
        break
    fi
    read -sr -p "Value for $API_KEY_NAME: " API_KEY_VALUE
    echo
    if [[ -z "$API_KEY_VALUE" ]]; then
        echo "Skipped empty value for $API_KEY_NAME"
        continue
    fi
    API_KEYS["$API_KEY_NAME"]="$API_KEY_VALUE"
    export "$API_KEY_NAME"="$API_KEY_VALUE"
done

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "✖ Input directory not found: $INPUT_DIR" >&2
    exit 1
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "✖ Config directory not found: $CONFIG_DIR" >&2
    exit 1
fi

mapfile -t AVAILABLE_CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f \( -iname '*.yaml' -o -iname '*.yml' -o -iname '*.json' \) -printf '%f\n' | sort)

choose_config() {
    local choice=${1:-}
    if [[ -n "$choice" && "$choice" =~ ^[0-9]+$ ]]; then
        local idx=$((choice - 1))
        if ((idx >= 0 && idx < ${#AVAILABLE_CONFIGS[@]})); then
            CONFIG_PATH="$CONFIG_DIR/${AVAILABLE_CONFIGS[$idx]}"
            return 0
        fi
    elif [[ -n "$choice" ]]; then
        local candidate="$CONFIG_DIR/$choice"
        if [[ -f "$candidate" ]]; then
            CONFIG_PATH="$candidate"
            return 0
        fi
    fi
    return 1
}

if ((${#AVAILABLE_CONFIGS[@]} > 0)); then
    echo "Available configs in $CONFIG_DIR:"
    for i in "${!AVAILABLE_CONFIGS[@]}"; do
        printf '  [%d] %s\n' "$((i + 1))" "${AVAILABLE_CONFIGS[$i]}"
    done
    echo "  [N] Provide another config file to copy here"
    DEFAULT_CHOICE=1
    if [[ -z "$REQUESTED_CONFIG" ]]; then
        read -r -p "Choose config [$DEFAULT_CHOICE]: " CONFIG_CHOICE
        CONFIG_CHOICE=${CONFIG_CHOICE:-$DEFAULT_CHOICE}
    else
        CONFIG_CHOICE="$REQUESTED_CONFIG"
        echo "Using requested config selector: $CONFIG_CHOICE"
    fi
    if ! choose_config "$CONFIG_CHOICE"; then
        if [[ "${CONFIG_CHOICE,,}" != "n" ]]; then
            echo "✖ Invalid choice: $CONFIG_CHOICE" >&2
            exit 1
        fi
    fi
fi

if [[ -z "$CONFIG_PATH" ]]; then
    if [[ -n "$REQUESTED_CONFIG" && -f "$REQUESTED_CONFIG" ]]; then
        CUSTOM_CONFIG="$REQUESTED_CONFIG"
        echo "Copying requested config file: $CUSTOM_CONFIG"
    else
        read -r -p "Path to config to copy into $CONFIG_DIR: " CUSTOM_CONFIG
    fi
    if [[ -z "$CUSTOM_CONFIG" ]]; then
        echo "✖ No config provided" >&2
        exit 1
    fi
    if [[ ! -f "$CUSTOM_CONFIG" ]]; then
        echo "✖ Config file not found: $CUSTOM_CONFIG" >&2
        exit 1
    fi

    CUSTOM_CONFIG_ABS=$(python - <<'PY'
import os, sys
path = sys.argv[1]
print(os.path.abspath(os.path.expanduser(path)))
PY
"$CUSTOM_CONFIG")

    DEFAULT_NAME="$(basename -- "$CUSTOM_CONFIG_ABS")"
    read -r -p "Save as [$DEFAULT_NAME]: " CUSTOM_NAME
    CUSTOM_NAME=${CUSTOM_NAME:-$DEFAULT_NAME}
    CONFIG_PATH="$CONFIG_DIR/$CUSTOM_NAME"
    cp -f -- "$CUSTOM_CONFIG_ABS" "$CONFIG_PATH"
    echo "Copied custom config to: $CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "✖ Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if ! command -v codeevolve >/dev/null 2>&1; then
    echo "⚠️  'codeevolve' CLI not found in PATH. Activate your env first: conda activate codeevolve" >&2
    exit 1
fi

echo "➡️  Using problem: $PROBLEM_NAME"
echo "   Input:  $INPUT_DIR"
echo "   Config: $CONFIG_PATH"
echo "   Output: $OUTPUT_DIR"

cmd=(
    codeevolve
    --inpt_dir="$INPUT_DIR"
    --cfg_path="$CONFIG_PATH"
    --out_dir="$OUTPUT_DIR"
    --load_ckpt="$LOAD_CKPT"
    --terminal_logging
)

echo "\nTip: conda activate codeevolve  # ensure the environment is ready"

set +e
if [[ -n "$CPU_LIST" ]]; then
    echo "Pinning to CPUs: $CPU_LIST"
    taskset --cpu-list "$CPU_LIST" "${cmd[@]}"
else
    "${cmd[@]}"
fi
status=$?
set -e

if ((${#API_KEYS[@]} > 0)); then
    echo "Cleaning up API key variables..."
    for key in "${!API_KEYS[@]}"; do
        unset "$key"
    done
fi

exit $status
