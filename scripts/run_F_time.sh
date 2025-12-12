#!/bin/bash

set -euo pipefail

# Ensure we run from repo root so relative paths resolve.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Default to local Ollama OpenAI-compatible endpoint if not already provided.
: "${API_BASE:=http://localhost:11434/v1}"
: "${API_KEY:=ollama}"
export API_BASE API_KEY

PROB_NAME="F_time"
BASE_DIR="problems/${PROB_NAME}"
INPT_DIR="${BASE_DIR}/"
CFG_PATH="${BASE_DIR}/configs/config.yaml"
OUT_DIR="experiments/${PROB_NAME}/test/"
RESULTS_DIR="${OUT_DIR}"
LOAD_CKPT=0
CPU_LIST="0"

# Ensure imports work when running from source.
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Pick a Python interpreter that can import PyYAML.
PYTHON_BIN=""
for candidate in \
	"/home/rag/miniconda3/envs/codeevolve-py313/bin/python" \
	"${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
	python \
	python3; do
	if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1; then
		if "$candidate" -c 'import yaml' >/dev/null 2>&1; then
			PYTHON_BIN="$candidate"
			break
		fi
	fi
done

if [[ -z "$PYTHON_BIN" ]]; then
	echo "Could not find a Python interpreter with PyYAML installed (import yaml failed)." >&2
	echo "Activate your env (e.g. conda env) and/or install PyYAML, then re-run." >&2
	exit 1
fi

# Invoke CLI via module to avoid relying on a PATH-installed console script.
RUN_CMD=("$PYTHON_BIN" -m codeevolve.cli --inpt_dir="$INPT_DIR" --cfg_path="$CFG_PATH" --out_dir="$OUT_DIR" --load_ckpt="$LOAD_CKPT")

if command -v taskset >/dev/null 2>&1; then
	taskset --cpu-list "$CPU_LIST" "${RUN_CMD[@]}"
else
	"${RUN_CMD[@]}"
fi

