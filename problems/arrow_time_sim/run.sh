#!/bin/bash
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Run script for the F_time problem.
# Copied from problems/run_template.sh and customized for this project.
#
# Usage:
#   cd problems/F_time && bash run.sh
#   # or from repo root:
#   bash problems/F_time/run.sh
#
# ===--------------------------------------------------------------------------------------===#

# ==================================
# CONFIGURATION - EDIT THESE VALUES
# ==================================

# Project name relative to the problems/ directory
PROJECT_NAME="arrow_time_sim"

# Config file name (without .yaml extension)
CONFIG_NAME="config"

# Output directory name (will be created under experiments/)
OUTPUT_NAME="run_$(date +%Y%m%d_%H%M%S)"

# Checkpoint to load (-1 for no checkpoint, or epoch number to resume from)
LOAD_CKPT=-1

# CPU affinity (leave empty for no restriction, or specify like "0-7" or "0,2,4,6")
CPU_LIST=""

# ==================================
# API CONFIGURATION (OPTIONAL)
# ==================================
# You can set API credentials here or use environment variables
# If set here, they will override environment variables

# Option 1: Set API key directly (NOT RECOMMENDED for shared/public projects)
# API_KEY="your-api-key-here"
# API_BASE="https://api.openai.com/v1"

# Option 2: Use environment variables (RECOMMENDED)
# Leave commented out to use existing environment variables
# Or set them here to override:
# export API_KEY="${API_KEY:1e28fb7fb3b5486e88cf34c33127ef71.hpbxvrNGSUlgNGFz6Mgp7q0Z}"
# export API_BASE="${API_BASE:http://localhost:11434/v1}"

# Option 3: Load from external file (MOST SECURE)
# Create a file with: export API_KEY="..." and export API_BASE="..."
# Then uncomment the line below:
#source ~/.codeevolve_api_keys

# ==================================
# AUTOMATIC PATH SETUP - DO NOT EDIT
# ==================================

# Get the absolute path to the science-codeevolve directory.
# We try git first (works from any subdirectory), then fall back to walking
# up the tree until we find a .git folder. Finally, default to one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v git &> /dev/null; then
    REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"
fi

if [ -z "${REPO_ROOT}" ]; then
    SEARCH_DIR="${SCRIPT_DIR}"
    while [ "${SEARCH_DIR}" != "/" ]; do
        if [ -d "${SEARCH_DIR}/.git" ]; then
            REPO_ROOT="${SEARCH_DIR}"
            break
        fi
        NEXT_DIR="$(cd "${SEARCH_DIR}/.." && pwd)"
        if [ "${NEXT_DIR}" = "${SEARCH_DIR}" ]; then
            break
        fi
        SEARCH_DIR="${NEXT_DIR}"
    done
fi

REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Construct paths based on the standard project structure:
# - Problem base directory: problems/PROJECT_NAME/
# - initial_program.py: problems/PROJECT_NAME/input/src/
# - evaluate.py: problems/PROJECT_NAME/input/
# - config.yaml: problems/PROJECT_NAME/configs/
BASE_DIR="${REPO_ROOT}/problems/${PROJECT_NAME}"
INPT_DIR="${BASE_DIR}/"
CFG_PATH="${BASE_DIR}/configs/${CONFIG_NAME}.yaml"
OUT_DIR="${REPO_ROOT}/experiments/${PROJECT_NAME}/${OUTPUT_NAME}"

# ==================================
# VALIDATION
# ==================================

echo "======================================"
echo "CodeEvolve Run Configuration"
echo "======================================"
echo "Project Name:    ${PROJECT_NAME}"
echo "Input Directory: ${INPT_DIR}"
echo "Config File:     ${CFG_PATH}"
echo "Output Directory: ${OUT_DIR}"
echo "Load Checkpoint: ${LOAD_CKPT}"
echo "CPU List:        ${CPU_LIST:-'(all CPUs)'}"
echo "======================================"
echo ""

# Check if required directories and files exist
if [ ! -d "${INPT_DIR}" ]; then
    echo "ERROR: Input directory does not exist: ${INPT_DIR}"
    echo "Expected structure: problems/${PROJECT_NAME}/"
    exit 1
fi

if [ ! -f "${CFG_PATH}" ]; then
    echo "ERROR: Config file does not exist: ${CFG_PATH}"
    echo "Available configs in ${BASE_DIR}/configs/:"
    ls -1 "${BASE_DIR}/configs/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

if [ ! -f "${INPT_DIR}/input/evaluate.py" ]; then
    echo "ERROR: evaluate.py not found in ${INPT_DIR}/input/"
    echo "Expected: ${INPT_DIR}/input/evaluate.py"
    exit 1
fi

if [ ! -f "${INPT_DIR}/input/src/initial_program.py" ] && [ ! -f "${INPT_DIR}/input/src/init_program.py" ]; then
    echo "WARNING: No initial program found in ${INPT_DIR}/input/src/"
    echo "Expected one of:"
    echo "  - ${INPT_DIR}/input/src/initial_program.py (default)"
    echo "  - ${INPT_DIR}/input/src/init_program.py (legacy)"
fi

# Check if codeevolve command is available
CODEEVOLVE_CMD=()

# Prefer a repo-local conda env if present (works even when not activated).
REPO_CONDA_PY="${REPO_ROOT}/.conda/bin/python"
REPO_CONDA_CODEEVOLVE="${REPO_ROOT}/.conda/bin/codeevolve"

PYTHON_BIN=""
if [ -n "${CODEEVOLVE_PYTHON}" ] && [ -x "${CODEEVOLVE_PYTHON}" ]; then
    PYTHON_BIN="${CODEEVOLVE_PYTHON}"
elif [ -x "${REPO_CONDA_PY}" ]; then
    PYTHON_BIN="${REPO_CONDA_PY}"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
fi

if command -v codeevolve &> /dev/null; then
    CODEEVOLVE_CMD=(codeevolve)
elif [ -x "${REPO_CONDA_CODEEVOLVE}" ]; then
    CODEEVOLVE_CMD=("${REPO_CONDA_CODEEVOLVE}")
else
    # Fall back to running the module directly from the repo.
    # This avoids requiring an editable install just to run a local experiment.
    if [ -z "${PYTHON_BIN}" ]; then
        echo "ERROR: Neither 'codeevolve' nor a usable Python interpreter was found."
        echo "Expected one of: codeevolve in PATH, ${REPO_CONDA_CODEEVOLVE}, python/python3 in PATH, or CODEEVOLVE_PYTHON=/path/to/python"
        exit 1
    fi

    export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
    CODEEVOLVE_CMD=("${PYTHON_BIN}" -m codeevolve.cli)
    echo "NOTE: 'codeevolve' CLI not found; using: ${PYTHON_BIN} -m codeevolve.cli"
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# ==================================
# API KEY SETUP
# ==================================

# Export API keys if they were set in the configuration section above
if [ ! -z "${API_KEY}" ]; then
    export API_KEY
    echo "Using API_KEY from run script configuration"
fi

if [ ! -z "${API_BASE}" ]; then
    export API_BASE
    echo "Using API_BASE from run script: ${API_BASE}"
fi

# Check if API keys are available (from any source)

# The CodeEvolve CLI currently requires both variables to exist in the environment.
# For local/self-hosted endpoints, API_KEY is often unused; exporting it as an empty
# string is sufficient.
if [ -z "${API_BASE+x}" ]; then
    export API_BASE="http://localhost:11434/v1"
    echo "NOTE: API_BASE not set; defaulting to ${API_BASE}"
fi

if [ -z "${API_KEY+x}" ]; then
    export API_KEY="ollama"
    echo "NOTE: API_KEY not set; defaulting to ${API_KEY}"
fi

if [ -z "${API_KEY}" ]; then
    echo "WARNING: API_KEY is empty. The run may fail if your LLM requires authentication."
    echo "Set it via:"
    echo "  1. Environment variable: export API_KEY='your-key'"
    echo "  2. In this run.sh file (see API CONFIGURATION section)"
    echo "  3. External file: source ~/.codeevolve_api_keys"
    echo ""
fi

# ==================================
# RUN CODEEVOLVE
# ==================================

echo "Starting CodeEvolve..."
echo ""

CODEEVOLVE_ARGS=(
    --inpt_dir="${INPT_DIR}"
    --cfg_path="${CFG_PATH}"
    --out_dir="${OUT_DIR}"
    --load_ckpt="${LOAD_CKPT}"
    --terminal_logging
)

if [ -n "${CPU_LIST}" ]; then
    # Run with CPU affinity
    if command -v taskset &> /dev/null; then
        taskset --cpu-list "${CPU_LIST}" "${CODEEVOLVE_CMD[@]}" "${CODEEVOLVE_ARGS[@]}"
    else
        echo "WARNING: 'taskset' not found; running without CPU affinity."
        "${CODEEVOLVE_CMD[@]}" "${CODEEVOLVE_ARGS[@]}"
    fi
else
    # Run without CPU affinity
    "${CODEEVOLVE_CMD[@]}" "${CODEEVOLVE_ARGS[@]}"
fi

# ==================================
# COMPLETION
# ==================================

EXIT_CODE=$?
echo ""
echo "======================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "CodeEvolve completed successfully!"
    echo "Results saved to: ${OUT_DIR}"
else
    echo "CodeEvolve exited with error code: ${EXIT_CODE}"
fi
echo "======================================"

exit ${EXIT_CODE}
