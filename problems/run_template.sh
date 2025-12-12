#!/bin/bash
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Generic template for running CodeEvolve on any project in the problems directory.
#
# Usage:
#   1. Copy this template to your project directory
#   2. Set the PROJECT_NAME variable to your project path (relative to problems/)
#   3. Adjust CONFIG_NAME if using a different config file
#   4. Run: bash run_template.sh
#
# ===--------------------------------------------------------------------------------------===#

# ==================================
# CONFIGURATION - EDIT THESE VALUES
# ==================================

# Project name relative to the problems/ directory
# Examples:
#   - "F_time"
#   - "alphaevolve_math_problems/circle_packing_square/26"
#   - "problem_template"
PROJECT_NAME="F_time"

# Config file name (without .yaml extension)
# Common options: config, config_mp_insp, config_insp, config_mp, config_no_evolve
CONFIG_NAME="config"

# Output directory name (will be created under experiments/)
OUTPUT_NAME="run_$(date +%Y%m%d_%H%M%S)"

# Checkpoint to load (-1 for no checkpoint, or epoch number to resume from)
LOAD_CKPT=-1

# CPU affinity (leave empty for no restriction, or specify like "0-7" or "0,2,4,6")
CPU_LIST=""

# ==================================
# AUTOMATIC PATH SETUP - DO NOT EDIT
# ==================================

# Get the absolute path to the science-codeevolve directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Construct paths based on the standard project structure:
# - init_program.py is always in: problems/PROJECT_NAME/input/src/
# - evaluate.py is always in: problems/PROJECT_NAME/input/
# - config.yaml is in: problems/PROJECT_NAME/configs/
BASE_DIR="${REPO_ROOT}/problems/${PROJECT_NAME}"
INPT_DIR="${BASE_DIR}/input/"
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
    echo "Expected structure: problems/${PROJECT_NAME}/input/"
    exit 1
fi

if [ ! -f "${CFG_PATH}" ]; then
    echo "ERROR: Config file does not exist: ${CFG_PATH}"
    echo "Available configs in ${BASE_DIR}/configs/:"
    ls -1 "${BASE_DIR}/configs/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

if [ ! -f "${INPT_DIR}/evaluate.py" ]; then
    echo "ERROR: evaluate.py not found in ${INPT_DIR}"
    echo "Expected: ${INPT_DIR}/evaluate.py"
    exit 1
fi

if [ ! -f "${INPT_DIR}/src/init_program.py" ]; then
    echo "WARNING: init_program.py not found in ${INPT_DIR}/src/"
    echo "Expected: ${INPT_DIR}/src/init_program.py"
fi

# Check if codeevolve command is available
if ! command -v codeevolve &> /dev/null; then
    echo "ERROR: codeevolve command not found. Please install the package:"
    echo "  pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# ==================================
# RUN CODEEVOLVE
# ==================================

echo "Starting CodeEvolve..."
echo ""

if [ -n "${CPU_LIST}" ]; then
    # Run with CPU affinity
    taskset --cpu-list "${CPU_LIST}" codeevolve \
        --inpt_dir="${INPT_DIR}" \
        --cfg_path="${CFG_PATH}" \
        --out_dir="${OUT_DIR}" \
        --load_ckpt="${LOAD_CKPT}" \
        --terminal_logging
else
    # Run without CPU affinity
    codeevolve \
        --inpt_dir="${INPT_DIR}" \
        --cfg_path="${CFG_PATH}" \
        --out_dir="${OUT_DIR}" \
        --load_ckpt="${LOAD_CKPT}" \
        --terminal_logging
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
