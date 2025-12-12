# CodeEvolve Problems Directory

This directory contains problem definitions and configurations for running CodeEvolve experiments.

## Directory Structure

Each project follows a standardized structure:

```
problems/
├── PROJECT_NAME/
│   ├── input/
│   │   ├── evaluate.py          # Evaluation script (required)
│   │   └── src/
│   │       └── init_program.py  # Initial program (required)
│   └── configs/
│       ├── config.yaml          # Main configuration
│       ├── config_mp_insp.yaml  # Config with meta-prompting and inspiration
│       ├── config_insp.yaml     # Config with inspiration only
│       ├── config_mp.yaml       # Config with meta-prompting only
│       └── config_no_evolve.yaml # Config for baseline evaluation
└── run_template.sh              # Generic run script template
```

### Required Files

For any project `PROJECT_NAME`:

1. **`input/src/init_program.py`** - The initial program to evolve
2. **`input/evaluate.py`** - Script to evaluate program fitness
3. **`configs/config.yaml`** - Configuration file with evolution parameters

## Quick Start

### Option 1: Using the Template Script

1. Copy the template to your project directory:
   ```bash
   cp problems/run_template.sh problems/YOUR_PROJECT/run.sh
   ```

2. Edit the configuration variables at the top of `run.sh`:
   ```bash
   PROJECT_NAME="YOUR_PROJECT"  # e.g., "F_time" or "alphaevolve_math_problems/heilbronn_convex/13"
   CONFIG_NAME="config"          # or "config_mp_insp", etc.
   ```

3. Run the script:
   ```bash
   cd problems/YOUR_PROJECT
   bash run.sh
   ```

### Option 2: Direct Command Line

Run CodeEvolve directly using the command line:

```bash
codeevolve \
    --inpt_dir="problems/PROJECT_NAME/input/" \
    --cfg_path="problems/PROJECT_NAME/configs/config.yaml" \
    --out_dir="experiments/PROJECT_NAME/run_001/" \
    --load_ckpt=-1 \
    --terminal_logging
```

## Project Examples

### Example 1: Simple Project Structure

```
problems/F_time/
├── input/
│   ├── evaluate.py
│   └── src/
│       └── init_program.py
└── configs/
    └── config.yaml
```

### Example 2: Hierarchical Project Structure

```
problems/alphaevolve_math_problems/heilbronn_convex/13/
├── input/
│   ├── evaluate.py
│   └── src/
│       └── init_program.py
└── configs/
    └── config.yaml
```

## Configuration Files

Different configuration variants enable different evolutionary features:

- **`config.yaml`** - Standard configuration
- **`config_mp_insp.yaml`** - Meta-prompting + Inspiration (most features)
- **`config_insp.yaml`** - Inspiration crossover only
- **`config_mp.yaml`** - Meta-prompting only
- **`config_no_evolve.yaml`** - Baseline evaluation without evolution

Choose the configuration that matches your experimental needs.

## Creating a New Project

1. Use the `problem_template` as a starting point:
   ```bash
   cp -r problems/problem_template problems/YOUR_PROJECT
   ```

2. Modify the files:
   - `input/src/init_program.py` - Your initial solution
   - `input/evaluate.py` - Your fitness evaluation logic
   - `configs/config.yaml` - Evolution parameters

3. Create a run script:
   ```bash
   cp problems/run_template.sh problems/YOUR_PROJECT/run.sh
   ```

4. Edit `run.sh` to set `PROJECT_NAME="YOUR_PROJECT"`

5. Run your experiment:
   ```bash
   cd problems/YOUR_PROJECT
   bash run.sh
   ```

## Template Variables

When using `run_template.sh`, you can customize:

| Variable | Description | Example |
|----------|-------------|---------|
| `PROJECT_NAME` | Project path relative to `problems/` | `"F_time"` or `"alphaevolve_math_problems/heilbronn_convex/13"` |
| `CONFIG_NAME` | Config file name (without `.yaml`) | `"config"` or `"config_mp_insp"` |
| `OUTPUT_NAME` | Output directory name | `"run_001"` (auto-generated with timestamp by default) |
| `LOAD_CKPT` | Checkpoint epoch to resume from | `-1` (start fresh) or `50` (resume from epoch 50) |
| `CPU_LIST` | CPU affinity specification | `""` (all CPUs) or `"0-7"` or `"0,2,4,6"` |

## Output Structure

Results are saved to `experiments/PROJECT_NAME/OUTPUT_NAME/`:

```
experiments/PROJECT_NAME/run_001/
├── checkpoints/        # Saved evolution checkpoints
├── logs/              # Execution logs
└── results/           # Final results and best solutions
```

## Troubleshooting

### Error: "codeevolve command not found"

Install the package:
```bash
pip install -e .
```

### Error: "Input directory does not exist"

Check that your project follows the required structure:
- `problems/PROJECT_NAME/input/` must exist
- `problems/PROJECT_NAME/input/evaluate.py` must exist
- `problems/PROJECT_NAME/input/src/init_program.py` should exist

### Error: "Config file does not exist"

Check available configs:
```bash
ls problems/PROJECT_NAME/configs/
```

Use one of the available config names (without `.yaml` extension).

## Additional Resources

- See `OPTIMIZATIONS.md` for performance tuning recommendations
- Check individual problem directories for problem-specific documentation
- Refer to the main README for general CodeEvolve usage
