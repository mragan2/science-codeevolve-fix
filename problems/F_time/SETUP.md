# F_time setup and run guide

The steps below reproduce a clean setup under `/home/rag/Projects` and execute the F_time benchmark with the provided run script.

## 1) Clone the repository into `/home/rag/Projects`
```bash
mkdir -p /home/rag/Projects
cd /home/rag/Projects
# If you use SSH, swap for git@github.com:inter-co/science-codeevolve.git
git clone https://github.com/inter-co/science-codeevolve.git
cd science-codeevolve
```

## 2) Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate codeevolve
```

If the environment already exists, update it instead of recreating:
```bash
conda activate base
conda env update -f environment.yml
conda activate codeevolve
```

## 3) Install the package locally
From the repository root, install CodeEvolve in editable mode so the `codeevolve` CLI is available:
```bash
pip install -e .
```

## 4) Provide API credentials (if your LLM provider requires them)
Set the API key and base URL in your shell before running, or source a file that exports them:
```bash
export API_KEY="1e28fb7fb3b5486e88cf34c33127ef71.hpbxvrNGSUlgNGFz6Mgp7q0Z"
export API_BASE="https://api.openai.com/v1"   # replace if using another provider
# or, if you keep them in ~/.codeevolve_api_keys
source ~/.codeevolve_api_keys
```

## 5) Run the F_time benchmark
From the repository root:
```bash
cd /home/rag/Projects/science-codeevolve
bash problems/F_time/run.sh
```

The script automatically resolves the repository root, so you can also run it from inside the problem folder:
```bash
cd /home/rag/Projects/science-codeevolve/problems/F_time
bash run.sh
```

## 6) Verify expected directories
If you see an error like `Input directory does not exist: .../problems/problems/F_time/input/`, ensure you are running the bundled `problems/F_time/run.sh` from this repository so it points to `problems/F_time/input/`. The default layout already includes the necessary `input/` and `configs/` folders.

export API_BASE="http://localhost:11434/v1" && export API_KEY="944ce3c4b46f4aa5a073887d88c18773.955is0NZY-YbcBVD7nzAYtNd" && /home/rag/Projects/science-codeevolve/.conda/bin/codeevolve --inpt_dir="problems/F_time" --cfg_path="problems/F_time/configs/config.yaml" --out_dir="experiments/F_time/run_$(date +%Y%m%d_%H%M%S)" --load_ckpt=-1 --terminal_logging

## 7) Outputs
Runs are written to `experiments/F_time/` with a timestamped subfolder. Check the script output footer for the run status and the exact output path.
