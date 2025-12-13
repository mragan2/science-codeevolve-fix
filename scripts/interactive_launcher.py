"""Interactive setup helper for Science CodeEvolve.

This script simplifies launching experiments by asking only for:
- Project name (within problems/ directory)
- Config name (when using existing config)
- Output directory name (optional, auto-suggested)

All paths are automatically derived from the standard project structure.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, List

import yaml


def prompt_path(
    prompt: str, must_exist: bool = False, create_parents: bool = False, default: str | None = None
) -> Path:
    """Prompt the user for a path and return it as an absolute ``Path``.

    Args:
        prompt: The text to display to the user.
        must_exist: Whether the path must already exist.
        create_parents: Whether to create parent directories if they do not exist.
        default: Optional default value to use when the user presses enter.
    """
    while True:
        suffix = f" [{default}]" if default else ""
        raw_value = input(f"{prompt}{suffix}: ").strip().strip('"')
        if not raw_value and default:
            raw_value = default
        expanded = Path(raw_value).expanduser().resolve()

        if must_exist and not expanded.exists():
            print(f"✖ Path does not exist: {expanded}")
            continue

        if create_parents and not expanded.parent.exists():
            expanded.parent.mkdir(parents=True, exist_ok=True)

        return expanded


def yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            answer = input(f"{question} {suffix}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # If we hit EOF or interrupt, return the default
            print()
            return default
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def parse_scalar(value: str) -> Any:
    """Best-effort parsing that turns simple strings into numbers/bools when possible."""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def prompt_overrides() -> Dict[str, Any]:
    print("Enter any configuration overrides you want to inject.")
    print("Leave the key empty to finish. Values are recorded as typed (numbers/bools auto-detected).")
    overrides: Dict[str, Any] = {}
    while True:
        key = input("Override key (blank to stop): ").strip()
        if not key:
            break
        value = input("Value: ").strip()
        overrides[key] = parse_scalar(value)
    return overrides


def find_projects(base_dir: Path) -> List[str]:
    """Find all project directories in the problems folder.
    
    Args:
        base_dir: Base directory to search (problems/)
        
    Returns:
        List of project names
    """
    if not base_dir.exists():
        return []
    
    projects = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it has input directory (standard structure)
            if (item / "input").exists():
                projects.append(item.name)
    return sorted(projects)


def find_configs(project_dir: Path) -> List[str]:
    """Find all config files in the project's configs directory.
    
    Args:
        project_dir: Path to the project directory
        
    Returns:
        List of config file names (without .yaml/.yml extension)
    """
    configs_dir = project_dir / "configs"
    if not configs_dir.exists():
        return []
    
    configs = []
    for item in configs_dir.iterdir():
        if item.is_file() and item.suffix.lower() in {'.yaml', '.yml'}:
            configs.append(item.stem)
    return sorted(configs)


def find_evaluator(project_dir: Path) -> Optional[Path]:
    """Find the evaluator file in the project's input directory.
    
    Args:
        project_dir: Path to the project directory
        
    Returns:
        Path to the evaluator file, or None if not found
    """
    # Try standard locations
    candidates = [
        project_dir / "input" / "evaluate.py",
        project_dir / "input" / "python" / "evaluate.py",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Search for any evaluate.py
    input_dir = project_dir / "input"
    if input_dir.exists():
        for eval_file in input_dir.rglob("evaluate.py"):
            return eval_file
    
    return None


def find_init_program(project_dir: Path) -> Optional[Path]:
    """Find the initial program file in the project's input directory.
    
    Args:
        project_dir: Path to the project directory
        
    Returns:
        Path to the initial program file, or None if not found
    """
    # Try standard locations
    candidates = [
        project_dir / "input" / "src" / "init_program.py",
        project_dir / "input" / "src" / "initial_program.py",
        project_dir / "input" / "python" / "src" / "init_program.py",
        project_dir / "input" / "python" / "src" / "initial_program.py",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Search for any init*.py or initial*.py in src subdirectories
    input_dir = project_dir / "input"
    if input_dir.exists():
        for pattern in ["**/src/init*.py", "**/src/initial*.py"]:
            for prog_file in input_dir.glob(pattern):
                if prog_file.is_file():
                    return prog_file
    
    return None


def print_conda_hint() -> None:
    """Remind the user how to prepare the conda environment."""

    environment_yml = Path("environment.yml").resolve()
    current_env = os.environ.get("CONDA_DEFAULT_ENV")

    if current_env == "codeevolve":
        print("✅ Conda environment detected: codeevolve")
        return

    print("⚠️  Tip: activate the recommended conda env before running heavy jobs.")
    if environment_yml.exists():
        print(f"    conda env create -f {environment_yml}")
    print("    conda activate codeevolve")
    if current_env:
        print(f"    (currently in '{current_env}'—switch if needed)")


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML or JSON config into a dictionary."""

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def save_config(payload: Mapping[str, Any], path: Path) -> None:
    """Save the config as YAML or JSON based on extension."""

    if path.suffix.lower() in {".yml", ".yaml"}:
        path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def edit_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Prompt the user to tweak each value in a mapping."""

    updated: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            print(f"\n➡️  Section: {key}")
            updated[key] = edit_mapping(value)
            continue

        new_value = input(f"{key} [{value!r}] (enter to keep): ").strip()
        updated[key] = value if not new_value else parse_scalar(new_value)

    return updated


def build_config_payload(
    base_config: Dict[str, Any] | None, allow_edit: bool, allow_overrides: bool
) -> Dict[str, Any]:
    """Build configuration payload with optional edits and overrides.
    
    Note: This doesn't include input_dir or evaluator paths, as those are
    passed via CLI arguments, not config file.
    """

    config_data: Dict[str, Any] = base_config.copy() if base_config else {}

    if allow_edit:
        if config_data:
            print("\nLet's walk through the current config; press enter to keep any value.")
        config_data = edit_mapping(config_data)
    else:
        print("Skipping per-parameter edits; you can adjust later by editing the saved file.")

    if allow_overrides:
        extra_overrides = prompt_overrides()
        config_data.update(extra_overrides)
    
    return config_data


def generate_command_line(
    input_dir: Path, config_path: Path, output_dir: Path
) -> str:
    """Generate the codeevolve command line based on provided paths.
    
    Args:
        input_dir: Path to the input directory
        config_path: Path to the config file
        output_dir: Path to the output directory
        
    Returns:
        Command line string ready to execute
    """
    return (
        f"codeevolve \\\n"
        f"    --inpt_dir=\"{input_dir}\" \\\n"
        f"    --cfg_path=\"{config_path}\" \\\n"
        f"    --out_dir=\"{output_dir}\" \\\n"
        f"    --load_ckpt=-1 \\\n"
        f"    --terminal_logging"
    )


def main() -> None:
    print("🚀 Welcome to the Science CodeEvolve interactive launcher!")
    print("This simplified launcher asks only for your project name and configuration.")
    print()

    print_conda_hint()
    print()

    # Get the problems directory
    repo_root = Path(__file__).resolve().parent.parent
    problems_dir = repo_root / "problems"
    
    if not problems_dir.exists():
        print(f"✖ Problems directory not found at: {problems_dir}")
        print("Please run this script from the repository root.")
        return
    
    # List available projects
    projects = find_projects(problems_dir)
    if projects:
        print(f"📁 Found {len(projects)} project(s):")
        for i, proj in enumerate(projects, 1):
            print(f"   {i}. {proj}")
        print()
    
    # Ask for project name
    try:
        project_name = input("Enter project name: ").strip()
    except EOFError:
        print("\n✖ Input interrupted.")
        return
    
    if not project_name:
        print("✖ Project name cannot be empty.")
        return
    
    project_dir = problems_dir / project_name
    if not project_dir.exists():
        print(f"✖ Project directory does not exist: {project_dir}")
        return
    
    # Find evaluator
    evaluator_path = find_evaluator(project_dir)
    if not evaluator_path:
        print(f"✖ Could not find evaluate.py in project: {project_name}")
        print(f"   Searched in: {project_dir / 'input'}")
        return
    print(f"✅ Found evaluator: {evaluator_path.relative_to(repo_root)}")
    
    # Find initial program
    init_program_path = find_init_program(project_dir)
    if not init_program_path:
        print(f"⚠️  Warning: Could not find initial program file in project: {project_name}")
        print(f"   This may cause issues if not configured properly.")
    else:
        print(f"✅ Found initial program: {init_program_path.relative_to(repo_root)}")
    
    # Input directory (parent of src/ or direct input/)
    input_dir = project_dir / "input"
    if not input_dir.exists():
        print(f"✖ Input directory does not exist: {input_dir}")
        return
    print(f"✅ Input directory: {input_dir.relative_to(repo_root)}")
    print()
    
    # Find configs
    configs = find_configs(project_dir)
    if configs:
        print(f"📋 Found {len(configs)} config file(s):")
        for i, cfg in enumerate(configs, 1):
            print(f"   {i}. {cfg}")
        print()
    
    # Ask if user wants to use existing config
    use_existing_config = yes_no("Do you want to use an existing config?", default=True)
    base_config: Dict[str, Any] | None = None
    config_path: Path
    
    if use_existing_config:
        try:
            config_name = input("Enter config name (without .yaml extension): ").strip()
        except EOFError:
            print("\n✖ Input interrupted.")
            return
        
        if not config_name:
            print("✖ Config name cannot be empty.")
            return
        
        # Try to find the config file
        config_path = project_dir / "configs" / f"{config_name}.yaml"
        if not config_path.exists():
            config_path = project_dir / "configs" / f"{config_name}.yml"
        
        if not config_path.exists():
            print(f"✖ Config file not found: {config_name}.yaml or {config_name}.yml")
            print(f"   Searched in: {project_dir / 'configs'}")
            return
        
        try:
            base_config = load_config(config_path)
            print(f"✅ Loaded config: {config_path.relative_to(repo_root)}")
        except Exception as exc:
            print(f"✖ Could not read config: {exc}")
            return
        
        # Option to save edits to new file
        if yes_no("Save edits to a new file so the original stays pristine?", default=True):
            try:
                new_config_name = input(f"New config name [{config_name}_edited]: ").strip()
            except EOFError:
                new_config_name = ""
                print()
            if not new_config_name:
                new_config_name = f"{config_name}_edited"
            config_path = project_dir / "configs" / f"{new_config_name}.yaml"
    else:
        # Create new config
        try:
            config_name = input("Enter name for new config file (without .yaml): ").strip()
        except EOFError:
            config_name = ""
            print()
        if not config_name:
            config_name = "generated_config"
        
        config_path = project_dir / "configs" / f"{config_name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        base_config = {}
    
    # Ask about editing parameters
    allow_edit = yes_no("Would you like to review/edit parameters?", default=False)
    allow_overrides = yes_no("Add quick overrides on top?", default=False) if not allow_edit else False
    
    # Only build/save config if we're creating new or explicitly editing
    if not use_existing_config or allow_edit or allow_overrides:
        # Build config with modifications
        config_payload = build_config_payload(base_config, allow_edit, allow_overrides)
        
        # Save config
        save_config(config_payload, config_path)
        print(f"\n💾 Saved config to: {config_path.relative_to(repo_root)}")
    else:
        # Using existing config without modifications
        print(f"\n✅ Using config: {config_path.relative_to(repo_root)}")
    
    # Ask for output directory
    print()
    experiments_dir = repo_root / "experiments" / project_name
    default_output = experiments_dir / "run_001"
    
    # Find next available run number
    if experiments_dir.exists():
        existing_runs = [d.name for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if existing_runs:
            run_numbers = []
            for run in existing_runs:
                try:
                    num = int(run.split("_")[1])
                    run_numbers.append(num)
                except (ValueError, IndexError):
                    pass
            if run_numbers:
                next_run = max(run_numbers) + 1
                default_output = experiments_dir / f"run_{next_run:03d}"
    
    try:
        output_choice = input(f"Output directory [{default_output.relative_to(repo_root)}]: ").strip()
    except EOFError:
        output_choice = ""
        print()
    
    if output_choice:
        # Handle both relative and absolute paths
        if output_choice.startswith('/'):
            output_dir = Path(output_choice)
        else:
            output_dir = repo_root / output_choice
    else:
        output_dir = default_output
    
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("✨ Setup Complete! Ready for launch!")
    print("="*70)
    print(f"\n📂 Project: {project_name}")
    print(f"📋 Config: {config_path.relative_to(repo_root)}")
    print(f"📤 Output: {output_dir.relative_to(repo_root)}")
    print()
    
    # Generate command line
    command = generate_command_line(input_dir, config_path, output_dir)
    print("🚀 Run this command to start evolution:")
    print()
    print(command)
    print()
    
    # Optionally save to a shell script
    if yes_no("Save this command to a shell script?", default=True):
        script_name = f"run_{project_name}.sh"
        script_path = repo_root / script_name
        
        script_content = f"""#!/bin/bash
# Auto-generated by interactive_launcher.py
# Project: {project_name}
# Generated: {__import__('datetime').datetime.now().isoformat()}

{command}
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)  # Make executable
        print(f"💾 Saved to: {script_path.name}")
        print(f"   Run with: bash {script_path.name}")
    
    print("\n✅ Thanks for using the launcher—happy experimenting!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Bye!")
    except Exception as exc:
        print(f"\n✖ Unexpected error: {exc}")
        traceback.print_exc()
