"""Interactive setup helper for Science CodeEvolve.

This script guides you through selecting the project object, evaluator,
configuration, and optional overrides. All provided paths are expanded to
absolute paths so they can be passed directly to other tooling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

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
        answer = input(f"{question} {suffix}: ").strip().lower()
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
    object_path: Path, evaluator_path: Path, base_config: Dict[str, Any] | None, allow_edit: bool
) -> Dict[str, Any]:
    """Combine object/evaluator paths with existing or new configuration content."""

    config_data: Dict[str, Any] = base_config.copy() if base_config else {}
    config_data["object"] = str(object_path)
    config_data["evaluator"] = str(evaluator_path)

    if allow_edit:
        if config_data:
            print("\nLet's walk through the current config; press enter to keep any value.")
        config_data = edit_mapping(config_data)
    else:
        print("Skipping per-parameter edits; you can adjust later by editing the saved file.")

    extra_overrides = prompt_overrides() if yes_no("Add quick overrides on top?", default=False) else {}
    config_data.update(extra_overrides)
    return config_data


def main() -> None:
    print("🚀 Welcome to the Science CodeEvolve interactive launcher!")
    print("You'll be prompted for paths, optional configuration tweaks, and environment tips.")

    print_conda_hint()
    print()

    object_path = prompt_path("Path to the object you want to process", must_exist=True)
    evaluator_path = prompt_path("Path to the evaluator (script or module)", must_exist=True)

    use_existing_config = yes_no("Do you want to start from an existing config?", default=True)
    base_config: Dict[str, Any] | None = None
    if use_existing_config:
        config_path = prompt_path("Path to existing config file", must_exist=True)
        try:
            base_config = load_config(config_path)
            print("Loaded existing config; we'll keep a backup untouched.")
        except Exception as exc:  # noqa: BLE001 - we want to show friendly failure
            print(f"✖ Could not read config: {exc}")
            return

        default_save = config_path
        if yes_no("Save edits to a new file so the original stays pristine?", default=True):
            default_save = config_path.with_name(f"{config_path.stem}_edited{config_path.suffix}")
        save_path = prompt_path(
            "Where should we save the updated config?", create_parents=True, default=str(default_save)
        )
    else:
        base_config = {}
        save_path = prompt_path(
            "Path to save the new config (e.g., configs/generated_config.yaml)",
            must_exist=False,
            create_parents=True,
        )

    allow_edit = yes_no("Would you like to fill in each parameter (diameter-by-diameter)?", default=True)

    config_payload = build_config_payload(object_path, evaluator_path, base_config, allow_edit)

    save_config(config_payload, save_path)
    print(f"\n💾 Saved config to: {save_path}")
    print("All paths have been expanded to absolute locations.")

    print("\nReady for launch! Suggested next steps:")
    print("  1) conda activate codeevolve")
    print(f"  2) Point your run command to: {save_path}")
    print("     (edit the file manually later if you want more tweaks)")
    print("\nThanks for using the launcher—happy experimenting!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Bye!")
