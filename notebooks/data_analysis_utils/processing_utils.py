# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements functions for processing experiment data from CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import os
import sys
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import itertools
from importlib import __import__
from collections import defaultdict
import pickle as pkl

import networkx as nx
import pandas as pd
import yaml

from codeevolve.utils.ckpt_utils import load_ckpt
from codeevolve.database import ProgramDatabase


def get_total_runtime(
    log_file_path: str, start_marker: str = "=== EVOLVE ALGORITHM ==="
) -> timedelta:
    """
    Calculates the total runtime by parsing a log file.

    This function reads a log file and computes the cumulative duration of all
    evolutionary segments. It identifies the start of each segment using a specific
    marker line and measures the time until the next marker or the end of the file,
    summing these durations to get the total runtime.

    Args:
        log_file_path: The path to the log file to be parsed.
        start_marker: A string that identifies the beginning of a timed segment
                    within the log file.

    Returns:
        A timedelta object representing the total calculated runtime.
    """

    def _parse_time(line: str) -> Optional[datetime]:
        """Extracts datetime from a log line.

        Args:
            line: Log line containing timestamp in the expected format.

        Returns:
            Parsed datetime object or None if parsing fails.
        """
        try:
            timestamp_str = line[11:].split(" | ", 1)[0]
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        except (ValueError, IndexError):
            return None

    with open(log_file_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return timedelta(0)

    start_indices = [i for i, line in enumerate(lines) if start_marker in line]
    if not start_indices:
        return timedelta(0)

    total_duration = timedelta(0)
    for i, start_index in enumerate(start_indices):
        start_time = _parse_time(lines[start_index])
        if not start_time:
            continue
        if i + 1 < len(start_indices):
            end_index = start_indices[i + 1] - 1
        else:
            end_index = len(lines) - 1
        end_time = _parse_time(lines[end_index])
        if end_time and end_time >= start_time:
            segment_duration = end_time - start_time
            total_duration += segment_duration
    return total_duration


def get_experiment_df(
    experiment_res: Dict[str, Any],
    results_dir,
    config: Dict[Any, Any],
    model_names: List[str],
) -> pd.DataFrame:
    """
    Compiles detailed experiment results into a Pandas DataFrame.

    This function processes the results from a multi-island experiment, iterating
    through each island to extract key performance and resource metrics. It gathers
    data such as training time, best fitness scores, error counts, and token usage
    for both the main program evolution and the meta-prompting process, then
    organizes this information into a structured DataFrame for analysis.

    Args:
        experiment_res: A dictionary containing the results for each island, keyed by island ID.
        results_dir: The base directory path where island-specific log files are stored.
        config: The experiment's configuration dictionary.
        model_names: A list of model names used in the experiment to correctly structure
                    the token usage columns in the DataFrame.

    Returns:
        A Pandas DataFrame summarizing the key metrics for each island in the experiment.
    """
    df_rows = []

    for isl_id in experiment_res.keys():
        island_results_dir = results_dir + f"{isl_id}/"

        training_time = get_total_runtime(island_results_dir + "results.log")
        training_time = training_time.seconds / 3600

        evolve_state = experiment_res[isl_id]["evolve_state"]
        num_epochs = experiment_res[isl_id]["num_epochs"]

        sol_db = experiment_res[isl_id]["sol_db"]
        best_fitness = sol_db.programs[sol_db.best_prog_id].fitness

        epoch_best_found = float("inf")
        for prog in sol_db.programs.values():
            if prog.fitness == best_fitness:
                epoch_best_found = (
                    min(prog.iteration_found, epoch_best_found) if prog.iteration_found else 0
                )

        num_sr_errors = len(
            [
                error_info
                for error_info in evolve_state["errors"]
                if error_info["motive"] == "sr_evolve_prog"
            ]
        )
        num_eval_errors = len([prog for prog in sol_db.programs.values() if prog.error])
        num_eval_warnings = len([prog for prog in sol_db.programs.values() if prog.warning])

        model2tok_usage = defaultdict(lambda: defaultdict(int))
        for tok_info in evolve_state["tok_usage"]:
            if tok_info["motive"] == "generate_prog":
                model2tok_usage[tok_info["model_name"]]["prompt_tok"] += tok_info["prompt_tok"]
                model2tok_usage[tok_info["model_name"]]["compl_tok"] += tok_info["compl_tok"]

        tok_usage_row = []
        for model_name in model_names:
            tok_usage_row += [
                model2tok_usage[model_name]["prompt_tok"],
                model2tok_usage[model_name]["compl_tok"],
            ]

        prompt_db = experiment_res[isl_id]["prompt_db"]
        mp_best_fitness = prompt_db.programs[prompt_db.best_prog_id].fitness

        mp_epoch_best_found = float("inf")
        for prog in prompt_db.programs.values():
            if prog.fitness == mp_best_fitness:
                mp_epoch_best_found = (
                    min(prog.iteration_found, mp_epoch_best_found) if prog.iteration_found else 0
                )

        mp_num_sr_errors = len(
            [
                error_info
                for error_info in evolve_state["errors"]
                if error_info["motive"] == "sr_meta_prompt"
            ]
        )
        mp_num_eval_errors = len([prog for prog in prompt_db.programs.values() if prog.error])
        mp_num_eval_warnings = len([prog for prog in prompt_db.programs.values() if prog.warning])

        mp_model2tok_usage = defaultdict(lambda: defaultdict(int))
        for tok_info in evolve_state["tok_usage"]:
            if tok_info["motive"] == "meta_prompt":
                mp_model2tok_usage[tok_info["model_name"]]["prompt_tok"] += tok_info["prompt_tok"]
                mp_model2tok_usage[tok_info["model_name"]]["compl_tok"] += tok_info["compl_tok"]

        mp_tok_usage_row = []
        for model_name in model_names:
            mp_tok_usage_row += [
                mp_model2tok_usage[model_name]["prompt_tok"],
                mp_model2tok_usage[model_name]["compl_tok"],
            ]

        df_rows.append(
            [
                isl_id,
                num_epochs,
                training_time,
                best_fitness,
                epoch_best_found,
                num_sr_errors,
                num_eval_errors,
                num_eval_warnings,
            ]
            + [
                mp_best_fitness,
                mp_epoch_best_found,
                mp_num_sr_errors,
                mp_num_eval_errors,
                mp_num_eval_warnings,
            ]
            + tok_usage_row
            + mp_tok_usage_row
        )

    tok_usage_cols = [
        model_name + suffix
        for model_name, suffix in itertools.product(model_names, ["(prompt_tok)", "(compl_tok)"])
    ]
    mp_tok_usage_cols = [
        "mp_" + model_name + suffix
        for model_name, suffix in itertools.product(model_names, ["(prompt_tok)", "(compl_tok)"])
    ]

    return pd.DataFrame(
        df_rows,
        columns=[
            "isl_id",
            "num_epochs",
            "training_time (hours)",
            "best_fitness",
            "epoch_best_found",
            "num_sr_errors",
            "num_eval_errors",
            "num_eval_warnings",
        ]
        + [
            "mp_best_fitness",
            "mp_epoch_best_found",
            "mp_num_sr_errors",
            "mp_num_eval_errors",
            "mp_num_eval_warnings",
        ]
        + tok_usage_cols
        + mp_tok_usage_cols,
    )


def process_experiments(
    args: Dict, model_names: List[str], model2cost: Dict[str, Dict[str, float]]
):
    """Loads, processes, and summarizes results from multiple experiment directories.

    This function serves as a primary entry point for analyzing a batch of experiments.
    It iterates through a list of specified experiment output directories, loads the
    latest checkpoint data for each island, generates a summary DataFrame of metrics,
    and calculates the estimated LLM cost based on token usage. All processed data
    is aggregated into a final dictionary, keyed by the experiment path.

    Args:
        args: A dictionary of arguments, including the input directory (`inpt_dir`)
            and a list of experiment output directories (`out_dirs`).
        model_names: A list of model names used, required for cost calculation and
                    DataFrame generation.
        model2cost: A dictionary mapping model names to their respective prompt and
                    completion token costs.

    Returns:
        A dictionary where keys are experiment paths and values are objects containing
        the raw results, configuration, summary DataFrame, and estimated cost.
    """
    experiments_res = {}
    for idx, result_dir in enumerate(args["out_dirs"]):

        cfg_fname = [fname for fname in os.listdir(result_dir) if fname.endswith(".yaml")][0]
        with open(result_dir + cfg_fname, "r") as f:
            config = yaml.safe_load(f)

        experiment_res: Dict[int, Dict[str, Any]] = {}

        ckpt = -1 if not args.get("ckpts", None) else args["ckpts"][idx]

        for isl_id in range(0, config["EVOLVE_CONFIG"]["num_islands"]):
            island_results_dir: str = result_dir + f"{isl_id}/"
            ckpt_dir: str = island_results_dir + "ckpt/"
            try:
                prompt_db, sol_db, evolve_state = load_ckpt(ckpt, ckpt_dir)
            except:
                ckpts: List[str] = [
                    f for f in os.listdir(ckpt_dir) if re.match(r"ckpt_\d+\.pkl$", f)
                ]
                if len(ckpts):
                    ckpt = max([int(re.search(r"ckpt_(\d+)\.pkl$", f).group(1)) for f in ckpts])
                    prompt_db, sol_db, evolve_state = load_ckpt(ckpt, ckpt_dir)
                else:
                    raise ValueError(f"No ckpts were found for island {isl_id}.")

            experiment_res[isl_id] = {
                "prompt_db": prompt_db,
                "sol_db": sol_db,
                "evolve_state": evolve_state,
                "num_epochs": ckpt,
            }

        experiment_df = get_experiment_df(experiment_res, result_dir, config, model_names)

        tok_cols = experiment_df.columns[-8:]
        estimated_cost = 0
        for col in tok_cols:
            mult = "prompt_pm" if "prompt" in col else "compl_pm"
            for model_name in model_names:
                if model_name in col:
                    estimated_cost += (
                        (experiment_df[[col]] * model2cost[model_name][mult] * 1e-6).sum().sum()
                    )

        best_island = experiment_df["best_fitness"].idxmax()

        experiments_res[result_dir] = {
            "res": experiment_res,
            "config": config,
            "df": experiment_df,
            "cost": estimated_cost,
        }

    return experiments_res


def get_experiment_sol(results_dir, sol_func_name, island_id: int):
    """
    Loads and returns the final solution object from an experiment's best program file.

    This function dynamically imports the `best_sol.py` file generated by a specific
    island during an experiment. It then executes a specified function within that
    module to instantiate and retrieve the final solution object. For caching, it
    also saves the retrieved solution to a `.pkl` file.

    Args:
        results_dir: The base directory of the experiment where island results are stored.
        sol_func_name: The name of the function inside `best_sol.py` to call to get the solution.
        island_id: The ID of the island from which to load the best solution.

    Returns:
        The solution object returned by the function from the loaded module.
    """
    program_path = results_dir + f"{island_id}/best_sol.py"
    if "best_sol" in sys.modules:
        del sys.modules["best_sol"]

    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        sol = getattr(program, sol_func_name)()
        pkl.dump(sol, open(results_dir + "best_sol.pkl", "wb"))
    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    return sol


def create_db_tree(db: ProgramDatabase) -> nx.DiGraph:
    """
    Constructs an evolutionary lineage tree from a ProgramDatabase.

    This function converts a ProgramDatabase object into a `networkx.DiGraph`,
    visualizing the evolutionary history of the programs. Each program becomes a node,
    and directed edges are drawn from parent programs to their offspring, creating a
    tree or forest structure. The graph is annotated with the ID of the best program
    and a list of root nodes.

    Args:
        db: The ProgramDatabase instance containing the full history of all programs.

    Returns:
        A networkx.DiGraph object representing the program lineage.
    """
    G = nx.DiGraph()
    for prog_id, prog in db.programs.items():
        G.add_node(prog_id)

    G.graph["best_prog_id"] = db.best_prog_id

    G.graph["roots"] = []
    for prog_id, prog in db.programs.items():
        if prog.parent_id:
            G.add_edge(prog.parent_id, prog_id)
        else:
            G.graph["roots"].append(prog_id)

    return G
