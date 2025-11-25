# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements functions for plotting experiment results from CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List, Optional

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np


def plot_experiments_statistical_summary(
    experiments: Dict[str, Dict[int, Any]],
    title,
    save_path: str = None,
    figsize: tuple = (6, 4),
    epsilon: float = 1e-3,
):
    """Plots statistical summary of fitness evolution across multiple experiments.

    This function creates a line plot showing the evolution of fitness values over
    epochs for multiple experiments. It uses a logarithmic transformation to better
    visualize convergence towards optimal fitness values and handles NaN values
    and variable-length histories across experiments.

    Args:
        experiments: Dictionary mapping experiment names to their results containing
                    fitness histories for each island/run.
        title: Title for the plot.
        save_path: Optional path to save the plot image. If None, plot is only displayed.
        figsize: Tuple specifying the figure size (width, height) in inches.
        epsilon: Small value used for numerical stability in logarithmic transformation.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    default_colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    default_markers = ["o", "s", "^", "v", "D", "p", "*", "h", "H", "+"]

    all_fitness_values = []
    experiment_data = {}

    for exp_name, results in experiments.items():
        all_best_hists = []

        for key, data in results.items():
            try:
                best_hist = data["evolve_state"]["best_fit_hist"]
                if best_hist:
                    all_best_hists.append(best_hist)
            except (KeyError, TypeError):
                continue

        if not all_best_hists:
            print(f"No valid data found for experiment: {exp_name}")
            continue

        experiment_data[exp_name] = all_best_hists

        for hist in all_best_hists:
            for value in hist:
                if not np.isnan(value):
                    all_fitness_values.append(value)

    if not all_fitness_values:
        print("No valid fitness values found across all experiments!")
        return

    MAX_FITNESS = max(all_fitness_values)

    for exp_idx, (exp_name, all_best_hists) in enumerate(experiment_data.items()):

        def pad_with_last_non_nan(hist):
            """Pad history with last non-NaN value, handling NaNs properly"""
            last_valid = None
            for i in range(len(hist) - 1, -1, -1):
                if not np.isnan(hist[i]):
                    last_valid = hist[i]
                    break
            if last_valid is None:
                return None

            cleaned_hist = []
            current_valid = None

            for value in hist:
                if not np.isnan(value):
                    current_valid = value
                    cleaned_hist.append(value)
                else:
                    cleaned_hist.append(current_valid if current_valid is not None else last_valid)

            return cleaned_hist

        cleaned_hists = []
        for hist in all_best_hists:
            cleaned = pad_with_last_non_nan(hist)
            if cleaned is not None:
                cleaned_hists.append(cleaned)

        if not cleaned_hists:
            print(f"No valid histories after cleaning NaNs for experiment: {exp_name}")
            continue

        max_len = max(len(hist) for hist in cleaned_hists)
        padded_best = []
        for hist in cleaned_hists:
            padded = hist + [hist[-1]] * (max_len - len(hist))
            padded_best.append(padded)

        best_array = np.array(padded_best)
        best_fitness = np.max(best_array, axis=0)
        std_best = np.std(best_array, axis=0)
        epochs = range(1, max_len + 1)

        y_data = np.maximum(MAX_FITNESS + epsilon - best_fitness, epsilon)

        color = default_colors[exp_idx]
        marker = default_markers[exp_idx % len(default_markers)]

        ax.plot(
            epochs,
            -np.log10(y_data),
            color=color,
            linewidth=2,
            marker=marker,
            markersize=4,
            markevery=5,
            label=f"{exp_name}",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")

    ax.set_ylabel("Target metric")

    ax.axhline(
        y=-np.log10(MAX_FITNESS + epsilon - 1),
        linestyle=":",
        color="gray",
        alpha=0.7,
        label=f"AlphaEvolve",
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_experiments_mean_std(
    experiments: Dict[str, List[Dict[int, Any]]],
    title: str,
    save_path: str = None,
    figsize: tuple = (6, 4),
    epsilon: float = 1e-3,
):
    """
    Plots the mean and standard deviation of fitness evolution across multiple experiments.

    This function creates a line plot showing the mean evolution of fitness values
    over epochs for multiple experiments. It also visualizes the standard deviation
    as a shaded area around the mean. It uses a logarithmic transformation to
    better visualize convergence towards optimal fitness values and handles NaN
    values and variable-length histories.

    Args:
        experiments: Dictionary mapping experiment names to a list of runs.
                     Each run is a dictionary containing fitness histories.
        title: Title for the plot.
        save_path: Optional path to save the plot image. If None, plot is only displayed.
        figsize: Tuple specifying the figure size (width, height) in inches.
        epsilon: Small value used for numerical stability in logarithmic transformation.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    default_markers = ["o", "s", "^", "v", "D", "p", "*", "h", "H", "+"]

    all_fitness_values = []
    experiment_data = {}

    for exp_name, runs in experiments.items():
        all_best_hists = []
        for single_run_results in runs:
            for key, data in single_run_results.items():
                try:
                    best_hist = data["evolve_state"]["best_fit_hist"]
                    if best_hist:
                        all_best_hists.append(best_hist)
                except (KeyError, TypeError):
                    continue

        if not all_best_hists:
            print(f"No valid data found for experiment: {exp_name}")
            continue

        experiment_data[exp_name] = all_best_hists
        for hist in all_best_hists:
            for value in hist:
                if not np.isnan(value):
                    all_fitness_values.append(value)

    if not all_fitness_values:
        print("No valid fitness values found across all experiments!")
        return

    MAX_FITNESS = max(all_fitness_values)

    for exp_idx, (exp_name, all_best_hists) in enumerate(experiment_data.items()):

        def pad_with_last_non_nan(hist):
            last_valid = None
            for i in range(len(hist) - 1, -1, -1):
                if not np.isnan(hist[i]):
                    last_valid = hist[i]
                    break
            if last_valid is None:
                return None
            cleaned_hist = []
            current_valid = None
            for value in hist:
                if not np.isnan(value):
                    current_valid = value
                    cleaned_hist.append(value)
                else:
                    cleaned_hist.append(current_valid if current_valid is not None else last_valid)
            return cleaned_hist

        cleaned_hists = [
            hist for hist in (pad_with_last_non_nan(h) for h in all_best_hists) if hist is not None
        ]

        if not cleaned_hists:
            print(f"No valid histories after cleaning NaNs for experiment: {exp_name}")
            continue

        max_len = max(len(hist) for hist in cleaned_hists)
        padded_best = [hist + [hist[-1]] * (max_len - len(hist)) for hist in cleaned_hists]
        best_array = np.array(padded_best)

        epochs = range(1, max_len + 1)
        color = default_colors[exp_idx]
        marker = default_markers[exp_idx % len(default_markers)]

        transformed_array = -np.log10(np.maximum(MAX_FITNESS + epsilon - best_array, epsilon))

        mean_transformed = np.mean(transformed_array, axis=0)
        std_transformed = np.std(transformed_array, axis=0)

        upper_bound = mean_transformed + std_transformed
        lower_bound = mean_transformed - std_transformed

        ax.plot(
            epochs,
            mean_transformed,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=4,
            markevery=5,
            label=f"{exp_name}",
        )

        ax.fill_between(
            epochs,
            lower_bound,
            upper_bound,
            color=color,
            alpha=0.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Target Metric")
    ax.axhline(
        y=-np.log10(MAX_FITNESS + epsilon - 1),
        linestyle=":",
        color="gray",
        alpha=0.7,
        label=f"AlphaEvolve",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_program_tree(
    G: nx.DiGraph,
    node_labels: Dict,
    color_name: str,
    node_colors: Dict,
    num_islands: int,
    node_to_island: Dict,
    save_path: str = None,
    figsize=(12, 10),
    node_size=160,
    font_size=8,
    title=None,
) -> None:
    """
    Visualizes the evolutionary tree of programs from a CodeEvolve.
    This function generates a hierarchical plot of the program evolution graph, where
    each node represents a program and edges show parent-child relationships.
    Nodes are colored based on a metric (like fitness), explained by a color bar.
    Different marker shapes distinguish programs from different islands, and the
    overall best program is highlighted with a larger size and a distinct border.
    Args:
        G: The networkx.DiGraph representing the program lineage.
        node_labels: Dictionary mapping node IDs to their string representation for display.
        color_name: Label for the color bar, describing the metric used for node colors.
        node_colors: Dictionary mapping node IDs to a numerical value for coloring.
        num_islands: The total number of islands used in the experiment.
        node_to_island: Dictionary mapping each program's node ID to its source island.
        save_path: Optional path to save the plot image. If None, the plot is only displayed.
        figsize: Tuple specifying the figure size (width, height) in inches.
        node_size: Base size for the plotted nodes.
        font_size: Font size for node labels and the legend.
        title: Optional title for the plot.
    """

    fig, ax = plt.subplots(figsize=figsize)

    pos = graphviz_layout(G, prog="dot")

    values = list(node_colors.values())
    vmin = min(values)
    vmax = max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    node_colors_cm = plt.get_cmap("autumn")(norm(np.array(values, dtype=float)))

    island_shapes = ["p", "8", "s", "o", "h"]

    best_id = G.graph["best_prog_id"]

    island_nodes = {}
    for node in G.nodes():
        if node != best_id:
            island_id = node_to_island.get(node, 0)
            if island_id not in island_nodes:
                island_nodes[island_id] = []
            island_nodes[island_id].append(node)

    legend_handles = []

    for island_id, nodes in island_nodes.items():
        if not nodes:
            continue

        shape = island_shapes[island_id % len(island_shapes)]

        island_node_colors = [node_colors_cm[list(G.nodes()).index(node)] for node in nodes]

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=nodes,
            node_color=island_node_colors,
            node_size=node_size,
            node_shape=shape,
            edgecolors="black",
            linewidths=1,
            alpha=0.8,
        )

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=shape,
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label=f"Island {island_id}",
            )
        )

    best_island_id = node_to_island.get(best_id, 0)
    best_shape = island_shapes[best_island_id % len(island_shapes)]
    best_node_color = node_colors_cm[list(G.nodes()).index(best_id)]

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[best_id],
        node_color=[best_node_color],
        node_size=node_size * 1.5,
        node_shape=best_shape,
        alpha=0.9,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[best_id],
        node_color="none",
        node_size=node_size * 1.5,
        node_shape=best_shape,
        edgecolors="blue",
        linewidths=2,
        alpha=1.0,
    )

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=8, edge_color="gray", alpha=0.8)

    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=font_size)

    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker=best_shape,
            color="w",
            markerfacecolor="white",
            markeredgecolor="blue",
            markersize=10,
            markeredgewidth=2,
            label=f"Best solution",
        )
    )

    legend_handles.sort(key=lambda x: x.get_label())
    ax.legend(handles=legend_handles, loc="lower right", fontsize=font_size)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("autumn"), norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar_label = color_name
    cbar.set_label(cbar_label)

    if title:
        ax.set_title(title, fontweight="bold")
    else:
        ax.set_title(f"Solution evolution forest of best island ({num_islands} islands)")

    plt.tight_layout()
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
    plt.show()
