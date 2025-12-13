# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
# ===--------------------------------------------------------------------------------------===#
#
"""Seasonal climate utilities for thermal resilience scoring."""

import ast
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

DEFAULT_FUNCTION_POOL: List[str] = [
    "len",
    "sum",
    "min",
    "max",
    "sorted",
    "enumerate",
    "zip",
    "range",
    "map",
    "filter",
]


@dataclass
class SeasonProfile:
    """Represents the active climate season for an epoch."""

    name: str
    climate: str  # "hot" or "cold"
    index: int


@dataclass
class ClimateConfig:
    """Configuration block for climate-based fitness adjustments."""

    enabled: bool = False
    seasons: List[str] = field(default_factory=lambda: ["perpetual"])
    season_length: int = 5
    function_pool: List[str] = field(default_factory=lambda: list(DEFAULT_FUNCTION_POOL))
    hot_fraction: float = 0.5
    survival_weight: float = 0.2
    neutral_baseline: float = 0.5
    seed: int | None = None


@dataclass
class ThermalEvaluation:
    """Computed thermal resilience statistics for a program."""

    season: SeasonProfile
    hot_traits: Set[str]
    cold_traits: Set[str]
    hot_hits: int
    cold_hits: int
    total_hits: int
    alignment: float
    survival_chance: float
    fitness_multiplier: float


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def season_profile(epoch: int, config: ClimateConfig) -> SeasonProfile:
    """Returns the active season and climate for the given epoch."""

    season_span: int = max(1, config.season_length)
    season_idx: int = ((max(epoch, 1) - 1) // season_span) % max(1, len(config.seasons))
    climate: str = "hot" if season_idx % 2 == 0 else "cold"
    return SeasonProfile(name=config.seasons[season_idx], climate=climate, index=season_idx)


def assign_thermal_traits(
    season: SeasonProfile, config: ClimateConfig, random_state: random.Random
) -> Tuple[Set[str], Set[str]]:
    """Assigns functions to hot or cold traits for the active season."""

    trait_rng = random.Random()
    seed_base = config.seed
    if seed_base is None:
        seed_base = random_state.randint(0, 10_000_000)
    trait_rng.seed(seed_base + season.index)

    pool: List[str] = list(dict.fromkeys(config.function_pool))
    trait_rng.shuffle(pool)
    hot_cutoff: int = max(1, int(len(pool) * _clamp(config.hot_fraction)))
    hot_traits: Set[str] = set(pool[:hot_cutoff])
    cold_traits: Set[str] = set(pool[hot_cutoff:])
    return hot_traits, cold_traits


def _count_call_names(code: str, pool: Set[str]) -> Dict[str, int]:
    """Counts simple function calls in code that match the pool."""

    counts: Dict[str, int] = {name: 0 for name in pool}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return counts

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr

            if name in counts:
                counts[name] += 1
    return counts


def evaluate_heat_resilience(
    code: str, epoch: int, config: ClimateConfig, random_state: random.Random
) -> ThermalEvaluation:
    """Evaluates how well a program aligns with the current climate season."""

    season = season_profile(epoch, config)
    hot_traits, cold_traits = assign_thermal_traits(season, config, random_state)

    pool: Set[str] = set(config.function_pool)
    counts: Dict[str, int] = _count_call_names(code=code, pool=pool)

    hot_hits: int = sum(counts[name] for name in hot_traits)
    cold_hits: int = sum(counts[name] for name in cold_traits)
    total_hits: int = hot_hits + cold_hits

    if total_hits == 0:
        alignment = config.neutral_baseline
    elif season.climate == "cold":
        alignment = cold_hits / total_hits
    else:
        alignment = hot_hits / total_hits

    alignment = _clamp(alignment)
    survival_chance: float = alignment if total_hits > 0 else config.neutral_baseline
    fitness_multiplier: float = 1 + config.survival_weight * (survival_chance - config.neutral_baseline)

    return ThermalEvaluation(
        season=season,
        hot_traits=hot_traits,
        cold_traits=cold_traits,
        hot_hits=hot_hits,
        cold_hits=cold_hits,
        total_hits=total_hits,
        alignment=alignment,
        survival_chance=survival_chance,
        fitness_multiplier=fitness_multiplier,
    )

