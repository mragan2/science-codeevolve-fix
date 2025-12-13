# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
# ===--------------------------------------------------------------------------------------===#
#
"""Adversarial multi-population utilities for CodeEvolve."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from codeevolve.database import Program


@dataclass
class AdversarialConfig:
    """Configuration block for adversarial multi-population evolution."""

    enabled: bool = False
    teams: List[str] = field(default_factory=lambda: ["red", "blue"])
    cross_eval_interval: int = 1
    opponents_per_eval: int = 2
    fitness_metric: str = "win_rate"  # supported: "win_rate", "elo", "hybrid"
    base_fitness_weight: float = 0.2
    elo_k: float = 32.0
    initial_rating: float = 1000.0
    alternating_phases: bool = False


@dataclass
class CompetitiveResult:
    """Summary of a cross-population evaluation round."""

    win_rate: float
    matches: int
    rating: float
    fitness: float


def assign_team(island_id: int, teams: List[str]) -> str:
    """Assigns an island to a team based on its index."""

    if not teams:
        return "default"
    return teams[island_id % len(teams)]


def should_cross_evaluate(epoch: int, team: str, config: AdversarialConfig) -> bool:
    """Determines whether to run a cross-population evaluation this epoch."""

    if not config.enabled:
        return False

    interval = max(1, config.cross_eval_interval)
    if epoch % interval != 0:
        return False

    if not config.alternating_phases:
        return True

    team_index: int = config.teams.index(team)
    return (epoch // interval) % len(config.teams) == team_index


def update_team_registry(
    registry: Optional[Dict[str, Program]], team: str, candidate: Program
) -> None:
    """Updates the shared registry with the best program for a team."""

    if registry is None:
        return

    best_prog: Optional[Program] = registry.get(team, None)
    if best_prog is None or candidate.fitness > best_prog.fitness:
        registry[team] = candidate


def sample_opponents(
    registry: Optional[Dict[str, Program]],
    team: str,
    teams: List[str],
    max_opponents: int,
    random_state,
) -> List[Program]:
    """Samples opponents from rival teams registered in the shared pool."""

    if registry is None:
        return []

    rival_programs: List[Program] = []
    for rival_team in teams:
        if rival_team == team:
            continue
        opponent: Optional[Program] = registry.get(rival_team, None)
        if opponent is not None:
            rival_programs.append(opponent)

    random_state.shuffle(rival_programs)
    return rival_programs[:max_opponents]


def _pair_score(candidate_score: float, opponent_score: float) -> float:
    """Returns the outcome score for Elo: 1 win, 0.5 draw, 0 loss."""

    if candidate_score > opponent_score:
        return 1.0
    if candidate_score < opponent_score:
        return 0.0
    return 0.5


def _elo_update(rating: float, opponent_rating: float, score: float, k: float) -> float:
    """Updates an Elo rating given a single match outcome."""

    expected: float = 1.0 / (1 + 10 ** ((opponent_rating - rating) / 400))
    return rating + k * (score - expected)


def compute_competitive_result(
    candidate: Program,
    opponents: List[Program],
    base_fitness_key: str,
    config: AdversarialConfig,
) -> CompetitiveResult:
    """Computes win-rate and Elo-based fitness against a set of opponents."""

    if not opponents:
        return CompetitiveResult(
            win_rate=0.0,
            matches=0,
            rating=candidate.rating,
            fitness=candidate.fitness,
        )

    wins: int = 0
    draws: int = 0
    rating: float = candidate.rating if candidate.rating is not None else config.initial_rating

    candidate_score: float = candidate.eval_metrics.get(base_fitness_key, candidate.fitness)
    for opponent in opponents:
        opponent_score: float = opponent.eval_metrics.get(base_fitness_key, opponent.fitness)
        score: float = _pair_score(candidate_score, opponent_score)
        wins += score == 1.0
        draws += score == 0.5
        opp_rating: float = opponent.rating if opponent.rating is not None else config.initial_rating
        rating = _elo_update(rating, opp_rating, score, config.elo_k)

    matches: int = len(opponents)
    win_rate: float = (wins + 0.5 * draws) / matches

    if config.fitness_metric == "elo":
        fitness: float = rating
    elif config.fitness_metric == "hybrid":
        fitness = config.base_fitness_weight * candidate_score + (1 - config.base_fitness_weight) * win_rate
    else:  # default to pure win rate
        fitness = win_rate

    return CompetitiveResult(
        win_rate=win_rate,
        matches=matches,
        rating=rating,
        fitness=fitness,
    )
