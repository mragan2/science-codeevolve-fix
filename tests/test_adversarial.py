# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#

"""Unit tests for adversarial multi-population helpers."""

import random

from codeevolve.adversarial import (
    AdversarialConfig,
    assign_team,
    compute_competitive_result,
    sample_opponents,
    should_cross_evaluate,
)
from codeevolve.database import Program


def _mk_prog(pid: str, fitness: float, rating: float = 1000.0) -> Program:
    prog = Program(id=pid, code="print('hi')", language="python")
    prog.fitness = fitness
    prog.rating = rating
    prog.eval_metrics = {"score": fitness}
    return prog


def test_assign_team_round_robin():
    cfg = AdversarialConfig(enabled=True, teams=["red", "blue", "green"])
    assert assign_team(0, cfg.teams) == "red"
    assert assign_team(1, cfg.teams) == "blue"
    assert assign_team(4, cfg.teams) == "blue"


def test_sample_opponents_prefers_rivals():
    registry = {
        "red": _mk_prog("r1", 0.1),
        "blue": _mk_prog("b1", 0.5),
    }
    opponents = sample_opponents(
        registry,
        team="red",
        teams=["red", "blue"],
        max_opponents=2,
        random_state=random.Random(0),
    )
    assert len(opponents) == 1
    assert opponents[0].id == "b1"


def test_competitive_result_win_rate_and_elo():
    cfg = AdversarialConfig(enabled=True, fitness_metric="hybrid", base_fitness_weight=0.5, elo_k=16)
    candidate = _mk_prog("c", fitness=0.8)
    opponents = [_mk_prog("o1", fitness=0.4), _mk_prog("o2", fitness=0.4)]

    result = compute_competitive_result(candidate, opponents, base_fitness_key="score", config=cfg)

    assert result.matches == 2
    assert result.win_rate == 1.0
    assert result.rating > candidate.rating
    # hybrid fitness blends base fitness and win rate
    assert result.fitness > candidate.fitness


def test_should_cross_evaluate_with_interval_and_alternation():
    cfg = AdversarialConfig(enabled=True, cross_eval_interval=2, alternating_phases=True, teams=["red", "blue"])
    assert should_cross_evaluate(epoch=2, team="red", config=cfg) is False
    assert should_cross_evaluate(epoch=2, team="blue", config=cfg) is True
    cfg.alternating_phases = False
    assert should_cross_evaluate(epoch=4, team="red", config=cfg) is True

