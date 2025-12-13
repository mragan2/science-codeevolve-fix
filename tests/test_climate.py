# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for climate-based thermal resilience helpers."""

import random

from codeevolve.climate import (
    ClimateConfig,
    assign_thermal_traits,
    evaluate_heat_resilience,
    season_profile,
)


def _wrap_single_call(fn_name: str) -> str:
    return f"def foo(xs):\n    return {fn_name}(xs)\n"


def test_season_rotation_and_climate_flag():
    cfg = ClimateConfig(enabled=True, seasons=["summer", "winter", "spring"], season_length=2)

    summer = season_profile(epoch=1, config=cfg)
    winter = season_profile(epoch=3, config=cfg)

    assert summer.name == "summer"
    assert summer.climate == "hot"
    assert winter.name == "winter"
    assert winter.climate == "cold"


def test_assign_thermal_traits_is_deterministic_with_seed():
    cfg = ClimateConfig(
        enabled=True,
        seasons=["dry"],
        function_pool=["len", "sum", "min"],
        hot_fraction=0.34,
        seed=123,
    )
    season = season_profile(epoch=1, config=cfg)

    hot_a, cold_a = assign_thermal_traits(season, cfg, random.Random(0))
    hot_b, cold_b = assign_thermal_traits(season, cfg, random.Random(5))

    assert hot_a == hot_b
    assert cold_a == cold_b
    assert len(hot_a) == 1  # max(1, hot_fraction * pool_size)


def test_heat_resilience_rewards_alignment_per_season():
    cfg = ClimateConfig(
        enabled=True,
        seasons=["hot", "cold"],
        season_length=1,
        function_pool=["len", "sum", "min", "max"],
        hot_fraction=0.5,
        survival_weight=0.5,
        neutral_baseline=0.5,
        seed=99,
    )

    hot_traits, cold_traits = assign_thermal_traits(
        season_profile(epoch=1, config=cfg), cfg, random.Random(0)
    )
    hot_favored = next(iter(hot_traits))
    hot_eval = evaluate_heat_resilience(
        code=_wrap_single_call(hot_favored), epoch=1, config=cfg, random_state=random.Random(1)
    )

    assert hot_eval.survival_chance > cfg.neutral_baseline
    assert hot_eval.fitness_multiplier > 1

    cold_traits_epoch2 = assign_thermal_traits(
        season_profile(epoch=2, config=cfg), cfg, random.Random(0)
    )[1]
    cold_favored = next(iter(cold_traits_epoch2))
    cold_eval = evaluate_heat_resilience(
        code=_wrap_single_call(cold_favored), epoch=2, config=cfg, random_state=random.Random(1)
    )

    assert cold_eval.survival_chance > cfg.neutral_baseline
    assert cold_eval.fitness_multiplier > 1

