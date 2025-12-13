# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements optional agent utilities for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from codeevolve.database import Program
from codeevolve.lm import OpenAILM
from codeevolve.prompt.sampler import format_prog_msg
from codeevolve.prompt.template import (
    NOVEL_AGENT_SYSTEM_PROMPT,
    NOVEL_AGENT_USER_TEMPLATE,
)


@dataclass
class NovelAgent:
    """LLM-based agent focused on injecting novelty into prompt evolution.

    The agent is designed to occasionally replace the standard meta-prompting
    step with a more exploratory proposal that intentionally searches for new
    algorithmic directions. It still returns a SEARCH/REPLACE diff compatible
    with the existing ``apply_diff_with_fallback`` utility, so it can be
    slotted directly into the current evolution loop without changing the
    downstream mechanics.

    Attributes:
        lm: Configured language model used to author the novel prompt diff.
        exploration_rate: Probability of invoking the agent when exploration is
            enabled for the epoch.
        max_inspirations: Maximum number of inspiration programs to include in
            the generated context.
        logger: Logger instance used for tracing agent activity.
    """

    lm: OpenAILM
    exploration_rate: float = 0.2
    max_inspirations: int = 2
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def should_activate(self, random_state) -> bool:
        """Determines whether the agent should run in the current epoch."""

        return random_state.uniform(0, 1) <= self.exploration_rate

    def _format_inspirations(self, inspirations: Optional[List[Program]]) -> str:
        """Formats inspiration programs into a readable block for the prompt."""

        if not inspirations:
            return "No inspiration programs supplied."

        insp_blocks: List[str] = []
        for idx, inspiration in enumerate(inspirations[: self.max_inspirations]):
            prog_msg: str = inspiration.prog_msg
            if prog_msg is None:
                prog_msg = format_prog_msg(prog=inspiration)
            insp_blocks.append(f"----------INSPIRATION {idx + 1}----------\n{prog_msg}")

        return "\n".join(insp_blocks)

    async def propose_prompt(
        self, prompt: Program, prog: Program, inspirations: Optional[List[Program]]
    ) -> Tuple[str, int, int]:
        """Generates a novel prompt diff emphasizing exploration."""

        prog_msg: str = prog.prog_msg
        if prog_msg is None:
            prog_msg = format_prog_msg(prog=prog)

        content: str = NOVEL_AGENT_USER_TEMPLATE.format(
            prompt=prompt.code,
            program=prog_msg,
            inspirations=self._format_inspirations(inspirations),
        )

        messages = [
            {"role": "system", "content": NOVEL_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        self.logger.info(
            "Attempting to run novel prompt proposal using %s...", self.lm.model_name
        )

        response, prompt_tok, compl_tok = await self.lm.generate(messages)

        self.logger.info(
            (
                "Novel agent response received (%s prompt tok, %s completion tok)."
            ),
            prompt_tok,
            compl_tok,
        )

        return response, prompt_tok, compl_tok
