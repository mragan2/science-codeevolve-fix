import asyncio
import time

from codeevolve.database import Program
from codeevolve.evaluator import Evaluator


def test_evaluate_batch_runs_programs_in_parallel():
    # Use a lightweight evaluator and monkeypatch execute to avoid subprocess calls.
    evaluator = Evaluator(
        eval_path="/dev/null", cwd=None, timeout_s=1, max_mem_b=None, mem_check_interval_s=None
    )

    # Create a few dummy programs to evaluate.
    programs = [
        Program(id=f"prog-{idx}", code="", language="python") for idx in range(3)
    ]

    async def run_batch():
        # Simulate work that takes time to help detect parallel execution.
        def fake_execute(prog: Program):
            time.sleep(0.1)
            prog.eval_metrics["finished"] = True

        evaluator.execute = fake_execute  # type: ignore[assignment]

        start = time.perf_counter()
        await evaluator.evaluate_batch(programs, max_workers=2)
        return time.perf_counter() - start

    duration = asyncio.run(run_batch())

    # Two workers processing three ~0.1s tasks should complete in comfortably
    # under 0.3s if execution overlaps.
    assert duration < 0.3
    assert all("finished" in program.eval_metrics for program in programs)
