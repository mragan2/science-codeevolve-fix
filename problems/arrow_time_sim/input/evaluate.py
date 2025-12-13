import importlib.util
import math
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List


def _safe_import_module(py_path: Path):
    spec = importlib.util.spec_from_file_location("candidate", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nie mogę załadować modułu z pliku: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _run_candidate(mod) -> List[Dict[str, float]]:
    if not hasattr(mod, "run"):
        raise AttributeError("Brak funkcji `run(steps, dt)` w initial_program.py")
    traj = mod.run(steps=60, dt=0.1)
    if not isinstance(traj, list) or len(traj) < 2:
        raise ValueError("`run()` musi zwrócić listę (>=2) stanów.")
    return traj


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def evaluate(candidate_file: str) -> Dict[str, Any]:
    """
    Zwraca słownik metryk. Klucz docelowy: `combined_score`.

    Uwaga: stdout/stderr kandydata są wyciszone dla stabilności i szybkości.
    """
    path = Path(candidate_file)
    if not path.exists():
        return {
            "combined_score": 0.0,
            "error": f"Brak pliku kandydata: {candidate_file}",
        }

    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            mod = _safe_import_module(path)
            traj = _run_candidate(mod)

        # Oczekiwane pola: t oraz S (reszta może być dowolna).
        t_vals = [float(p.get("t", float("nan"))) for p in traj]
        S_vals = [float(p.get("S", float("nan"))) for p in traj]

        if any(not math.isfinite(x) for x in t_vals) or any(not math.isfinite(x) for x in S_vals):
            raise ValueError("Trajektoria zawiera NaN/Inf w t lub S.")

        # --- Metryki: strzałka czasu ---
        dt_vals = [t_vals[i] - t_vals[i - 1] for i in range(1, len(t_vals))]
        dS_vals = [S_vals[i] - S_vals[i - 1] for i in range(1, len(S_vals))]

        # 1) „czas do przodu” (penalizuj kroki wstecz / zerowe)
        neg_dt = sum(1 for d in dt_vals if d < 0)
        zero_dt = sum(1 for d in dt_vals if abs(d) < 1e-12)
        time_forward_score = _clamp01(1.0 - (neg_dt + 0.25 * zero_dt) / max(1, len(dt_vals)))

        # 2) monotoniczność entropii przy rosnącym czasie
        # (jeśli dt > 0, oczekujemy dS >= 0)
        bad_entropy = 0
        checked = 0
        for dti, dSi in zip(dt_vals, dS_vals):
            if dti > 0:
                checked += 1
                if dSi < -1e-12:
                    bad_entropy += 1
        entropy_monotone_score = _clamp01(1.0 - bad_entropy / max(1, checked))

        # 3) entropia nieujemna
        min_S = min(S_vals)
        entropy_nonneg_score = 1.0 if min_S >= -1e-12 else _clamp01(1.0 / (1.0 + abs(min_S)))

        # 4) ograniczenie „wybuchu” wartości (prosty stabilizator)
        max_abs_t = max(abs(x) for x in t_vals)
        max_abs_S = max(abs(x) for x in S_vals)
        boundedness_score = _clamp01(1.0 / (1.0 + 0.05 * (max_abs_t + max_abs_S)))

        # 5) gładkość (mniejsze „szarpanie” = lepiej)
        # używamy średniej z |drugiej różnicy| dla S
        ddS = []
        for i in range(2, len(S_vals)):
            ddS.append(S_vals[i] - 2 * S_vals[i - 1] + S_vals[i - 2])
        smoothness = sum(abs(x) for x in ddS) / max(1, len(ddS))
        smoothness_score = _clamp01(1.0 / (1.0 + 5.0 * smoothness))

        # --- Wynik łączny ---
        combined_score = (
            0.30 * time_forward_score
            + 0.35 * entropy_monotone_score
            + 0.15 * entropy_nonneg_score
            + 0.10 * boundedness_score
            + 0.10 * smoothness_score
        )

        # `feat1` jest celowo wystawione, żeby Twoje MAP_ELITES z configu działało bez zmian.
        return {
            "combined_score": float(combined_score),
            "feat1": float(entropy_monotone_score),
            "time_forward_score": float(time_forward_score),
            "entropy_monotone_score": float(entropy_monotone_score),
            "entropy_nonneg_score": float(entropy_nonneg_score),
            "boundedness_score": float(boundedness_score),
            "smoothness_score": float(smoothness_score),
            "min_S": float(min_S),
            "max_abs_t": float(max_abs_t),
            "max_abs_S": float(max_abs_S),
        }

    except Exception as e:
        return {
            "combined_score": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }


import json

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) != 3:
        print("Usage: python evaluate.py <candidate_program.py> <results.json>", file=sys.stderr)
        return 2

    program_path = argv[1]
    results_path = argv[2]

    metrics = evaluate(program_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
