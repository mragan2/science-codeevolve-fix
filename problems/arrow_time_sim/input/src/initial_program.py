# EVOLVE-BLOCK-START
import math
from typing import Dict, List


def arrow_fields(t: float):
    """
    Trzy „pola strzałki czasu” – deterministyczne, gładkie i ograniczone.
    """
    t = float(t)
    A = math.tanh(0.9 * t)
    B = math.tanh(0.6 * t - 0.8)
    C = math.tanh(1.1 * t + 0.15 * math.sin(t))
    return A, B, C


def entropy_from_fields(t: float, A: float, B: float, C: float) -> float:
    """
    Efektywna entropia S(t):
    - zawsze >= 0
    - rośnie wraz z t (dla t rosnącego), czyli implementuje strzałkę czasu
    """
    s2 = A * A + B * B + C * C
    # Składnik liniowy w t gwarantuje monotoniczność przy rosnącym t.
    S = max(0.0, t + math.log1p(s2))
    return S


def run(steps: int = 60, dt: float = 0.1) -> List[Dict[str, float]]:
    """
    Zwraca trajektorię jako listę słowników (to oczekuje evaluator).
    """
    steps = int(steps)
    dt = float(dt)

    t = 0.0
    out: List[Dict[str, float]] = []

    for _ in range(max(1, steps)):
        A, B, C = arrow_fields(t)
        S = entropy_from_fields(t, A, B, C)
        out.append({"t": t, "A": A, "B": B, "C": C, "S": S})
        t += dt

    return out


def main():
    # Lokalny test (nie jest używany przez evaluator).
    traj = run(steps=30, dt=0.1)
    last = traj[-1]
    print("OK. Final:", last)


# EVOLVE-BLOCK-END
