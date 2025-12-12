"""
Time-Force Idea Evaluator (Updated for Event Horizon).

Zadanie:
- Mamy seed: "czas jest siłą", który ewoluował w stronę relatywistyki i czarnych dziur.
- OpenEvolve ma budować kod, który:
    1) Eksploruje naturę czasu (siła, rozmycie, horyzont zdarzeń),
    2) Wykorzystuje bogatą strukturę (klasy, dziedziczenie, polimorfizm),
    3) Jest poprawny technicznie i dobrze udokumentowany.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any

import json

import numpy as np

# ZAKTUALIZOWANE SŁOWA KLUCZOWE
# Dodaliśmy terminy związane z czarnymi dziurami, horyzontem i Boone'em
KEYWORDS_PL = [
    "czas", "siła", "popycha", "ewolucja", "strzałka czasu",
    "przyszłość", "przeszłość", "dynamika",
    "horyzont", "osobliwość", "grawitacja", "zatrzymanie",
    "odwrócenie", "boone", "rozmycie",
]
KEYWORDS_EN = [
    "time", "force", "flow", "arrow of time", "evolution", "state",
    "event horizon", "singularity", "gravity", "stop", "reversal",
    "blur", "relative",
]


def _sanitize_candidate_file(path: Path) -> None:
    """Usuwa bloki ``` jeśli kandydat został wklejony jako Markdown."""
    try:
        text = path.read_text(encoding="utf-8")
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


def _load_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _syntax_score(src: str) -> float:
    """Sprawdza, czy kod parsuje się jako AST. 1.0 jeśli tak, 0.0 jeśli nie."""
    try:
        ast.parse(src)
        return 1.0
    except SyntaxError:
        return 0.0


def _idea_alignment_score(src: str) -> float:
    """
    Sprawdza, na ile tekst kodu pasuje do 'czas jako siła' ORAZ nowych koncepcji
    (czarne dziury, relatywistyka).
    """
    text = src.lower()
    search_terms = KEYWORDS_PL + KEYWORDS_EN

    found = set()
    for w in search_terms:
        if w in text:
            found.add(w)

    # 5–6 trafień to już bardzo dobry wynik
    score = len(found) / 6.0
    return min(1.0, score)


def _structure_score_from_ast(src: str) -> float:
    """
    Mierzy wyrafinowanie struktury:
    - liczba klas (premiujemy dziedziczenie np. TimeForce -> EventHorizonForce)
    - liczba funkcji
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0.0

    class Counter(ast.NodeVisitor):
        def __init__(self) -> None:
            self.n_classes = 0
            self.n_funcs = 0
            self.max_depth = 0
            self.has_inheritance = False

        def generic_visit(self, node, depth=0):
            self.max_depth = max(self.max_depth, depth)
            super().generic_visit(node)

        def visit_ClassDef(self, node):
            self.n_classes += 1
            if node.bases:
                self.has_inheritance = True
            for child in ast.iter_child_nodes(node):
                self.generic_visit(child, depth=1)

        def visit_FunctionDef(self, node):
            self.n_funcs += 1
            for child in ast.iter_child_nodes(node):
                self.generic_visit(child, depth=1)

    c = Counter()
    c.visit(tree)

    cls_score = min(1.0, c.n_classes / 3.0)
    fn_score = min(1.0, c.n_funcs / 6.0)
    depth_score = min(1.0, c.max_depth / 4.0)
    inheritance_bonus = 0.2 if c.has_inheritance else 0.0

    base_score = 0.4 * cls_score + 0.4 * fn_score + 0.2 * depth_score
    return min(1.0, base_score + inheritance_bonus)


def _documentation_score(src: str) -> float:
    """Liczba linii komentarzy i docstringi."""
    lines = src.splitlines()
    if not lines:
        return 0.0

    n_comment = sum(1 for l in lines if l.strip().startswith("#"))
    comment_ratio = n_comment / max(len(lines), 1)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0.0

    has_module_doc = ast.get_docstring(tree) is not None

    n_docstrings = 1 if has_module_doc else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is not None:
                n_docstrings += 1

    docstring_score = min(1.0, n_docstrings / 5.0)
    comment_score = min(1.0, comment_ratio / 0.15)

    return 0.5 * docstring_score + 0.5 * comment_score


def _introspection_score(module: Any) -> float:
    """
    Sprawdza API i obecność kluczowych klas.
    """
    names = [n for n in dir(module) if not n.startswith("_")]
    objs = [getattr(module, n) for n in names]

    n_callables = sum(callable(o) for o in objs)
    n_tests = sum(
        1 for n, o in zip(names, objs)
        if callable(o) and n.startswith("test_")
    )

    has_force_class = any(
        ("force" in n.lower() and isinstance(getattr(module, n), type))
        for n in names
    )

    api_score = min(1.0, n_callables / 8.0)
    test_score = min(1.0, n_tests / 3.0)
    force_bonus = 0.2 if has_force_class else 0.0

    return float(0.6 * api_score + 0.3 * test_score + force_bonus)


# -------------------------------------------------------------------
#  GŁÓWNA FUNKCJA EWALUACJI (bez etapów / stages)
# -------------------------------------------------------------------

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Główna funkcja ewaluacji dla CodeEvolve.

    Zwraca słownik metryk, w tym:
    - combined_score        – wewnętrzny score 0–1
    - COMBINED_SCORE        – alias używany jako fitness_key
    - feat1                 – oś dla MAP-Elites (pomysł + struktura)
    - syntax, idea_alignment, structure, documentation, introspection, stability
    """
    metrics: Dict[str, float] = {}
    path = Path(program_path)
    _sanitize_candidate_file(path)

    src = _load_source(path)
    if not src:
        return {
            "combined_score": 0.0,
            "COMBINED_SCORE": 0.0,
            "feat1": 0.0,
            "stability": 1.0,
        }

    # 1. Składnia
    syntax = _syntax_score(src)
    if syntax == 0.0:
        return {
            "combined_score": 0.0,
            "COMBINED_SCORE": 0.0,
            "feat1": 0.0,
            "stability": 1.0,
        }
    metrics["syntax"] = syntax

    # 2. Alignment z ideą (czas jako siła + horyzont, rozmycie, Boone)
    metrics["idea_alignment"] = _idea_alignment_score(src)

    # 3. Struktura (AST) + bonus za dziedziczenie
    metrics["structure"] = _structure_score_from_ast(src)

    # 4. Dokumentacja
    metrics["documentation"] = _documentation_score(src)

    # 5. Import + introspekcja
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        metrics["introspection"] = _introspection_score(module)
    except Exception:
        metrics["introspection"] = 0.0

    # GŁÓWNY SCORE
    score = (
        0.30 * metrics.get("idea_alignment", 0.0) +
        0.25 * metrics.get("structure", 0.0) +
        0.20 * metrics.get("documentation", 0.0) +
        0.15 * metrics.get("introspection", 0.0) +
        0.10 * metrics.get("syntax", 0.0)
    )

    metrics["combined_score"] = float(np.clip(score, 0.0, 1.0))
    # Alias dla fitness_key: 'COMBINED_SCORE'
    metrics["COMBINED_SCORE"] = metrics["combined_score"]

    # Oś dla MAP-Elites: mieszanka idei i struktury
    feat1 = 0.5 * metrics.get("idea_alignment", 0.0) + 0.5 * metrics.get("structure", 0.0)
    metrics["feat1"] = float(np.clip(feat1, 0.0, 1.0))

    metrics["stability"] = 1.0
    return metrics


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

