# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements an example of an initial solution in python.
#
# ===--------------------------------------------------------------------------------------===#


# EVOLVE-BLOCK-START
import math
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

# Próbujemy zaimportować biblioteki dla lepszej wizualizacji
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback dla track
    def track(iterable, description=""):
        return iterable

# ===--------------------------------------------------------------------------------------===#
# Klasa SystemState reprezentuje stan dynamiczny układu fizycznego.
# Uwzględnia: czas kosmiczny (t), pozycję (x), prędkość (v), entropię (S) i czas subiektywny (tau).
# ===--------------------------------------------------------------------------------------===#

@dataclass
class SystemState:
    """Stan układu fizycznego z jawnym modelem czasu."""
    t: float = 0.0                  # Czas kosmiczny (obiektywny)
    x: float = 0.0                  # Pozycja w przestrzeni 1D
    v: float = 0.0                  # Prędkość
    S: float = 0.0                  # Entropia układu (miara chaosu)
    tau: float = 0.0                # Czas subiektywny (odczuwany przez obserwatora)
    tension: float = 1.0            # Napięcie czasoprzestrzeni
    history: List[Dict] = field(default_factory=list)  # Historia stanów (do analizy)
    recent_states: deque = field(default_factory=lambda: deque(maxlen=5))  # Ostatnie stany dla analizy lokalnej

    def as_dict(self) -> Dict[str, Any]:
        """Zwraca aktualny stan jako słownik."""
        return {
            "t": self.t,
            "x": self.x,
            "v": self.v,
            "S": self.S,
            "tau": self.tau,
            "tension": self.tension
        }

    def evolve(self, dt: float, dx: float = 0.0, dv: float = 0.0, dS: float = 0.0, dtau: float = 0.0, dtension: float = 0.0):
        """Ewoluuje stan o zadane przyrosty."""
        self.t += dt
        self.x += dx
        self.v += dv
        self.S += dS
        self.tau += dtau
        self.tension += dtension
        # Zapisz stan do ostatnich stanów
        self.recent_states.append(self.as_dict().copy())

# ===--------------------------------------------------------------------------------------===#
# Klasy sił czasowych – abstrakcyjne operatory wpływające na układ.
# ===--------------------------------------------------------------------------------------===#

class TimeForce:
    """
    Bazowa klasa dla sił czasowych.
    Czas nie jest tylko parametrem – jest operatorem, który deformuje stan układu.
    """

    def apply(self, state: SystemState, dt: float) -> None:
        """
        Zastosuj siłę czasu do stanu.
        Domyślnie: liniowe przesunięcie czasu kosmicznego.
        """
        state.evolve(dt=dt, dS=0.01 * abs(dt))


class TemporalDrift(TimeForce):
    """
    Drift czasowy – stała siła przesuwająca układ w czasie i przestrzeni.
    Modeluje "wiatr czasu", który popycha układ do przodu z określoną siłą.
    """

    def __init__(self, strength: float = 1.0, spatial_push: float = 0.1):
        self.strength = strength
        self.spatial_push = spatial_push  # Stała siła przestrzenna

    def apply(self, state: SystemState, dt: float) -> None:
        dx = self.spatial_push * dt
        dS = 0.01 * abs(dt) * self.strength  # Rosnąca entropia
        state.evolve(dt=dt * self.strength, dx=dx, dS=dS)


class EventHorizonForce(TimeForce):
    """
    Siła horyzontu zdarzeń – czas deformuje się w pobliżu promienia krytycznego.
    W tym modelu: im bliżej x=10.0, tym wolniej płynie czas (time dilation).
    Zawiera zabezpieczenia przed niestabilnością numeryczną.
    """

    def __init__(self, radius: float = 10.0, epsilon: float = 1e-5, time_distortion: float = 2.0):
        self.radius = radius
        self.epsilon = epsilon
        self.time_distortion = time_distortion

    def apply(self, state: SystemState, dt: float) -> None:
        distance = max(self.epsilon, abs(state.x))  # unikamy dzielenia przez zero
        
        # W pobliżu horyzontu czas zwalnia i zakrzywia się
        proximity = max(0, self.radius - distance) / self.radius
        time_factor = 1.0 / (1.0 + self.time_distortion * proximity)
        time_factor = max(0.01, time_factor)  # ograniczenie na minimalny upływ czasu
        
        # Zastosowanie siły czasu
        local_dt = dt * time_factor
        
        # Przyciąganie w kierunku horyzontu z nieliniową siłą
        attraction = -0.5 * proximity * (1.0 + 0.5 * proximity)
        
        # Czas subiektywny zwalnia w pobliżu horyzontu
        gamma = max(0.001, time_factor)  # ograniczenie minimalne gamma
        local_dtau = gamma * dt
        
        # Entropia rośnie szybciej w pobliżu osobliwości
        entropy_factor = 1.0 + (1.0 - time_factor)
        dS = 0.01 * dt * entropy_factor * (1.0 + 0.5 * proximity)
        
        # Dynamika przestrzenna – przyciąganie do horyzontu
        pull = attraction * dt

        # Zmiana napięcia czasoprzestrzeni
        dtension = 0.1 * proximity * dt

        state.evolve(
            dt=local_dt,
            dx=pull,
            dS=dS,
            dtau=local_dtau,
            dtension=dtension
        )
        
        # Zapobieganie niestabilności numerycznej
        if abs(state.x) > 1e6 or abs(state.v) > 1e3:
            state.v *= 0.1  # tłumimy prędkość przy ekstremalnych wartościach


class CurvedTimeField(TimeForce):
    """
    Zakrzywione pole czasowe - nieliniowe przyspieszanie/hamowanie czasu 
    w zależności od pozycji w przestrzeni.
    """

    def __init__(self, curvature: float = 0.1, amplitude: float = 0.5):
        self.curvature = curvature
        self.amplitude = amplitude

    def apply(self, state: SystemState, dt: float) -> None:
        # Nieliniowa modyfikacja czasu w zależności od pozycji
        time_factor = 1.0 + self.amplitude * math.sin(self.curvature * state.x)
        local_dt = dt * time_factor
        
        # Zmiana prędkości zależna od gradientu pola czasowego
        dv = self.curvature * math.cos(self.curvature * state.x) * 0.1 * dt
        
        # Entropia rośnie w nieliniowym polu czasowym
        dS = 0.01 * abs(time_factor) * dt
        
        # Zmiana napięcia czasoprzestrzeni
        dtension = 0.05 * math.cos(self.curvature * state.x) * dt

        state.evolve(
            dt=local_dt,
            dv=dv,
            dS=dS,
            dtau=local_dt * 0.9,  # Czas subiektywny płynie nieco wolniej
            dtension=dtension
        )


class TemporalOscillator(TimeForce):
    """
    Oscylator czasowy - czas lokalnie oscyluje wokół wartości średniej,
    tworząc fluktuacje w przepływie czasu.
    """

    def __init__(self, frequency: float = 0.5, amplitude: float = 0.3):
        self.frequency = frequency
        self.amplitude = amplitude

    def apply(self, state: SystemState, dt: float) -> None:
        # Oscylujący współczynnik czasu
        oscillation = 1.0 + self.amplitude * math.sin(self.frequency * state.t)
        local_dt = dt * oscillation
        
        # Zmiana entropii zależna od szybkości oscylacji
        dS = 0.005 * abs(oscillation) * dt
        
        # Zmiana napięcia czasoprzestrzeni
        dtension = 0.02 * math.cos(self.frequency * state.t) * dt

        state.evolve(
            dt=local_dt,
            dS=dS,
            dtau=local_dt * (0.8 + 0.2 * math.cos(self.frequency * state.t)),  # Zmodyfikowany czas subiektywny
            dtension=dtension
        )

# ===--------------------------------------------------------------------------------------===#
# Integrator – strategia ewolucji układu przez siły czasowe.
# ===--------------------------------------------------------------------------------------===#

class Integrator:
    """Podstawowy integrator Eulera z możliwością rozbudowy."""

    @staticmethod
    def step(state: SystemState, forces: List[TimeForce], dt: float = 0.1):
        """Jeden krok całkowania przez listę sił czasowych."""
        for force in forces:
            force.apply(state, dt)

# ===--------------------------------------------------------------------------------------===#
# Obserwator – loguje i analizuje trajektorię układu.
# ===--------------------------------------------------------------------------------------===#

class Observer:
    """Prosty obserwator, który śledzi historię stanu."""

    @staticmethod
    def observe(state: SystemState):
        """Zapisuje aktualny stan do historii."""
        state.history.append(state.as_dict().copy())
    
    @staticmethod
    def print_trajectory(history: List[Dict], max_points: int = 20):
        """Drukuje skróconą trajektorię w formie tekstowej."""
        if not history:
            print("Brak danych do wyświetlenia")
            return
            
        step_count = len(history)
        if step_count <= max_points:
            points = history
        else:
            # Wybierz równomiernie rozłożone punkty
            indices = [int(i * (step_count - 1) / (max_points - 1)) for i in range(max_points)]
            points = [history[i] for i in indices]
        
        if RICH_AVAILABLE:
            console = Console()
            table = Table(title="Ewolucja Układu Czasowego")
            table.add_column("Krok", style="cyan")
            table.add_column("t", style="magenta")
            table.add_column("x", style="green")
            table.add_column("v", style="yellow")
            table.add_column("S", style="red")
            table.add_column("τ", style="blue")
            table.add_column("Tension", style="purple")
            
            for i, point in enumerate(points):
                table.add_row(
                    str(i),
                    f"{point['t']:.3f}",
                    f"{point['x']:.3f}",
                    f"{point['v']:.3f}",
                    f"{point['S']:.3f}",
                    f"{point['tau']:.3f}",
                    f"{point['tension']:.3f}"
                )
            console.print(table)
        else:
            # Wersja tekstowa bez rich
            print("\nEwolucja układu:")
            print("Krok\tt\t\tx\t\tv\t\tS\t\tτ\t\tTension")
            print("-" * 70)
            for i, point in enumerate(points):
                print(f"{i:2d}\t{point['t']:6.3f}\t\t{point['x']:6.3f}\t\t{point['v']:6.3f}\t\t{point['S']:6.3f}\t\t{point['tau']:6.3f}\t\t{point['tension']:6.3f}")

# ===--------------------------------------------------------------------------------------===#
# Funkcje wizualizacji w terminalu
# ===--------------------------------------------------------------------------------------===#

def visualize_time_flow(history: List[Dict]) -> None:
    """
    Wizualizacja przepływu czasu w terminalu.
    Pokazuje jak zmienia się tempo upływu czasu.
    """
    if not history or len(history) < 2:
        return
    
    print("\n=== Wizualizacja przepływu czasu ===")
    
    # Oblicz przyrosty czasu
    dt_values = [history[i]['t'] - history[i-1]['t'] for i in range(1, len(history))]
    max_dt = max(dt_values) if max(dt_values) > 0 else 1.0
    
    for i, dt in enumerate(dt_values[::max(1, len(dt_values)//20)]):  # Pokaż maksymalnie 20 punktów
        # Normalizuj do paska 20 znaków
        bar_length = int(20 * dt / max_dt)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{i:2d}: |{bar}| ({dt:.3f})")
    
    print("=== Koniec wizualizacji ===\n")


# ===--------------------------------------------------------------------------------------===#
# Symulacja – punkt wejścia do działania systemu.
# ===--------------------------------------------------------------------------------------===#

def run(steps: int = 30, dt: float = 0.1, seed: Optional[int] = 42) -> List[Dict]:
    """
    Uruchamia symulację układu z aktywnymi siłami czasowymi.

    :param steps: liczba kroków symulacji
    :param dt: bazowy krok czasowy
    :param seed: ziarno losowości
    :return: historia stanów układu
    """
    if seed is not None:
        random.seed(seed)

    # Inicjalizacja stanu układu z losowymi warunkami początkowymi
    state = SystemState(
        t=0.0,
        x=random.uniform(-2.0, 2.0),
        v=random.uniform(-0.5, 0.5),
        S=0.1,       # Minimalna entropia na start
        tau=0.0,
        tension=1.0
    )

    # Definicja sił czasowych
    forces = [
        TemporalDrift(strength=1.0, spatial_push=0.2),
        EventHorizonForce(radius=5.0, time_distortion=2.0),
        CurvedTimeField(curvature=0.15, amplitude=0.3),
        TemporalOscillator(frequency=0.8, amplitude=0.25)
    ]

    # Symulacja z pasekiem postępu
    current_dt = dt
    for _ in track(range(steps), description="Ewolucja czasu..."):
        Observer.observe(state)
        Integrator.step(state, forces, current_dt)
        
        # Dynamiczna zmiana dt w zależności od stanu układu
        current_dt = max(0.01, dt * (1.0 + abs(state.v) * 0.1))

    # Ostatni stan
    Observer.observe(state)
    
    # Wyświetlenie trajektorii
    Observer.print_trajectory(state.history)
    
    # Wizualizacja przepływu czasu
    visualize_time_flow(state.history)
    
    return state.history


def run_simulation(steps: int = 30, dt: float = 0.1) -> List[SystemState]:
    """
    Kompatybilność wsteczna z oryginalnym API.
    """
    history = run(steps, dt)
    # Konwersja z listy słowników do listy SystemState (dla kompatybilności)
    result = []
    for entry in history:
        state = SystemState()
        for key, value in entry.items():
            if hasattr(state, key):
                setattr(state, key, value)
        result.append(state)
    return result


# ===--------------------------------------------------------------------------------------===#
# Funkcja główna
# ===--------------------------------------------------------------------------------------===#

def main():
    """
    Główna funkcja uruchamiająca symulację.
    """
    print("Symulacja 'czasu jako siły' - Ewolucja stanu układu")
    print("=" * 55)
    
    history = run(steps=25, dt=0.2)
    
    if history:
        final_state = history[-1]
        print(f"\nKońcowy stan układu:")
        print(f"  Czas kosmiczny (t):     {final_state['t']:.3f}")
        print(f"  Czas subiektywny (τ):   {final_state['tau']:.3f}")
        print(f"  Pozycja (x):            {final_state['x']:.3f}")
        print(f"  Prędkość (v):           {final_state['v']:.3f}")
        print(f"  Entropia:               {final_state['S']:.3f}")
        print(f"  Napięcie czasoprzestrzeni: {final_state['tension']:.3f}")
        
        return final_state
    else:
        return {"error": "Symulacja nie zwróciła wyników"}


# EVOLVE-BLOCK-END
