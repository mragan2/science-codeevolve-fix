# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Increased resolution for better approximation and potentially higher C2
    num_intervals: int = 200 # Increased from 50
    learning_rate: float = 0.01
    # Increased number of steps for higher resolution and better convergence
    num_steps: int = 50000 # Increased from 15000
    # Adjusted warmup steps proportionally
    warmup_steps: int = 5000 # Increased from 1000


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        """
        f_non_negative = jax.nn.relu(f_values)

        # Unscaled discrete autoconvolution
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Ensure f is non-negative
        f_non_negative = jax.nn.relu(f_values)

        # Calculate integral of f for the denominator: (∫f)²
        # Assuming f_non_negative represents f(x) for x in [0, 1] with N intervals (dx = 1/N)
        N = self.hypers.num_intervals
        dx_f = 1.0 / N
        integral_f = jnp.sum(f_non_negative) * dx_f
        # Add a small epsilon to prevent division by zero if integral_f is 0
        integral_f = jnp.maximum(integral_f, 1e-9)

        # Unscaled discrete autoconvolution using FFT
        # Padded length for linear convolution. If f_values has N points, padded_f is 2N points.
        # Convolution of f on [0,1] with f on [0,1] is supported on [0,2].
        padded_f = jnp.pad(f_non_negative, (0, N)) # Pads to 2N elements
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Determine the spatial step for the convolution result
        # 'convolution' array has length 2N, spans [0, 2]. So dx_conv = 2.0 / (2N) = 1.0 / N.
        dx_conv = 1.0 / N

        # Calculate L2-norm squared of the convolution: ||f ★ f||₂² = ∫(f★f)(x)² dx
        # If f is piecewise constant, f*f is piecewise linear.
        # The integral formula for piecewise linear functions: ∫ g(x)^2 dx over [a,b] is (b-a)/3 * (g(a)^2 + g(a)g(b) + g(b)^2).
        # The `convolution` array has `2N` points, representing (f*f)(k * dx_conv).
        # We sum over `2N` intervals from 0 to 2.
        # We need values at interval endpoints. If convolution[k] is at x=k*dx_conv, then
        # y1 covers [conv_0, ..., conv_{2N-1}] and y2 covers [conv_1, ..., conv_{2N-1}, 0].
        y_points_for_integral = jnp.concatenate([convolution, jnp.array([0.0])]) # Adds a zero at the end for the last interval
        y1, y2 = y_points_for_integral[:-1], y_points_for_integral[1:]
        l2_norm_squared = jnp.sum((dx_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution: ||f ★ f||_{∞} = sup|(f★f)(x)|
        norm_inf = jnp.max(jnp.abs(convolution))
        # Add a small epsilon to prevent division by zero
        norm_inf = jnp.maximum(norm_inf, 1e-9)

        # Calculate C2 ratio: C₂ = ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        denominator = (integral_f**2) * norm_inf
        # Add a small epsilon to prevent division by zero (integral_f is squared, so 1e-9 squared is 1e-18)
        denominator = jnp.maximum(denominator, 1e-18)

        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,))

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        return jax.nn.relu(f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
