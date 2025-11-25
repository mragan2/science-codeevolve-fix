# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 200 # Increased resolution for better approximation of f(x)
    learning_rate: float = 0.005 # Slightly reduced learning rate for stability with more steps
    num_steps: int = 50000 # Increased steps for convergence with higher resolution
    warmup_steps: int = 5000 # Adjusted warmup steps to match increased total steps


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
        # Ensure non-negativity by squaring the parameters.
        # This allows gradients to flow for all parameter values, unlike relu,
        # and guarantees f_actual >= 0.
        f_actual = f_values**2

        # Define discretization step size (assuming f is sampled on [0,1])
        N = self.hypers.num_intervals
        dx = 1.0 / N

        # Calculate integral_f = ∫f dx (using Riemann sum for f_actual on [0,1])
        integral_f = jnp.sum(f_actual) * dx
        # Add a small epsilon to prevent division by zero if f_actual collapses to all zeros.
        integral_f_stable = jnp.maximum(integral_f, 1e-9)

        # Unscaled discrete autoconvolution using FFT.
        # If f_actual has N elements (on [0,1]), padded_f needs 2N elements for linear convolution.
        # The convolution of N points on [0,1] with N points on [0,1] yields 2N-1 points on [0,2].
        # The FFT result `convolution` will have 2N elements.
        padded_f = jnp.pad(f_actual, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # The convolution f*f is supported on [0, 2] and sampled at `len(convolution)` points (which is 2N).
        # The effective step size for integrating the convolution is dx_conv = 2.0 / len(convolution) = 2.0 / (2*N) = 1.0 / N.
        dx_conv = dx # As derived, dx_conv is the same as dx

        # Calculate L2-norm squared of the convolution: ||f ★ f||₂² = ∫(f★f)(x)² dx
        # Since f is piecewise constant, f*f is piecewise linear.
        # We use a more accurate numerical integration (Simpson's rule for g^2) over each segment.
        # ∫_x_i^x_{i+1} g(x)^2 dx = h/3 * (g(x_i)^2 + g(x_i)g(x_{i+1}) + g(x_{i+1})^2)
        # Summing over all `len(convolution)-1` intervals.
        l2_norm_squared = (dx_conv / 3) * jnp.sum(
            convolution[:-1]**2 + convolution[:-1]*convolution[1:] + convolution[1:]**2
        )

        # Calculate infinity-norm of the convolution: ||f ★ f||_{∞} = sup|(f★f)(x)|
        # Since f_actual >= 0, convolution >= 0, so jnp.abs is not needed.
        norm_inf = jnp.max(convolution)
        # Add a small epsilon to prevent division by zero.
        norm_inf_stable = jnp.maximum(norm_inf, 1e-9)

        # Calculate C2 ratio using the problem's identity: ||f ★ f||₁ = (∫f)²
        # C₂ = ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        denominator = (integral_f_stable**2) * norm_inf_stable
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
        # Initialize `f_values` (the parameters that are squared to form f_actual).
        # A small uniform random initialization allows for exploration while ensuring f_actual is not too large initially.
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,)) * 0.1

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
        # Return the actual function values (f_values squared), not the raw parameters.
        return f_values**2, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
