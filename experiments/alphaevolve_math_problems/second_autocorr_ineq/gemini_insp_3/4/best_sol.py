# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Enable 64-bit floating point precision for JAX computations.
# This is crucial for numerical stability and higher precision in constant discovery.
jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Increased resolution and training time for finding finer function structures,
    # aligned with the successful inspiration programs.
    num_intervals: int = 2048 # Increased resolution for better approximation of f
    learning_rate: float = 0.005 # Maintain optimal learning rate.
    num_steps: int = 250000 # Increased steps to allow convergence with higher resolution
    warmup_steps: int = 25000 # Increased proportionally to num_steps.


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
        This version incorporates best practices from high-performing implementations.
        """
        # Enforce f(x) >= 0 by squaring learnable parameters. This is smooth
        # and generally better for gradient-based optimization than relu.
        f_non_negative = jnp.square(f_values)

        # Define discretization step for f, assuming f is supported on [0, 0.5]
        # (as found effective in inspiration programs).
        N = self.hypers.num_intervals
        dx_f = 0.5 / N

        # Normalize f such that its integral is 1. This removes scale invariance,
        # stabilizing optimization and simplifying the objective.
        integral_f = jnp.sum(f_non_negative) * dx_f
        f_normalized = f_non_negative / (integral_f + 1e-9)

        # Padded autoconvolution using the *normalized* function.
        # Pad f to length 2N for convolution.
        padded_f = jnp.pad(f_normalized, (0, N)) # f_normalized len N, padded len 2N
        fft_f = jnp.fft.fft(padded_f)
        # Scale convolution by dx_f to approximate the continuous integral.
        convolution = jnp.fft.ifft(fft_f * fft_f).real * dx_f

        # With f normalized, ||f ★ f||_1 = (∫f)^2 = 1. The denominator simplifies.
        norm_1 = 1.0

        # Calculate infinity-norm of the convolution.
        norm_inf = jnp.max(jnp.abs(convolution)) + 1e-9 # Add epsilon for robustness

        # Calculate L2-norm squared of the convolution (rigorous method).
        # Since f is on [0, 0.5], f*f is on [0, 1].
        # Use the formula for integrating the square of a piecewise linear function.
        num_conv_points = len(convolution) # convolution has 2N points
        # The step size for the convolution integral over [0, 1].
        # If convolution has 2N points and spans [0, 1],
        # it corresponds to 2N-1 intervals, or 2N+1 points including boundaries.
        # The inspiration uses 1.0 / (num_conv_points + 1) which implies 2N+1 points over [0,1].
        h = 1.0 / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate C2 ratio with simplified denominator.
        denominator = norm_1 * norm_inf + 1e-9
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

        # Set numpy random seed for full reproducibility alongside JAX's PRNGKey.
        np.random.seed(42)

        # Initialize parameters with small positive random values. When squared,
        # this creates a stable, non-zero starting function. A seed ensures reproducibility.
        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        f_values = (
            jax.random.uniform(key, shape=(N,), dtype=jnp.float64) * 0.1 + 0.01
        )

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        # Use a more appropriate progress reporting interval for the increased step count.
        report_interval = max(1000, self.hypers.num_steps // 50)
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % report_interval == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.12f}") # Print C2 with higher precision

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.12f}") # Print final C2 with higher precision

        # Return the final function f that achieves the C2 score.
        # This involves squaring the optimized parameters and normalizing.
        final_f_non_negative = jnp.square(f_values)
        dx_f = 0.5 / N
        integral_f = jnp.sum(final_f_non_negative) * dx_f
        final_f_normalized = final_f_non_negative / (integral_f + 1e-9)
        return final_f_normalized, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
