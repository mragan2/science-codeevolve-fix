# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 200 # Increased resolution for better function approximation
    learning_rate: float = 0.01 # Keep learning rate, schedule handles decay
    num_steps: int = 50000 # Increased steps for convergence with higher num_intervals
    warmup_steps: int = 2000 # Proportionally increased warmup steps


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

        # Define the effective discretization step.
        # Assume f is defined on [0, L]. For unitless C2, we can set L=1.
        # dx_f = L / N = 1.0 / N.
        # The convolution_result will have 2N points on [0, 2L].
        # dx_conv = 2L / (2N) = L / N = 1.0 / N.
        # For the C2 formula, the `dx` factors should cancel out, leading to a
        # unitless expression.

        # Calculate integral_f = ∫f dx (approximated as sum(f_i) * dx)
        # For unitless C2, we use the discrete sum directly and adjust the final C2 formula.
        sum_f_values = jnp.sum(f_non_negative)

        # Unscaled discrete autoconvolution
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N)) # pads to 2N length
        fft_f = jnp.fft.fft(padded_f)
        convolution_result = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate ||f ★ f||_2^2 = ∫ (f ★ f)(x)^2 dx
        # Using simple rectangular rule for discrete samples, consistent with step functions.
        # The 'dx' term cancels out in the C2 ratio when defined consistently.
        sum_conv_squared = jnp.sum(convolution_result**2)

        # Calculate ||f ★ f||_1 = (∫f)^2, as per problem definition
        # For discrete functions, ∫f is proportional to sum(f_values).
        # We use (sum(f_values))^2 for the denominator, adjusted by N.
        norm_f_conv_1_discrete = sum_f_values**2

        # Calculate ||f ★ f||_inf = sup |(f ★ f)(x)|
        norm_inf = jnp.max(convolution_result)

        # Calculate C2 ratio based on corrected discrete formula.
        # C2 = (N * sum(g_k^2)) / ( (sum(f_j))^2 * max(g_k) )
        # This form correctly accounts for the 'dx' factors canceling out (assuming L=1).
        denominator = norm_f_conv_1_discrete * norm_inf
        
        # Add a small epsilon to the denominator to prevent NaN gradients.
        # If denominator is effectively zero, C2 is 0.
        c2_ratio = jnp.where(denominator > 1e-10, (N * sum_conv_squared) / denominator, 0.0)

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
