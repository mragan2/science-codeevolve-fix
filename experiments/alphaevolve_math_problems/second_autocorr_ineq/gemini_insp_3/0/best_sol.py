# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Adopt aggressive hyperparameters from the world-record-breaking inspiration program (Insp 1)
    num_intervals: int = 2000 # Significantly increased resolution to push boundaries further, as in Insp 2
    learning_rate: float = 0.005 # Retained from Insp 2 for stability with increased parameters
    num_steps: int = 500000 # Substantially increased steps for deeper optimization, matches Insp 2
    warmup_steps: int = 20000 # Adjusted warmup steps proportionally to new num_steps, as in Insp 2


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        # Removed the num_intervals % 2 check as we are no longer explicitly enforcing symmetry
        # by splitting f_param in half, aligning with Insp 1's approach.

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        This implementation is heavily inspired by Inspiration Program 1, which
        achieved a new world record. It normalizes f to have integral 1, which
        stabilizes optimization and simplifies the C2 objective.
        """
        N_f = self.hypers.num_intervals

        # Ensure f(x) >= 0 using ReLU, allowing for step-like functions, as in Insp 2.
        f_non_negative = jax.nn.relu(f_values)

        # Calculate integral of f. f is piecewise constant over N_f intervals on [0, 1].
        dx_f = 1.0 / N_f
        current_integral_f = dx_f * jnp.sum(f_non_negative)
        
        # Normalize f_non_negative such that its integral is 1.
        # This simplifies the C2 denominator as (∫f)^2 becomes 1.
        # Guard against division by zero if f_values are all zero initially.
        f_normalized = f_non_negative / (current_integral_f + 1e-12)

        # Unscaled discrete autoconvolution using FFT.
        # Padded to 2*N_f points for linear convolution, corresponding to domain [0, 2].
        # Use the normalized function for convolution.
        padded_f = jnp.pad(f_normalized, (0, N_f))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate L2-norm squared of the convolution using the rigorous piecewise-linear integration
        # from Inspiration Program 2.
        # The convolution `g = f*f` is supported on [0, 2].
        # `convolution` array has `2N_f` points.
        # `num_conv_points` is 2*N_f. `dx_conv` is 2.0 / (2*N_f) = 1.0 / N_f.
        num_conv_points = len(convolution)
        dx_conv = 2.0 / num_conv_points

        # To apply the piecewise-linear integration formula over `num_conv_points` intervals,
        # we need `num_conv_points + 1` points. We append `g(2)=0`.
        y_points_for_integral = jnp.concatenate([convolution, jnp.array([0.0])])
        y1, y2 = y_points_for_integral[:-1], y_points_for_integral[1:]
        l2_norm_squared = jnp.sum((dx_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # With f normalized, its integral is 1, so ||f ★ f||₁ = (∫f)^2 = 1.
        # The denominator simplifies to just ||f ★ f||_inf.
        # Since f >= 0 and relu is used, f*f >= 0, so abs() is not needed.
        norm_inf = jnp.max(convolution)

        # Calculate C2 ratio. Use jnp.maximum for norm_inf to prevent division by zero
        # and ensure stability, as seen in Insp 2's world-record-breaking implementation.
        denominator = jnp.maximum(norm_inf, 1e-12)
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step. Optimizes f_values."""
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
        # Initialize f_values with small positive random numbers to ensure non-trivial integral
        # and allow the optimizer to explore from a relatively "flat" but non-zero start, as in Insp 2.
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,)) * 0.01

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            # Adjusted print frequency to match Inspiration 2 for longer training runs.
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.12f}") # Increased precision for reporting

        final_c2 = -self._objective_fn(f_values) # Recalculate with final f_values
        print(f"\nFinal C2 lower bound found: {final_c2:.12f}") # Increased precision for final output
        
        # Check if a new world record has been set, using the current best from Insp 2.
        current_record = 0.9185398982313954
        if final_c2 > current_record:
            print(f"🎉 NEW WORLD RECORD! Surpassed {current_record:.12f}")
        else:
            print(f"Did not surpass current record of {current_record:.12f}")
            
        # Return the final optimized function f, which is the ReLU-applied and normalized version
        # of the raw optimized parameters.
        final_f_non_negative = jax.nn.relu(f_values) # Apply ReLU as used in objective
        h = 1.0 / self.hypers.num_intervals
        integral_f = jnp.sum(final_f_non_negative) * h
        final_f_normalized = final_f_non_negative / (integral_f + 1e-9)
        return final_f_normalized, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    # The external verifier expects the 'f_values' argument to have shape (num_intervals,).
    # The returned optimized_f is the final, normalized function that achieved the C2 score.
    return np.array(optimized_f), float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
