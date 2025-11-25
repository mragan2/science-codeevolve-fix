# EVOLVE-BLOCK-START
import jax
# Enable 64-bit floating point precision for higher accuracy, crucial for finding high-precision constants.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 3000 # Further increased resolution (from 2000 to 3000) for finer function representation.
    learning_rate: float = 0.003 # Learning rate remains consistent, higher precision might allow for a slightly larger one.
    num_steps: int = 600000 # Increased steps (from 400k to 600k) to allow for better convergence with higher num_intervals and x64 precision.
    warmup_steps: int = 15000 # Increased warmup steps proportionally to total steps.


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method,
    with enforced symmetry for the function f.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        # Removed explicit symmetry enforcement; f_values will be optimized directly.
        # num_intervals now directly corresponds to the number of optimized parameters.
        # The constraint for num_intervals to be even is no longer strictly necessary for symmetry,
        # but often preferred for FFT-based methods or consistent discretization.

    def _objective_fn(self, f_values_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function for the C2 constant.
        f_values_params represents the full function f on its domain [0, 1].
        """
        # Ensure f_values are non-negative.
        # jax.nn.relu ensures non-negativity and differentiability.
        f_values = jax.nn.relu(f_values_params)
        # We now optimize the full f_values directly, without enforcing f(x)=f(1-x) symmetry.

        # Unscaled discrete autoconvolution
        N = self.hypers.num_intervals
        # Pad f_values to 2N length for linear convolution using FFT
        padded_f = jnp.pad(f_values, (0, N)) 
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Define step size for f on [0, 1].
        # We assume f is a piecewise constant function (step function) over N intervals on [0, 1].
        dx_f = 1.0 / N

        # Calculate (integral f)^2, which is ||f*f||_1 as per the problem statement.
        # This is the sum of f_i (heights) multiplied by the interval width dx_f, then squared.
        integral_f = jnp.sum(f_values) * dx_f
        norm_1_convolution = integral_f**2

        # Define step size for convolution values.
        # If f is on [0, 1] with N points, then f*f is on [0, 2] with 2N points.
        # The step size for convolution values is (2 - 0) / (2N) = 1/N.
        # The convolution array has length 2N.
        dx_conv = 2.0 / len(convolution) # This simplifies to 1.0 / N, same as dx_f

        # Calculate infinity-norm of the convolution.
        # Since f(x) >= 0, f*f(x) >= 0, so jnp.abs is not strictly needed.
        norm_inf_convolution = jnp.max(convolution)

        # Calculate L2-norm squared of the convolution (integral of g(x)^2 dx).
        # When f is piecewise constant, f*f (g) is piecewise linear.
        # The integral of g(x)^2 for a piecewise linear g given points g_k and step dx_conv is:
        # sum_{k} (dx_conv / 3) * (g_k^2 + g_k * g_{k+1} + g_{k+1}^2)
        # The convolution array has 2N points (g_0 to g_{2N-1}).
        # This integrates over 2N-1 intervals, spanning the domain [0, 2 - dx_conv].
        y1_conv = convolution[:-1] # Values g_0, ..., g_{2N-2}
        y2_conv = convolution[1:]  # Values g_1, ..., g_{2N-1}
        l2_norm_squared_convolution = jnp.sum((dx_conv / 3) * (y1_conv**2 + y1_conv * y2_conv + y2_conv**2))

        # Calculate C2 ratio.
        # Add a small epsilon to the denominator to prevent division by zero,
        # which can occur if f_values collapse to all zeros during optimization.
        denominator = norm_1_convolution * norm_inf_convolution
        epsilon = 1e-12 # Small constant for numerical stability
        c2_ratio = l2_norm_squared_convolution / (denominator + epsilon)

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values_params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step on f_values_params."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values_params)
        f_values_params = optax.apply_updates(f_values_params, updates)
        # Ensure f_values_params remain non-negative after update.
        # jax.nn.relu in objective takes care of the strict non-negativity for calculation,
        # but clipping here ensures the parameters themselves don't become strongly negative,
        # which can sometimes lead to numerical issues or slow convergence.
        f_values_params = jnp.clip(f_values_params, a_min=0.0) 
        return f_values_params, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-5, # Lower end LR for finer tuning at the end.
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        
        # Initialize f_values_params with a uniform random distribution within a small positive range.
        # This breaks any implicit symmetry bias from a Gaussian initialization and encourages
        # the discovery of non-symmetric optimal functions, which are known to achieve higher C2.
        # Adding a small minimum value ensures all initial f_values are non-zero.
        f_values_params = jax.random.uniform(
            key, 
            (self.hypers.num_intervals,), 
            minval=0.01, # Ensure values are positive and non-trivial
            maxval=0.1   # Keep the initial range small to prevent large gradients
        )
        # Ensure initial values are positive (though minval already does this)
        f_values_params = jax.nn.relu(f_values_params)


        opt_state = self.optimizer.init(f_values_params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, "
            f"Number of optimized parameters: {self.hypers.num_intervals}, " # Now N parameters
            f"Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values_params, opt_state, loss = train_step_jit(f_values_params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")

        # The final optimized f_values are directly f_values_params (after relu from objective)
        # We don't need to reconstruct a symmetric function anymore.
        final_f_values_non_negative = jax.nn.relu(f_values_params)
        
        final_c2 = -self._objective_fn(f_values_params) # Pass optimized f_values_params to objective_fn
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        return final_f_values_non_negative, final_c2 # Return the full f_values


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    # optimizer.run_optimization now returns the full f_values
    optimized_f_full, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f_full) # Use the full f_values for output

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
