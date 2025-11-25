# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 4096 # Further increased resolution (power of 2 for FFT efficiency) to allow for even finer details in the optimal function.
    learning_rate: float = 0.001 # Retain optimal learning rate for stable convergence.
    num_steps: int = 1000000 # Increased steps to ensure full convergence with higher resolution and complex landscape.
    warmup_steps: int = 12500 # Adjusted warmup steps proportionally to new num_steps to maintain schedule characteristics.


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    The function f is represented by its parameters f_params, and `jax.nn.relu`
    is applied to enforce non-negativity.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function (negative C2) using the unitless norm calculation.
        f_params are the parameters being optimized. f_non_negative = relu(f_params).
        """
        # Enforce non-negativity using ReLU, as observed in higher-performing inspirations.
        f_non_negative = jax.nn.relu(f_params)

        N = self.hypers.num_intervals
        # Assuming f is supported on [0, 1) and sampled at N points.
        dx_f = 1.0 / N

        # Calculate integral of f for L1 norm simplification: ||f ★ f||₁ = (∫f)²
        # Use trapezoidal rule for better accuracy (from inspirations).
        # For N points from 0 to 1-1/N, this approximates integral over [0, 1-1/N].
        integral_f = (jnp.sum(f_non_negative) - 0.5 * (f_non_negative[0] + f_non_negative[-1])) * dx_f
        # Ensure integral_f is not zero for numerical stability (from inspirations).
        integral_f = jnp.maximum(integral_f, 1e-9)
        norm_1 = integral_f**2

        # Discrete autoconvolution using FFT
        # Pad f to length 2N for convolution on domain [0, 2)
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution_raw = jnp.fft.ifft(fft_f * fft_f).real

        # Scale convolution to represent continuous convolution samples: (f*f)(x) = ∫f(t)f(x-t)dt
        # This 2.0 factor is crucial for correct scaling, as consistently seen in inspirations.
        convolution_samples = 2.0 * convolution_raw

        # The convolution g(x) = (f*f)(x) is supported on [0, 2) if f is on [0, 1).
        M_conv_points = len(convolution_samples) # This is 2N
        # The integration step for convolution, for piecewise linear integral over M_conv_points intervals
        # spanning [0, 2]. (Consistent with Inspirations 2 & 3 for mathematical rigor).
        h_conv = 2.0 / M_conv_points

        # Calculate L2-norm squared of the convolution (integral of g(x)^2 dx)
        # We append a zero at the end, assuming g(2) = 0 for compact support.
        y_points_for_int = jnp.concatenate([convolution_samples, jnp.array([0.0])])
        y1, y2 = y_points_for_int[:-1], y_points_for_int[1:]

        l2_norm_squared = jnp.sum((h_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution
        norm_inf = jnp.max(convolution_samples) # Since convolution_samples >= 0, abs is not needed.
        # Ensure norm_inf is not zero for numerical stability (from inspirations).
        norm_inf = jnp.maximum(norm_inf, 1e-9)

        # Calculate C2 ratio
        # Add a small epsilon to the denominator for numerical stability.
        denominator = norm_1 * norm_inf + 1e-10
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
        # Removed gradient clipping, as higher-performing inspirations (1 & 2) did not use it.
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        
        # Initialize f_params with a "U" shape, as found to be promising in inspirations (1, 2, 3).
        # This provides a strong prior for a function that performs well and helps guide
        # the optimizer towards better local optima than simple box or random initialization.
        x_grid = jnp.linspace(0.0, 1.0, self.hypers.num_intervals, endpoint=False, dtype=jnp.float32)
        f_params_initial = (x_grid - 0.5)**2 + 0.1 
        f_params_initial = jnp.array(f_params_initial, dtype=jnp.float32)

        opt_state = self.optimizer.init(f_params_initial)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        f_params_current = f_params_initial # Initialize the current parameters for iteration
        for step in range(self.hypers.num_steps):
            f_params_current, opt_state, loss = train_step_jit(f_params_current, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # Recompute C2 explicitly for printing for clarity and robustness (from inspirations).
                current_c2 = -self._objective_fn(f_params_current)
                print(f"Step {step:5d} | C2 ≈ {current_c2:.8f}")

        final_c2 = -self._objective_fn(f_params_current)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the actual function values after applying ReLU to ensure non-negativity (from inspirations).
        return jax.nn.relu(f_params_current), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
