# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2500  # Increased resolution to 2500 for finer function representation
    learning_rate: float = 0.0015  # Slightly reduced learning rate for stability with higher N
    num_steps: int = 600000  # Increased optimization steps for more thorough convergence
    warmup_steps: int = 12000  # Adjusted warmup steps proportionally to num_steps


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    The function f is represented by its parameters f_params, and the actual
    function values are f_actual = f_params**2 to enforce non-negativity.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        f_params are the parameters being optimized, f_actual = f_params**2.
        """
        # Enforce non-negativity by squaring the parameters.
        # This ensures f_actual >= 0 and provides better gradient properties than relu.
        # Add a small epsilon to f_actual to ensure it's strictly positive,
        # preventing it from collapsing to exactly zero and aiding optimization stability.
        # Inspired by Inspiration Program 3.
        f_actual = jnp.square(f_params) + 1e-6

        N = self.hypers.num_intervals
        # Assuming f is supported on [0, 1) and sampled at N points.
        # So, dx_f = 1.0 / N.
        dx_f = 1.0 / N

        # Calculate integral of f for L1 norm simplification: ||f ★ f||₁ = (∫f)²
        # Use trapezoidal rule for better accuracy, as in Inspirations 2/3.
        integral_f = (jnp.sum(f_actual) - 0.5 * (f_actual[0] + f_actual[-1])) * dx_f

        # Discrete autoconvolution using FFT
        # Pad f to length 2N for convolution on domain [0, 2)
        padded_f = jnp.pad(f_actual, (0, N)) # Length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution_raw = jnp.fft.ifft(fft_f * fft_f).real

        # Scale convolution to represent continuous convolution samples: (f*f)(x) = ∫f(t)f(x-t)dt
        # As established in Inspiration Program 2, the correct scaling factor is 2.0.
        g_samples = 2.0 * convolution_raw

        # The convolution g(x) = (f*f)(x) is supported on [0, 2) if f is on [0, 1).
        # It has `len(g_samples)` points (which is 2N).
        # Calculate L2-norm squared of the convolution (integral of g(x)^2 dx).
        # Use the highly accurate integration formula for a piecewise linear function,
        # which is the exact integral of the square of the linear interpolant of g.
        # Assume g is compactly supported, so g=0 at the boundaries of its domain [0, 2].
        # As in Inspirations 2/3, we add zeros at both ends for integration over [0, 2].
        y_points = jnp.concatenate([jnp.array([0.0]), g_samples, jnp.array([0.0])])
        # The convolution g has `len(g_samples)` points. Adding boundary zeros makes `len(g_samples) + 2` points.
        # This creates `len(g_samples) + 1` segments over its domain [0, 2].
        # So the step size for this integral is `h_conv_integral = 2.0 / (len(g_samples) + 1)`.
        h_conv_integral = 2.0 / (len(g_samples) + 1)

        y1_l2, y2_l2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv_integral / 3) * (y1_l2**2 + y1_l2 * y2_l2 + y2_l2**2))

        # Calculate infinity-norm of the convolution
        # Add a small epsilon to norm_inf for numerical stability, preventing division by zero if g_samples are all zero.
        # Inspired by Inspiration Program 3.
        norm_inf = jnp.max(g_samples) + 1e-10 # Since g_samples >= 0, abs is not needed.

        # Calculate C2 ratio
        # Add a small epsilon to the denominator for numerical stability, preventing division by zero.
        # Inspired by Inspiration 3's handling of integral_f for norm_1.
        denominator = ((integral_f + 1e-10)**2) * norm_inf + 1e-10
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_params)
        f_params = optax.apply_updates(f_params, updates)
        return f_params, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-5, # Decay to an even smaller value for finer convergence, as in Inspiration 1.
        )
        # Remove gradient clipping, as Inspiration Program 3's analysis suggested it might
        # hinder the optimizer from reaching higher C2 values.
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        
        # Initialize f_values with a shape known to be promising (U-shape),
        # as observed in Inspirations 2 and 3 that achieved the record C2.
        N = self.hypers.num_intervals
        # Use endpoint=False for consistency with f being supported on [0, 1) and dx_f = 1.0/N.
        x = jnp.linspace(0.0, 1.0, N, endpoint=False, dtype=jnp.float32)
        # A quadratic U-shape initialization: (x-0.5)**2 + 0.1
        f_actual_initial = (x - 0.5)**2 + 0.1
        
        # Convert initial f_actual values to f_params for optimization (f_actual = f_params**2)
        # Since f_actual_initial is strictly positive, jnp.sqrt is safe.
        f_params_initial = jnp.sqrt(f_actual_initial)

        opt_state = self.optimizer.init(f_params_initial)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        f_params_current = f_params_initial # Initialize the current parameters
        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_params_current, opt_state, loss = train_step_jit(f_params_current, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")

        final_c2 = -self._objective_fn(f_params_current)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the actual function values (f_params**2 + 1e-6) consistent with the objective function's definition.
        return jnp.square(f_params_current) + 1e-6, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
