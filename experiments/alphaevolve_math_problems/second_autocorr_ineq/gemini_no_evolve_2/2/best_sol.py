# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 200 # Increased for higher resolution
    learning_rate: float = 0.01
    num_steps: int = 40000 # Increased for longer training
    warmup_steps: int = 1000


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
        Assumes f is defined on [0, 1] and its convolution f*f is on [0, 2].
        """
        f_non_negative = jax.nn.relu(f_values)

        # Number of intervals for f
        N_f = self.hypers.num_intervals
        # Discretization step for f, assuming f is on [0, 1]
        dx_f = 1.0 / N_f

        # Calculate ||f * f||_1 = (integral f)^2 for the denominator
        # Integral f approx = sum(f_values) * dx_f (using rectangle rule for f)
        integral_f = jnp.sum(f_non_negative) * dx_f
        # Add a small epsilon to prevent division by zero if integral_f is initially zero
        integral_f_safe = jnp.maximum(integral_f, 1e-9)
        norm_1_denom = integral_f_safe**2

        # Unscaled discrete autoconvolution using FFT
        # Padded_f will have length 2*N_f.
        padded_f = jnp.pad(f_non_negative, (0, N_f))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # The convolution `g = f*f` is on domain [0, 2] and has 2*N_f samples.
        # So the effective step size for numerical integration of `g` is dx_g = 2.0 / (2*N_f) = 1.0 / N_f.
        h_integration = 1.0 / N_f

        # Calculate L2-norm squared of the convolution (rigorous method)
        # Interprets convolution values as samples of a piecewise-linear function.
        # The integration range for g is [0, 2]. We implicitly add 0s at boundaries.
        # y_points will have len(convolution) + 2 points.
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        # The sum is over len(convolution) + 1 intervals.
        # Each interval has length h_integration.
        l2_norm_squared = jnp.sum((h_integration / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution
        norm_inf = jnp.max(jnp.abs(convolution))
        # Add a small epsilon to prevent division by zero if norm_inf is initially zero
        norm_inf_safe = jnp.maximum(norm_inf, 1e-9)

        # Calculate C2 ratio
        denominator = norm_1_denom * norm_inf_safe
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
    # Updated hyperparameters for higher resolution and longer training
    hypers = Hyperparameters(num_intervals=200, num_steps=40000)
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
