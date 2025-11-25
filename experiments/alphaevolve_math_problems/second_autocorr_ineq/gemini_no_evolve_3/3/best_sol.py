# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 200 # Increased resolution for better function representation
    learning_rate: float = 0.01 # Kept same, might need tuning
    num_steps: int = 50000 # Increased steps for higher resolution and longer convergence
    warmup_steps: int = 1000 # Kept same


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

        # Discretization parameters: f is on [0, 1] with N intervals.
        N = self.hypers.num_intervals
        dx_f = 1.0 / N

        # Calculate integral of f: ∫f dt
        integral_f = jnp.sum(f_non_negative) * dx_f

        # Unscaled discrete autoconvolution using FFT.
        # Padded_f will have length 2N, representing f on [0, 1] with N points,
        # then N zeros. The convolution (f*f) will be on [0, 2].
        padded_f = jnp.pad(f_non_negative, (0, N)) # Length 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real # Length 2N

        # Ensure convolution is strictly non-negative due to numerical precision,
        # as f*f must be non-negative for non-negative f.
        convolution = jax.nn.relu(convolution)

        # Calculate L2-norm squared of the convolution (rigorous method).
        # This method assumes f*f is piecewise linear between samples.
        # The convolution has `M = len(convolution) = 2N` samples.
        # We integrate over `M+1` intervals (by adding zero endpoints), covering the domain [0, 2].
        # So, the effective interval width `h_integral` is 2.0 / (M+1).
        num_conv_samples = len(convolution)
        h_integral = 2.0 / (num_conv_samples + 1)

        # y_points for the piecewise linear integration, assuming f*f starts and ends at 0.
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_integral / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate L1-norm of the convolution: ||f ★ f||_1 = (∫f)²
        # For non-negative f, f*f is non-negative, and ∫(f*f) = (∫f)^2.
        norm_1 = integral_f**2

        # Calculate infinity-norm of the convolution.
        # Since convolution is now guaranteed non-negative, abs is not needed.
        norm_inf = jnp.max(convolution)

        # Calculate C2 ratio. Add epsilon for numerical stability.
        denominator = norm_1 * norm_inf
        epsilon = 1e-12 # Small epsilon to prevent division by zero
        c2_ratio = l2_norm_squared / (denominator + epsilon)

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
