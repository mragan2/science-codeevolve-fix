# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 256 # Increased resolution for better approximation of the function
    learning_rate: float = 0.005 # Slightly reduced learning rate to prevent oscillations with higher N
    num_steps: int = 50000 # Increased steps for convergence with higher N
    warmup_steps: int = 2000 # Increased warmup steps proportionally


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

        N = self.hypers.num_intervals
        # Assuming f is defined on [0,1], the discretization step for f is dx_f.
        dx_f = 1.0 / N

        # Calculate integral of f for the denominator: (integral f)^2.
        # We assume f_values are samples for a piecewise constant function on [0,1].
        integral_f = jnp.sum(f_non_negative) * dx_f
        # Guard against division by zero if integral_f is very small.
        integral_f_squared = jnp.maximum(integral_f**2, 1e-12)

        # Unscaled discrete autoconvolution using FFT.
        # Padded_f length will be 2*N. Convolution length will be 2*N.
        # This implicitly assumes f is zero-padded to length 2N for aperiodic convolution.
        padded_f = jnp.pad(f_non_negative, (0, N)) # f_values is N long, pad with N zeros
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # The convolution `g = f*f` is supported on [0,2] if `f` is on [0,1].
        # If `convolution` has `2N` points, then the effective discretization step for `g` is dx_g = 2 / (2N) = 1/N.
        dx_g = 1.0 / N

        # Calculate L2-norm squared of the convolution (rigorous method).
        # Using piecewise linear integration for g^2.
        # Since f is piecewise constant, f*f is piecewise linear.
        # We assume g(0) = 0 and g(2) = 0 for accurate integration bounds, so pad the convolution array.
        # y_points will have len(convolution) + 2 points.
        # The number of segments for integration is len(convolution) + 1.
        # Each segment has length dx_g.
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((dx_g / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution.
        # Since f is non-negative, f*f is also non-negative, so jnp.abs is not strictly needed.
        norm_inf = jnp.max(convolution)

        # Calculate C2 ratio: ||f*f||_2^2 / ( (integral f)^2 * ||f*f||_inf ).
        denominator = integral_f_squared * norm_inf
        # Guard against division by zero for denominator.
        denominator = jnp.maximum(denominator, 1e-12)
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
