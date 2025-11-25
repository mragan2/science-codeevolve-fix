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
    learning_rate: float = 0.01
    num_steps: int = 50000  # Increased steps for convergence with higher resolution
    warmup_steps: int = 2000 # Adjusted warmup steps for longer training duration


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
        # Assume f is defined on [0, L]. For a unitless constant, L can be 1.
        # So f_values are N samples of f on [0, 1], with step size h_f = 1/N.
        h_f = 1.0 / N

        # Calculate integral of f for the L1-norm of convolution: (∫f)²
        # For piecewise constant f, ∫f ≈ h_f * sum(f_values).
        integral_f = h_f * jnp.sum(f_non_negative)
        # Ensure integral_f is not zero to avoid division by zero.
        integral_f = jnp.where(integral_f < 1e-9, 1e-9, integral_f)
        norm_1_denom = integral_f**2

        # Unscaled discrete autoconvolution using FFT.
        # Pad f_values to 2N length for convolution.
        # If f is on [0,1] with N points, f*f is on [0,2] with 2N points.
        padded_f = jnp.pad(f_non_negative, (0, N)) # Padded to 2N points (length 2N)
        fft_f = jnp.fft.fft(padded_f)
        discrete_convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Scale discrete convolution to approximate continuous convolution integral.
        # g(x) ≈ h_f * g_discrete_j
        convolution = h_f * discrete_convolution

        # Calculate L2-norm squared of the convolution.
        # The convolution 'convolution' is defined over [0, 2L] = [0, 2].
        # It has num_conv_points = 2N samples.
        # So the step size for integration is h_conv = (2L) / (2N) = L/N = h_f.
        num_conv_points = len(convolution) # This will be 2N
        h_conv = 1.0 / N # Correct step size for integration over [0, 2] with 2N points.

        # Using Simpson's rule for integration of convolution squared (piecewise linear approximation).
        # We append 0s at boundaries, assuming convolution goes to zero outside [0, 2L].
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution.
        norm_inf = jnp.max(jnp.abs(convolution))
        # Ensure norm_inf is not zero.
        norm_inf = jnp.where(norm_inf < 1e-9, 1e-9, norm_inf)

        # Calculate C2 ratio.
        denominator = norm_1_denom * norm_inf
        # Ensure denominator is not zero.
        denominator = jnp.where(denominator < 1e-12, 1e-12, denominator)
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
                # Calculate and print current C2 based on the current f_values
                current_c2 = -self._objective_fn(f_values) # Re-evaluate C2 for accurate reporting
                print(f"Step {step:5d} | C2 ≈ {current_c2:.8f}")

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
