# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Increased resolution is crucial for finding more complex functions.
    # Increased resolution is crucial for finding more complex functions.
    # Increasing num_intervals significantly to allow for finer step function approximations.
    num_intervals: int = 1024  # Increased from 200 to provide higher resolution
    learning_rate: float = 0.005
    # More steps are needed for the increased number of parameters to converge.
    num_steps: int = 200000 # Increased from 100000 for more thorough optimization
    warmup_steps: int = 10000 # Increased proportionally from 5000
    tv_lambda: float = 1e-5 # Added Total Variation regularization strength to encourage step-like functions


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, half_f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using physically-scaled norms and enforcing
        even function symmetry.
        """
        # Enforce symmetry by mirroring the half-array of parameters.
        # This reduces the search space and is a common assumption for this problem.
        f_values = jnp.concatenate([half_f_values, half_f_values[::-1]])
        # Use softplus for non-negativity: it's smoother than relu and always provides gradients,
        # which can help prevent parameters from getting stuck at zero.
        f_non_negative = jax.nn.softplus(f_values)

        N = self.hypers.num_intervals
        # Define discretization step, assuming f is supported on a domain of length 1.
        dx = 1.0 / N

        # Use the fundamental property ||f*f||_1 = (integral(f))^2 for the L1 norm.
        # This is more accurate and stable than integrating the convolved result.
        integral_f = jnp.sum(f_non_negative) * dx
        norm_1_conv = integral_f**2

        # Unscaled discrete autoconvolution via FFT
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        g_fft = jnp.fft.ifft(fft_f * fft_f).real

        # Scale the convolution result to approximate the continuous convolution integral.
        # g(x) = integral(f(t)f(x-t)dt) approx sum(f_k * f_{n-k} * dx)
        g_continuous = g_fft * dx

        # Calculate infinity-norm of the scaled convolution
        norm_inf = jnp.max(g_continuous)

        # Calculate L2-norm squared of the convolution using a rigorous
        # piecewise-linear integration, now correctly scaled by dx.
        # The convolved function g is supported on a domain of length 2.
        # The 2N points are spaced by dx.
        y_points = jnp.concatenate([jnp.array([0.0]), g_continuous, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((dx / 3) * (y1**2 + y1 * y2 + y2**2))

        # The denominator is scale-invariant if calculated this way.
        denominator = norm_1_conv * norm_inf

        # Add a small epsilon to prevent division by zero during early optimization steps.
        c2_ratio = l2_norm_squared / (denominator + 1e-12)

        # Add Total Variation (TV) regularization to encourage piecewise constant functions.
        # The TV norm is the sum of absolute differences between adjacent points.
        tv_norm = jnp.sum(jnp.abs(jnp.diff(f_non_negative)))
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add the TV regularization term to the objective.
        return -c2_ratio + self.hypers.tv_lambda * tv_norm

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        assert self.hypers.num_intervals % 2 == 0, "num_intervals must be even for symmetry"

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        # Initialize only half of the function values, since we enforce symmetry.
        num_half_intervals = self.hypers.num_intervals // 2
        # Initialize only half of the function values, since we enforce symmetry.
        num_half_intervals = self.hypers.num_intervals // 2
        
        # Initialize with a Gaussian-like shape to provide a more structured starting point.
        # This can help the optimizer find better local minima compared to uniform random.
        # The 'x' values for the half-function are from 0 to (num_half_intervals-1)*dx.
        dx = 1.0 / self.hypers.num_intervals
        x_half = jnp.linspace(0.0, (num_half_intervals - 1) * dx, num_half_intervals)
        
        # Choose sigma such that the Gaussian has reasonable decay over the domain [0, 0.5].
        # A sigma of 0.4 provides a broader Gaussian, which is a better initial guess
        # for C2, yielding values closer to 0.886 for the initial state.
        sigma = 0.4 # Increased from 0.18 for a broader initial Gaussian
        # Initialize raw parameters with a Gaussian shape.
        # softplus(x) will be applied later to ensure non-negativity.
        half_f_values = jnp.exp(- (x_half**2) / (2 * sigma**2))
        
        # Normalize the initial function so its peak (before softplus) is 1.0.
        # This helps with initial stability and ensures the function is within a reasonable scale.
        half_f_values /= jnp.max(half_f_values) + 1e-6 # Add epsilon for numerical stability

        opt_state = self.optimizer.init(half_f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals} (Optimizing {num_half_intervals} symmetric params)"
        )
        print(f"Steps: {self.hypers.num_steps}, LR: {self.hypers.learning_rate}")

        # The train step now operates on the half-array of parameters.
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            half_f_values, opt_state, loss = train_step_jit(half_f_values, opt_state)
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")

        final_c2 = -self._objective_fn(half_f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")

        # Reconstruct the full symmetric function for returning.
        full_f_values = jnp.concatenate([half_f_values, half_f_values[::-1]])
        return jax.nn.relu(full_f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
