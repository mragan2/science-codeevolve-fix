# EVOLVE-BLOCK-START
import jax
# Enable float64 (double precision) for JAX computations.
# This is crucial for achieving high precision in mathematical constant discovery.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    # Inspired by high-performing configurations from analysis (e.g., Inspiration Program 3)
    # Increased resolution and training steps are critical for finding better constants.
    num_intervals: int = 2000
    learning_rate: float = 0.0025
    num_steps: int = 500000
    warmup_steps: int = 20000


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using a mathematically rigorous, scale-invariant formulation.
        This implementation is synthesized from the best practices of the inspiration programs,
        ensuring correct scaling for continuous norms approximated from discrete data.
        
        - f is a piecewise constant function on [0, 1].
        - f*f is a piecewise linear function on [0, 2].
        """
        f_non_negative = jax.nn.relu(f_values)
        N_f = self.hypers.num_intervals

        # --- Norm 1 Term: (∫f)² ---
        # We use the analytical simplification ||f*f||₁ = (∫f)², which is exact for f>=0.
        # The integral is approximated by a Riemann sum.
        dx_f = 1.0 / N_f
        integral_f = dx_f * jnp.sum(f_non_negative)
        norm_1 = integral_f**2

        # --- Convolution for Norms ∞ and 2 ---
        # The result of IFFT(FFT*FFT) is a discrete convolution. To approximate the values
        # of the continuous convolution g(x) = ∫f(t)f(x-t)dt, we must scale by dx_f.
        # g(n*dx) ≈ dx * Σ_k f(k*dx)f((n-k)*dx)
        padded_f = jnp.pad(f_non_negative, (0, N_f))
        fft_f = jnp.fft.fft(padded_f)
        convolution_continuous = jnp.fft.ifft(fft_f * fft_f).real * dx_f

        # --- Norm ∞ Term: sup|f*f| ---
        # Since f>=0, f*f is also non-negative.
        norm_inf = jnp.max(convolution_continuous)

        # --- Norm 2 Term: ∫(f*f)² dx ---
        # f*f is piecewise linear. We use the exact integral formula for a squared
        # piecewise linear function to achieve high accuracy.
        # The convolution is on [0, 2]. Number of points is 2*N_f.
        num_conv_points = len(convolution_continuous)
        dx_conv = 2.0 / num_conv_points  # This correctly evaluates to 1.0/N_f = dx_f
        
        # We append a zero for the point f*f(2)=0 to correctly integrate the last segment.
        y_points_for_integral = jnp.concatenate([convolution_continuous, jnp.array([0.0], dtype=convolution_continuous.dtype)])
        y1, y2 = y_points_for_integral[:-1], y_points_for_integral[1:]
        # Formula for ∫g² where g is piecewise linear: Σ (h/3)*(y1² + y1*y2 + y2²)
        l2_norm_squared = jnp.sum((dx_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # --- C2 Ratio ---
        denominator = norm_1 * norm_inf
        # Add a small epsilon for numerical stability, especially at the start of training.
        c2_ratio = l2_norm_squared / (denominator + 1e-12)

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
        # Initialize f_values with small positive random numbers and float64 dtype.
        # This prevents large initial gradients and aligns with high-precision computation.
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,), dtype=jnp.float64) * 0.01

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
