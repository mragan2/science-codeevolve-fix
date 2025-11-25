# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 512  # Further increased resolution for better accuracy, as suggested for groundbreaking results
    learning_rate: float = 0.01
    num_steps: int = 20000  # Increased steps for better convergence with higher N
    warmup_steps: int = 1000
    initial_support_width_f: float = 2.0  # Initial width for f, f is on [-L, L]


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, params: dict) -> jnp.ndarray:
        """
        Computes the objective function for C2 constant.
        C2 = ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        """
        f_values = params['f_values']
        # Ensure support_width_f is positive by using exp()
        support_width_f = jnp.exp(params['log_support_width_f'])

        N = self.hypers.num_intervals

        # Ensure f_values are non-negative
        f_non_negative = jax.nn.relu(f_values)

        # Calculate dx for integration. f is defined on [-support_width_f, support_width_f]
        # N points mean N-1 intervals.
        dx_f = (2.0 * support_width_f) / (N - 1) if N > 1 else 1.0

        # Calculate integral_f = ∫f dx
        integral_f = jnp.sum(f_non_negative) * dx_f

        # Condition for trivial f: if integral_f is too small, it means f is effectively zero.
        # This condition will be used later with jnp.where.
        integral_f_is_trivial = integral_f < 1e-9

        # Normalize f such that ∫f dx = 1.
        # This simplifies the (∫f)² term in the C2 denominator to 1.
        # Use jnp.where to prevent division by zero / inf propagation if integral_f is trivial.
        # If integral_f is trivial, set f_normalized to zeros to avoid NaNs in subsequent convolution.
        f_normalized = jnp.where(
            integral_f_is_trivial,
            jnp.zeros_like(f_non_negative),
            f_non_negative / integral_f
        )

        # Autoconvolution of normalized f using FFT
        # The convolution of N points with N points results in 2N-1 points.
        # Pad f_normalized to length 2N-1 for non-periodic convolution.
        conv_len = 2 * N - 1
        padded_f_normalized = jnp.pad(f_normalized, (0, N - 1))

        # Perform FFT and IFFT. Using n=conv_len ensures correct length.
        fft_f = jnp.fft.fft(padded_f_normalized, n=conv_len)
        convolution_samples = jnp.fft.ifft(fft_f * fft_f).real

        # The convolution g = f ★ f is supported on [-2*support_width_f, 2*support_width_f].
        # The step size for convolution samples is consistent with dx_f.
        integral_g_dx = dx_f

        # Calculate ||g||_2^2 using the piecewise-linear integral method (Simpson's-like rule)
        # Add zeros at ends for integration, assuming g goes to zero outside the sampled range.
        y_points_g = jnp.concatenate([jnp.array([0.0]), convolution_samples, jnp.array([0.0])])
        y1_g, y2_g = y_points_g[:-1], y_points_g[1:]
        l2_norm_squared_g = jnp.sum((integral_g_dx / 3.0) * (y1_g**2 + y1_g * y2_g + y2_g**2))

        # Calculate ||g||_inf = sup|g(x)|
        norm_inf_g = jnp.max(jnp.abs(convolution_samples))

        # Condition for trivial g: if norm_inf_g is too small, it means g is effectively zero.
        # This condition will be used later with jnp.where.
        norm_inf_g_is_trivial = norm_inf_g < 1e-9

        # C2 = ||g||₂² / ((∫f)² ||g||_{∞})
        # Since f was normalized such that ∫f dx = 1, (∫f)² = 1.
        # Guard the denominator of C2_ratio to prevent division by exactly zero,
        # which can lead to NaN/inf. Use jnp.maximum to ensure a minimum positive value.
        c2_ratio_denominator = jnp.maximum(norm_inf_g, 1e-9)
        c2_ratio = l2_norm_squared_g / c2_ratio_denominator

        # Combine conditions for invalid states. If either f or g is trivial,
        # the C2 constant is ill-defined or zero, leading to an infinite loss (penalty).
        is_invalid_state = integral_f_is_trivial | norm_inf_g_is_trivial

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # If in an invalid state, return positive infinity as a penalty.
        objective_value = jnp.where(is_invalid_state, jnp.array(jnp.inf), -c2_ratio)

        # Add L2 regularization to f_values to encourage smoother solutions and prevent extreme values.
        # Also, add a small L2 regularization to log_support_width_f to help prevent it from diverging.
        # These regularization terms are added to the objective (loss) function.
        f_l2_reg_strength = 1e-5
        L_f_reg_strength = 1e-6 # Regularize the log value directly

        l2_f_values_norm = jnp.sum(f_values**2)
        l2_log_L_f_norm = params['log_support_width_f']**2

        objective_value += f_l2_reg_strength * l2_f_values_norm
        objective_value += L_f_reg_strength * l2_log_L_f_norm

        return objective_value

    def train_step(self, params: dict, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

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

        key = jax.random.PRNGKey(42)  # Fixed random seed for reproducibility

        # Initialize f_values to approximate a triangle function, a known good candidate.
        # The triangle function f(x) = max(0, 1 - |x|/L_base) has maximum 1 at x=0
        # and support [-L_base, L_base].
        N = self.hypers.num_intervals
        initial_L_f = self.hypers.initial_support_width_f
        
        # Grid for f_values from -initial_L_f to initial_L_f
        x_grid_f_init = jnp.linspace(-initial_L_f, initial_L_f, N)
        
        # Initialize as a triangle function that spans from -initial_L_f/2 to initial_L_f/2.
        # So its base is initial_L_f.
        triangle_f_init = jnp.maximum(0.0, 1.0 - jnp.abs(x_grid_f_init) / (initial_L_f / 2.0))
        
        # Initialize parameters for optimization: f_values and log_support_width_f
        initial_params = {
            'f_values': triangle_f_init,
            'log_support_width_f': jnp.log(initial_L_f)
        }

        opt_state = self.optimizer.init(initial_params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        current_params = initial_params
        loss = jnp.inf  # Initialize loss
        for step in range(self.hypers.num_steps):
            current_params, opt_state, loss = train_step_jit(current_params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # Recompute C2 and L_f for logging to get actual values from current_params
                actual_c2 = -self._objective_fn(current_params)
                current_L_f = jnp.exp(current_params['log_support_width_f'])
                print(f"Step {step:5d} | C2 ≈ {actual_c2:.8f} | L_f ≈ {current_L_f:.4f}")

        final_c2 = -self._objective_fn(current_params)
        final_f_values = jax.nn.relu(current_params['f_values'])
        final_L_f = jnp.exp(current_params['log_support_width_f'])
        print(f"Final C2 lower bound found: {final_c2:.8f} with L_f = {final_L_f:.4f}")
        return final_f_values, float(final_c2), float(final_L_f)


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    # The run_optimization now returns final_L_f as well
    optimized_f, final_c2_val, final_L_f_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    # Return final_L_f_val as an additional metric if needed, or just keep n_points
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
