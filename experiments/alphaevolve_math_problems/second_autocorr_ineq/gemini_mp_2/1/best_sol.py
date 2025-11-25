# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 4096 # Significantly increased resolution
    learning_rate: float = 0.01
    num_steps: int = 150000 # Increased steps for higher resolution and L optimization
    warmup_steps: int = 1000
    lambda_smooth: float = 2e-7 # Slightly increased for better numerical stability at high resolution


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    Now also optimizes the support length L and includes smoothness regularization.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _compute_c2_from_params(self, params: dict) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Helper function to compute the C2 ratio and intermediate values from parameters without regularization.
        Returns: (c2_ratio, integral_f, norm_inf_conv)
        """
        f_values = params['f_values']
        L_param = params['L']
        L = jnp.maximum(L_param, 1e-4) # Ensure L is positive

        f_non_negative = jax.nn.relu(f_values)

        N = self.hypers.num_intervals
        dx_f = L / N

        integral_f = dx_f * (jnp.sum(f_non_negative) - 0.5 * (f_non_negative[0] + f_non_negative[-1]))

        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        g_conv_unscaled = jnp.fft.ifft(fft_f * fft_f).real
        g_conv = g_conv_unscaled * dx_f

        dx_conv = (2 * L) / len(g_conv)

        norm_1_conv = integral_f**2
        g_conv_sq = g_conv**2
        l2_norm_squared_conv = dx_conv * (jnp.sum(g_conv_sq) - 0.5 * (g_conv_sq[0] + g_conv_sq[-1]))
        norm_inf_conv = jnp.max(g_conv)

        denominator = norm_1_conv * norm_inf_conv
        c2_ratio = l2_norm_squared_conv / (denominator + 1e-9)

        return c2_ratio, integral_f, norm_inf_conv

    def _objective_fn(self, params: dict) -> jnp.ndarray:
        """
        Computes the objective function, including smoothness regularization.
        """
        c2_ratio, integral_f, norm_inf_conv = self._compute_c2_from_params(params)
        f_non_negative = jax.nn.relu(params['f_values']) # Used for smoothness penalty

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        loss = -c2_ratio

        # Add regularization for smoothness of f
        smoothness_penalty = self.hypers.lambda_smooth * jnp.sum(jnp.diff(f_non_negative)**2)
        loss = loss + smoothness_penalty

        # Handle trivial/invalid cases (e.g., f is all zeros) in a JIT-compatible way.
        is_invalid = (integral_f <= 1e-9) | (norm_inf_conv <= 1e-9)
        return jnp.where(is_invalid, jnp.inf, loss)

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
            end_value=self.hypers.learning_rate * 5e-5, # Slightly higher end_value to prevent premature stalling
        )
        # Ensure numpy randomness is also seeded for reproducibility
        np.random.seed(42)
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        # Initialize f_values with a triangular pulse, a known good starting point
        N = self.hypers.num_intervals
        initial_L = jnp.array(1.0) # L is now an optimizable parameter, initialized to 1.0
        x_points = jnp.linspace(0.0, initial_L, N, endpoint=False) # Sample points for f
        mid_point_idx = N // 2
        f_values_initial = jnp.zeros(N)
        # Construct a triangular pulse: f(x) = x for x in [0, L/2], f(x) = L-x for x in [L/2, L]
        f_values_initial = f_values_initial.at[:mid_point_idx].set(x_points[:mid_point_idx])
        f_values_initial = f_values_initial.at[mid_point_idx:].set(initial_L - x_points[mid_point_idx:])
        # Add a small constant to ensure non-zero initial values and avoid issues with ReLU
        f_values_initial = f_values_initial + 1e-6

        # Initialize parameters as a PyTree (dictionary) for optax
        params = {'f_values': f_values_initial, 'L': initial_L}

        opt_state = self.optimizer.init(params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            params, opt_state, loss = train_step_jit(params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # Retrieve current L from params for printing (clamped for display)
                current_L = jnp.maximum(params['L'], 1e-4)
                # Calculate C2 without regularization for reporting
                current_c2, _, _ = self._compute_c2_from_params(params) # Unpack the returned tuple
                print(f"Step {step:5d} | C2 ≈ {current_c2:.8f} | L ≈ {current_L:.4f}")

        final_c2_val, _, _ = self._compute_c2_from_params(params) # Final evaluation without regularization
        final_L_val = jnp.maximum(params['L'], 1e-4) # Get final L (clamped)
        print(f"Final C2 lower bound found: {final_c2_val:.8f} with optimized L: {final_L_val:.4f}")
        # Return f_values from the optimized parameters
        return jax.nn.relu(params['f_values']), final_c2_val


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
