# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2048 # Doubled resolution to capture even finer details for potentially higher C2 (from Inspiration 2).
    learning_rate: float = 0.003 # Maintain current learning rate, proven stable for this problem (from Inspiration 2).
    num_steps: int = 500000 # Doubled steps to allow for convergence with twice the parameters at higher resolution (from Inspiration 2).
    warmup_steps: int = 50000 # Doubled warmup steps proportionally (from Inspiration 2).
    tv_regularization_coeff: float = 5e-6 # Added TV regularization to encourage step-like solutions (from Inspiration 2).


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using a mathematically rigorous and consistent integral method.
    The function f is represented by its parameters f_params, and the actual
    function values are f_actual = f_params**2 to enforce non-negativity.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function (negative C2) using the full, stable formula,
        plus a Total Variation regularization term.
        C₂ = ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        f_params are the parameters being optimized, f_actual = f_params**2 (from Inspiration 2).
        """
        # Enforce non-negativity by squaring the parameters, provides smoother gradients (from Inspiration 2).
        f_actual = jnp.square(f_params)

        N = self.hypers.num_intervals
        # Step size for f, assuming f is defined on [0, 1) and sampled at N points (from Inspiration 2).
        h_f = 1.0 / N

        # 1. Calculate integral of f using the trapezoidal rule for better accuracy.
        # Consistent with linspace(..., endpoint=False) from Inspiration 2.
        integral_f = (jnp.sum(f_actual) - 0.5 * (f_actual[0] + f_actual[-1])) * h_f
        integral_f = jnp.maximum(integral_f, 1e-9) # Add epsilon for stability

        # 2. Compute autoconvolution g = f ★ f using FFT.
        padded_f = jnp.pad(f_actual, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution_raw = jnp.fft.ifft(fft_f * fft_f).real

        # The scaling from discrete (FFT) to continuous convolution is h_f.
        # continuous_conv_values = h_f * (L_pad * IFFT(...)) = h_f * (2N * IFFT(...)) = 2.0 * IFFT(...)
        g_values = 2.0 * convolution_raw

        # 3. Calculate norms of the convolution g = f ★ f.
        # ||g||₂²: L2-norm squared of g.
        # Use a highly accurate integration formula for a piecewise linear function.
        # g is supported on [0, 2], so g(0)=0 and g(2)=0.
        # `g_values` contains samples g(k*h_f) for k=0..2N-1.
        # Construct `y_points` for integration, explicitly setting g(0)=0 and g(2)=0 (from Inspiration 1).
        g_for_integration = jnp.concatenate([jnp.array([0.0]), g_values[1:], jnp.array([0.0])])
        
        # `g_for_integration` now has `2N+1` points defining `2N` segments over domain `[0,2]`.
        # So the step size for integration is `h_g = 2.0 / (2N) = 1.0 / N = h_f`.
        h_g = h_f 

        y1, y2 = g_for_integration[:-1], g_for_integration[1:]
        l2_norm_squared = jnp.sum((h_g / 3) * (y1**2 + y1 * y2 + y2**2))

        # ||g||_{∞}: Infinity-norm of g.
        norm_inf = jnp.max(g_values)
        norm_inf = jnp.maximum(norm_inf, 1e-9) # Add epsilon for stability

        # 4. Calculate C2 using the full, stable formula.
        denominator = (integral_f**2) * norm_inf
        c2_ratio = l2_norm_squared / jnp.maximum(denominator, 1e-12)

        # 5. Add Total Variation (TV) regularization to the loss (from Inspiration 2).
        # This encourages piecewise constant solutions, which are known to be effective.
        tv_loss = jnp.sum(jnp.abs(jnp.diff(f_actual)))
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # The TV regularization term is added to the loss.
        return -c2_ratio + self.hypers.tv_regularization_coeff * tv_loss

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
            end_value=self.hypers.learning_rate * 1e-4,
        )
        # Removed gradient clipping based on empirical evidence from Inspiration 2,
        # which achieved higher C2 without clipping than with it in previous runs.
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        
        # Initialize f_actual_initial with a U-shape, which proved effective.
        # Assumes f is defined over [0, 1) and sampled at N points (from Inspiration 2).
        N = self.hypers.num_intervals
        x_grid = jnp.linspace(0.0, 1.0, N, endpoint=False, dtype=jnp.float32)
        
        # A quadratic U-shape initialization: (x-0.5)^2 + 0.1
        f_actual_initial = (x_grid - 0.5)**2 + 0.1 
        
        # Convert initial f_actual values to f_params for optimization (f_actual = f_params**2)
        f_params_initial = jnp.sqrt(f_actual_initial)
        f_params_initial = jnp.array(f_params_initial, dtype=jnp.float32) # Ensure correct dtype

        opt_state = self.optimizer.init(f_params_initial)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        f_params_current = f_params_initial # Initialize the current parameters
        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_params_current, opt_state, loss = train_step_jit(f_params_current, opt_state)
            if step % 2500 == 0 or step == self.hypers.num_steps - 1: # Increased print frequency for longer runs
                # To report the true C2 value, we must calculate it from the current f_params
                # by subtracting the TV penalty from the composite loss.
                f_actual_current = jnp.square(f_params_current)
                tv_penalty = self.hypers.tv_regularization_coeff * jnp.sum(jnp.abs(jnp.diff(f_actual_current)))
                c2_val_at_step = tv_penalty - loss # C2 = TV_coeff * TV_loss - (-C2_objective)
                print(f"Step {step:6d} | C2 ≈ {c2_val_at_step:.8f} | Total Loss: {loss:.4f}")

        # Compute final C2 with the optimized f_params.
        # We must re-calculate it without the regularization term for the true C2 value.
        loss_final = self._objective_fn(f_params_current) # This loss includes TV
        f_actual_final = jnp.square(f_params_current)
        tv_penalty_final = self.hypers.tv_regularization_coeff * jnp.sum(jnp.abs(jnp.diff(f_actual_final)))
        final_c2 = tv_penalty_final - loss_final # True C2 without regularization
        
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the actual function values (f_params**2)
        return jnp.square(f_params_current), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    # The loss_val should directly reflect the negative C2, without TV regularization for clarity.
    # The final_c2_val already has TV regularization subtracted.
    loss_val = -final_c2_val 
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
