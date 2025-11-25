# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Set random seeds for reproducibility across libraries (Inspired by Insp 1)
np.random.seed(42)
# Enable 64-bit floating-point precision for JAX computations for higher accuracy (Inspired by Insp 1)
jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2048  # From Insp 1, higher C2 with this N
    learning_rate: float = 0.01  # From Insp 1
    num_steps: int = 250000  # From Insp 1
    warmup_steps: int = 25000  # From Insp 1
    domain_length: float = 2.0  # From Insp 1, higher C2 with this L
    regularization_strength: float = 5e-8  # L2 penalty on 1st derivative (from Insp 1)
    curvature_reg_strength: float = 5e-9  # L2 penalty on 2nd derivative (from Insp 1)
    gradient_clip_value: float = 1.0  # From Insp 1


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.L = self.hypers.domain_length # For easier access to domain length

    def _objective_fn(self, f_values: jnp.ndarray, apply_regularization: bool = True) -> jnp.ndarray:
        """
        Computes the objective function C₂ = ||f★f||₂² / ||f★f||_{∞},
        assuming f is normalized such that ∫f dx = 1.
        Includes optional regularization for training.
        """
        # 1. Enforce f(x) >= 0 using relu (from Insp 1), ensuring float64
        f_non_negative = jax.nn.relu(f_values.astype(jnp.float64))

        # 2. Normalize f such that its numerical integral is 1.
        N = self.hypers.num_intervals
        L = self.L  # Domain length for f is [0, L)
        h_f = L / N  # Step size for f

        integral_f = jnp.sum(f_non_negative) * h_f
        # Add a small epsilon from jnp.finfo to prevent division by zero
        integral_f_safe = integral_f + jnp.finfo(jnp.float64).eps
        f_normalized = f_non_negative / integral_f_safe

        # 3. Compute autoconvolution g = f ★ f using FFT with scaling * h_f (from Insp 1)
        # The result g is supported on [0, 2L).
        M = 2 * N
        padded_f_norm = jnp.pad(f_normalized, (0, N))
        fft_f_norm = jnp.fft.fft(padded_f_norm)
        
        # Scaling by h_f as per problem description and Insp 1
        g_values = jnp.fft.ifft(fft_f_norm * fft_f_norm).real * h_f

        # 4. Calculate norms of the convolution g.
        # The convolution g has M points over domain [0, 2L).
        num_conv_points = len(g_values) # This is M = 2N

        # ||g||_inf = sup g(x)
        norm_inf_convolution = jnp.max(jnp.abs(g_values))
        # Add epsilon from jnp.finfo for numerical stability
        norm_inf_convolution_safe = norm_inf_convolution + jnp.finfo(jnp.float64).eps

        # ||g||_2^2 = ∫ g(x)^2 dx using robust piecewise-linear integral formula (from Insp 1)
        h_conv_integr = (2 * L) / (num_conv_points + 1) # Integration step size for convolution domain
        # Concatenate zeros at boundaries for accurate Simpson-like integration
        y_points = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), g_values, jnp.array([0.0], dtype=jnp.float64)])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv_integr / 3) * (y1**2 + y1 * y2 + y2**2))

        # 5. Calculate C2 constant
        # Since ∫f_normalized = 1, then ||f★f||₁ = 1.
        c2_ratio = l2_norm_squared / norm_inf_convolution_safe

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        loss = -c2_ratio

        # 6. Apply regularization during training (from Insp 1)
        if apply_regularization:
            if self.hypers.regularization_strength > 0:
                # Penalize L2 norm of the first derivative (encourages smoothness)
                smoothness_penalty = jnp.sum(jnp.diff(f_non_negative)**2)
                loss += self.hypers.regularization_strength * smoothness_penalty
            if self.hypers.curvature_reg_strength > 0:
                # Penalize L2 norm of the second derivative (encourages less curvature)
                curvature_penalty = jnp.sum(jnp.diff(jnp.diff(f_non_negative))**2)
                loss += self.hypers.curvature_reg_strength * curvature_penalty

        return loss

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        # The objective function is differentiated with respect to its first argument (f_values).
        # `apply_regularization` defaults to True, correctly signaling a training step.
        loss, grads = jax.value_and_grad(lambda f: self._objective_fn(f, apply_regularization=True))(f_values)
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
            # Further reduce end_value to allow for extremely fine-grained convergence over a long run.
            end_value=self.hypers.learning_rate * 1e-6,
        )
        # Apply gradient clipping and Adam optimizer (from Insp 1)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.gradient_clip_value),
            optax.adam(learning_rate=schedule)
        )

        key = jax.random.PRNGKey(42) # Keep for reproducibility

        N = self.hypers.num_intervals
        L = self.L # Use domain length from self.L
        # Ensure x_points are float64
        x_points = jnp.linspace(0.0, L, N, endpoint=False, dtype=jnp.float64)
        
        # Initialize with a sum of Gaussians + noise (from Insp 1)
        num_gaussians = 3
        # Scale means and std_devs to the new domain_length L=2.0
        amplitudes = jnp.array([1.5, 2.0, 1.5], dtype=jnp.float64)
        means = jnp.array([0.2 * L, 0.5 * L, 0.8 * L], dtype=jnp.float64)
        std_devs = jnp.array([0.07 * L, 0.07 * L, 0.07 * L], dtype=jnp.float64)

        f_initial = jnp.zeros_like(x_points, dtype=jnp.float64)
        for i in range(num_gaussians):
            f_initial += amplitudes[i] * jnp.exp(-((x_points - means[i])**2) / (2 * std_devs[i]**2))
            
        noise_key, _ = jax.random.split(key)
        # Generate noise with float64 precision
        noise = jax.random.uniform(noise_key, (N,), minval=-0.05, maxval=0.05, dtype=jnp.float64) * jnp.max(f_initial) * 0.1
        f_values = f_initial + noise
        f_values = jax.nn.relu(f_values) # Ensure initial non-negativity (from Insp 1)

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}, L: {self.L}"
        )
        
        # JIT compile the train step and the objective function without regularization for reporting
        train_step_jit = jax.jit(self.train_step)
        objective_fn_no_reg = jax.jit(lambda f: self._objective_fn(f, apply_regularization=False))

        loss = jnp.inf # Initialize with infinity for tracking minimum loss
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # For printing, report the pure C2 value without the regularization penalty
                current_c2 = -objective_fn_no_reg(f_values)
                print(f"Step {step:5d} | C2 ≈ {current_c2:.8f} (Loss with reg: {loss:.8f})")
        
        final_c2 = -objective_fn_no_reg(f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the final regularized loss, which is the actual loss minimized during training.
        final_regularized_loss = loss 
        return jax.nn.relu(f_values), final_c2, final_regularized_loss


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    # The run_optimization now returns the final regularized loss in addition to f and c2
    optimized_f, final_c2_val, final_regularized_loss_val = optimizer.run_optimization()

    # The 'loss' metric should be the final value of the *regularized* loss function
    # that was minimized during the optimization process, as specified in the prompt.
    loss_val = final_regularized_loss_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
