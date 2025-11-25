# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    # Default values, these will be overridden by stage_configs in run_optimization.
    num_intervals: int = 512
    learning_rate: float = 0.01
    num_steps: int = 15000
    warmup_steps: int = 1500
    regularization_strength: float = 1e-7  # L2 on first derivative (smoothness)
    curvature_reg_strength: float = 1e-8  # L2 on second derivative (curvature)
    gradient_clip_value: float = 1.0  # Global norm for gradient clipping


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.L = 1.0 # Define the compact support length L for numerical calculations

    def _objective_fn(self, f_values: jnp.ndarray, current_regularization_strength: float = None, current_curvature_reg_strength: float = None) -> jnp.ndarray:
        """
        Computes the objective function for maximizing C2, adhering strictly
        to the mathematical framework and including regularization terms.
        """
        f_non_negative = jax.nn.relu(f_values)

        N = self.hypers.num_intervals
        dx_f = self.L / N

        # Use the key simplification: ||f ★ f||₁ = (∫f)²
        integral_f = jnp.sum(f_non_negative) * dx_f
        norm_1 = integral_f ** 2

        # Autoconvolution g = f ★ f, using FFT.
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        # CRITICAL FIX: Scale the FFT output by dx_f to approximate the continuous convolution integral.
        convolution = jnp.fft.ifft(fft_f * fft_f).real * dx_f

        # Calculate L2-norm squared of the convolution (rigorous piecewise linear integration)
        # Inspired by Insp. 3: Add boundary zeros for g(0)=0 and g(2L)=0
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        dx_conv = (2 * self.L) / (len(y_points) - 1)
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((dx_conv / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution
        norm_inf = jnp.max(convolution)
        
        # Calculate C2 ratio with numerical stability guards (inspired by Insp. 3)
        epsilon = 1e-12
        denominator = norm_1 * norm_inf
        denominator = jnp.where(denominator > epsilon, denominator, epsilon)

        c2_ratio = l2_norm_squared / denominator
        loss = -c2_ratio

        # Apply regularization (inspired by Insp. 3)
        reg_strength = self.hypers.regularization_strength if current_regularization_strength is None else current_regularization_strength
        curv_reg_strength = self.hypers.curvature_reg_strength if current_curvature_reg_strength is None else current_curvature_reg_strength

        if reg_strength > 0:
            smoothness_penalty = jnp.sum(jnp.diff(f_non_negative)**2)
            loss += reg_strength * smoothness_penalty
        if curv_reg_strength > 0:
            curvature_penalty = jnp.sum(jnp.diff(jnp.diff(f_non_negative))**2)
            loss += curv_reg_strength * curvature_penalty
        
        return loss

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(
            f_values, self.hypers.regularization_strength, self.hypers.curvature_reg_strength
        )
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def _interpolate_f_values(self, f_values_old: jnp.ndarray, N_old: int, N_new: int) -> jnp.ndarray:
        """
        Interpolates f_values from N_old points to N_new points for multi-stage optimization.
        """
        if N_old == N_new:
            return f_values_old
        x_old = jnp.linspace(0, self.L, N_old, endpoint=False)
        x_new = jnp.linspace(0, self.L, N_new, endpoint=False)
        f_values_new = jnp.interp(x_new, x_old, f_values_old, left=0.0, right=0.0)
        return jax.nn.relu(f_values_new)

    def run_optimization(self):
        """Sets up and runs the full multi-stage optimization process."""
        stage_configs = [
            Hyperparameters(num_intervals=512, learning_rate=0.01, num_steps=20000, warmup_steps=2000,
                            regularization_strength=5e-7, curvature_reg_strength=2e-8, gradient_clip_value=1.0),
            Hyperparameters(num_intervals=1024, learning_rate=0.005, num_steps=15000, warmup_steps=1500,
                            regularization_strength=2e-7, curvature_reg_strength=1e-8, gradient_clip_value=0.8),
            Hyperparameters(num_intervals=2048, learning_rate=0.001, num_steps=10000, warmup_steps=1000,
                            regularization_strength=1e-7, curvature_reg_strength=5e-9, gradient_clip_value=0.5),
            Hyperparameters(num_intervals=4096, learning_rate=0.0005, num_steps=10000, warmup_steps=1000,
                            regularization_strength=5e-8, curvature_reg_strength=2e-9, gradient_clip_value=0.3),
        ]

        # Initialize f_values with a Gaussian pulse (inspired by Insp. 3)
        initial_N = stage_configs[0].num_intervals
        x_points_initial = jnp.linspace(0.0, self.L, initial_N, endpoint=False)
        mu, sigma = self.L / 2.0, self.L / 6.0
        f_values = jnp.exp(-((x_points_initial - mu)**2) / (2 * sigma**2))

        final_c2_val, final_regularized_loss = 0.0, jnp.inf
        N_previous_stage = initial_N
        
        for stage_idx, stage_hypers in enumerate(stage_configs):
            print(f"\n--- Starting Stage {stage_idx + 1} ---")
            self.hypers = stage_hypers

            if stage_idx > 0:
                f_values = self._interpolate_f_values(f_values, N_previous_stage, self.hypers.num_intervals)
            
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=self.hypers.learning_rate,
                warmup_steps=self.hypers.warmup_steps,
                decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
                end_value=self.hypers.learning_rate * 1e-4,
            )
            # Optimizer with gradient clipping (inspired by Insp. 3)
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.hypers.gradient_clip_value),
                optax.adam(learning_rate=schedule)
            )
            opt_state = self.optimizer.init(f_values)

            print(f"Stage {stage_idx+1}: N={self.hypers.num_intervals}, Steps={self.hypers.num_steps}, LR={self.hypers.learning_rate}, Clip={self.hypers.gradient_clip_value}")
            train_step_jit = jax.jit(self.train_step)

            for step in range(self.hypers.num_steps):
                f_values, opt_state, loss = train_step_jit(f_values, opt_state)
                if step % (self.hypers.num_steps // 10) == 0 or step == self.hypers.num_steps - 1:
                    c2_for_display = -self._objective_fn(f_values, 0.0, 0.0)
                    print(f"  Stage {stage_idx+1} Step {step:5d} | C2 ≈ {c2_for_display:.8f} (Loss: {loss:.8f})")

            final_c2_val = -self._objective_fn(f_values, 0.0, 0.0)
            final_regularized_loss = loss
            print(f"--- End of Stage {stage_idx+1} | C2: {final_c2_val:.8f} ---")
            N_previous_stage = self.hypers.num_intervals
            
        print(f"\nFinal C2 lower bound found after all stages: {final_c2_val:.8f}")
        return jax.nn.relu(f_values), final_c2_val, final_regularized_loss


def run():
    """Entry point for running the optimization."""
    # Set numpy random seed for any non-JAX random operations for full determinism.
    np.random.seed(42)
    
    # Initial hyperparameters are placeholders; the multi-stage configs are defined inside run_optimization.
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    
    # The run_optimization now returns the final regularized loss in addition to f and c2.
    optimized_f, final_c2_val, final_regularized_loss_val = optimizer.run_optimization()

    # The 'loss' metric should be the final value of the *regularized* loss function
    # that was minimized during the optimization process.
    loss_val = final_regularized_loss_val
    f_values_np = np.array(optimized_f)

    # Return the num_intervals from the optimizer's state after the final stage.
    return f_values_np, float(final_c2_val), float(loss_val), optimizer.hypers.num_intervals


# EVOLVE-BLOCK-END
