# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for a single stage of the optimization process."""
    num_intervals: int
    learning_rate: float
    num_steps: int
    warmup_steps: int
    regularization_strength: float
    curvature_reg_strength: float
    gradient_clip_value: float


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using a multi-stage optimization curriculum inspired by Inspiration 3.
    """

    def __init__(self, hypers: Hyperparameters):
        # `hypers` is now a placeholder; it will be updated for each stage.
        self.hypers = hypers
        self.L = 1.0 # Define the compact support length L for numerical calculations

    def _objective_fn(self, f_values: jnp.ndarray, current_regularization_strength: float = None) -> jnp.ndarray:
        """
        Computes the objective function. This function is stage-agnostic and uses
        the currently set self.hypers for its parameters (like N).
        """
        f_non_negative = jax.nn.relu(f_values)

        N = self.hypers.num_intervals
        L = self.L
        h_f = L / N

        integral_f = jnp.sum(f_non_negative) * h_f
        integral_f = jnp.where(integral_f > 1e-9, integral_f, 1e-9)
        norm_1 = integral_f ** 2

        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real * h_f

        num_conv_points = len(convolution)
        h_conv_integr = (2 * L) / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv_integr / 3) * (y1**2 + y1 * y2 + y2**2))

        norm_inf = jnp.max(jnp.abs(convolution))
        norm_inf = jnp.where(norm_inf > 1e-9, norm_inf, 1e-9)

        denominator = norm_1 * norm_inf
        denominator = jnp.where(denominator > 1e-18, denominator, 1e-18)
        c2_ratio = l2_norm_squared / denominator

        loss = -c2_ratio

        if current_regularization_strength is None:
            if self.hypers.regularization_strength > 0:
                loss += self.hypers.regularization_strength * jnp.sum(jnp.diff(f_non_negative)**2)
            if self.hypers.curvature_reg_strength > 0:
                loss += self.hypers.curvature_reg_strength * jnp.sum(jnp.diff(jnp.diff(f_non_negative))**2)
        elif current_regularization_strength > 0:
            loss += current_regularization_strength * jnp.sum(jnp.diff(f_non_negative)**2)

        return loss

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def _interpolate_f_values(self, f_values_old: jnp.ndarray, N_old: int, N_new: int) -> jnp.ndarray:
        """Interpolates f_values from N_old points to N_new points for multi-stage refinement."""
        if N_old == N_new:
            return f_values_old
        x_old = jnp.linspace(0, self.L, N_old, endpoint=False)
        x_new = jnp.linspace(0, self.L, N_new, endpoint=False)
        f_values_new = jnp.interp(x_new, x_old, f_values_old, left=0.0, right=0.0)
        return jax.nn.relu(f_values_new)

    def run_optimization(self):
        """Sets up and runs the full multi-stage optimization process."""
        # Adopted stage configurations from Inspiration Program 1, which achieved the highest C2.
        # These stages progressively increase resolution and reduce regularization for fine-tuning.
        stage_configs = [
            # Stage 1: Initial exploration with slightly increased regularization, to potentially guide
            # the initial broad search more effectively towards smoother function shapes.
            Hyperparameters(num_intervals=1024, learning_rate=0.01, num_steps=50000, warmup_steps=2500,
                            regularization_strength=5e-7, curvature_reg_strength=1e-7, gradient_clip_value=1.0),
            # Stage 2: Increased resolution, slightly lower LR, reduced regularization.
            Hyperparameters(num_intervals=2048, learning_rate=0.005, num_steps=40000, warmup_steps=2000,
                            regularization_strength=2e-7, curvature_reg_strength=1e-8, gradient_clip_value=0.75),
            # Stage 3: Higher resolution, lower LR, further reduced regularization.
            Hyperparameters(num_intervals=4096, learning_rate=0.001, num_steps=35000, warmup_steps=1750,
                            regularization_strength=5e-8, curvature_reg_strength=1e-9, gradient_clip_value=0.5),
            # Stage 4: High resolution, very low LR, minimal regularization for fine-tuning.
            Hyperparameters(num_intervals=8192, learning_rate=0.0001, num_steps=30000, warmup_steps=1500,
                            regularization_strength=1e-8, curvature_reg_strength=1e-10, gradient_clip_value=0.25),
            # Stage 5: Maximum resolution, extremely low LR, almost no regularization.
            Hyperparameters(num_intervals=16384, learning_rate=0.00001, num_steps=25000, warmup_steps=1250,
                            regularization_strength=1e-9, curvature_reg_strength=1e-11, gradient_clip_value=0.1),
            # Stage 6: Even higher resolution push, ultra-low LR, minimal regularization for final polish.
            Hyperparameters(num_intervals=32768, learning_rate=0.000001, num_steps=20000, warmup_steps=1000,
                            regularization_strength=1e-10, curvature_reg_strength=1e-12, gradient_clip_value=0.05)
        ]

        # Initialize f_values with a Gaussian pulse for the very first stage, as used in Inspiration 1
        # which yielded the highest C2.
        key = jax.random.PRNGKey(42)
        initial_N = stage_configs[0].num_intervals
        x_points_initial = jnp.arange(initial_N) * (self.L / initial_N)
        mu = self.L / 2.0  # Center the Gaussian in the domain [0, L]
        sigma = self.L / 6.0  # Standard deviation, chosen to fit well within [0, L]
        f_values = jnp.exp(-((x_points_initial - mu)**2) / (2 * sigma**2))
        f_values = jax.nn.relu(f_values) # Ensure non-negativity

        final_c2_val, final_regularized_loss = 0.0, jnp.inf
        N_previous_stage = initial_N

        for stage_idx, stage_hypers in enumerate(stage_configs):
            print(f"\n--- Starting Stage {stage_idx + 1}/{len(stage_configs)} ---")
            self.hypers = stage_hypers

            if stage_idx > 0:
                f_values = self._interpolate_f_values(f_values, N_previous_stage, self.hypers.num_intervals)

            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=self.hypers.learning_rate,
                warmup_steps=self.hypers.warmup_steps,
                decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
                end_value=self.hypers.learning_rate * 1e-5,
            )
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.hypers.gradient_clip_value),
                optax.adam(learning_rate=schedule)
            )
            opt_state = self.optimizer.init(f_values)

            print(f"Stage {stage_idx+1}: N={self.hypers.num_intervals}, Steps={self.hypers.num_steps}, LR={self.hypers.learning_rate}, Reg1={self.hypers.regularization_strength}, Reg2={self.hypers.curvature_reg_strength}, Clip={self.hypers.gradient_clip_value}")
            train_step_jit = jax.jit(self.train_step)

            for step in range(self.hypers.num_steps):
                f_values, opt_state, loss = train_step_jit(f_values, opt_state)
                # Print progress more frequently for multi-stage visibility
                if step % (self.hypers.num_steps // 10) == 0 or step == self.hypers.num_steps - 1:
                    # For printing, show C2 without regularization (consistent with inspirations)
                    c2_display = -self._objective_fn(f_values, current_regularization_strength=0.0)
                    print(f"  Stage {stage_idx+1} Step {step:5d} | C2 ≈ {c2_display:.8f} (Loss: {-loss:.8f})")

            # Final evaluation for the current stage
            current_stage_final_c2 = -self._objective_fn(f_values, current_regularization_strength=0.0)  # Unregularized C2
            current_stage_final_loss = loss  # The 'loss' variable holds the LAST regularized loss value from this stage.
            print(f"--- End of Stage {stage_idx+1} | C2: {current_stage_final_c2:.8f} (Reg. Loss: {current_stage_final_loss:.8f}) ---")

            final_c2_val = current_stage_final_c2  # Store the unregularized C2
            final_regularized_loss_val = current_stage_final_loss  # Store the regularized loss
            N_previous_stage = self.hypers.num_intervals

        print(f"\nFinal C2 lower bound found after all stages: {final_c2_val:.8f}")
        return jax.nn.relu(f_values), final_c2_val, final_regularized_loss_val  # Return both

def run():
    """Entry point for running the optimization."""
    np.random.seed(42)
    # Pass a dummy Hyperparameters object; the actual ones are defined in run_optimization.
    hypers = Hyperparameters(0,0,0,0,0,0,0)
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val, final_regularized_loss_val = optimizer.run_optimization()

    loss_val = float(final_regularized_loss_val)
    f_values_np = np.array(optimized_f)
    # Return the num_intervals from the optimizer's state after the final stage.
    return f_values_np, float(final_c2_val), loss_val, optimizer.hypers.num_intervals


# EVOLVE-BLOCK-END
