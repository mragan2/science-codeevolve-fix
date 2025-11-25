# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Enable float64 for higher precision, crucial for subtle improvements in C2.
# This must be set before any JAX operations involving floating point numbers.
jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Further increased resolution (doubled) for even more accurate function representation.
    # Further increased resolution (doubled) for even more accurate function representation.
    num_intervals: int = 2048 # High resolution for function representation
    learning_rate: float = 0.01 # Peak learning rate, slightly higher than Insp 1's 0.005 but still moderate
    num_steps: int = 250000 # Further increased steps for deeper convergence with x64 and higher resolution
    warmup_steps: int = 25000 # Proportional warmup
    domain_length: float = 2.0 # Wider domain to allow for broader function supports, from Insp 1
    # Regularization aligned with Insp 1's successful very low values
    regularization_strength: float = 5e-8 # L2 penalty on the first derivative (smoothness)
    curvature_reg_strength: float = 5e-9 # L2 penalty on the second derivative (curvature)
    gradient_clip_value: float = 1.0 # For training stability


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray, apply_regularization: bool = True) -> jnp.ndarray:
        """
        Computes the objective function C₂ = ||f★f||₂² / ||f★f||_{∞},
        assuming f is normalized such that ∫f dx = 1.
        Includes optional regularization for training.
        """
        # 1. Enforce f(x) >= 0
        f_non_negative = jax.nn.relu(f_values.astype(jnp.float64)) # Ensure float64

        # 2. Normalize f such that its numerical integral is 1.
        N = self.hypers.num_intervals
        L = self.hypers.domain_length
        h_f = L / N # Grid spacing for f

        integral_f = jnp.sum(f_non_negative) * h_f
        # Add a small epsilon to denominator to prevent division by zero if f_non_negative is all zeros.
        f_normalized = f_non_negative / (integral_f + jnp.finfo(jnp.float64).eps)

        # 3. Compute autoconvolution g = f ★ f using FFT.
        # Pad to 2N for linear convolution. The result will have 2N-1 non-zero points.
        M = 2 * N
        padded_f = jnp.pad(f_normalized, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        
        # The scaling factor for the continuous convolution.
        # jnp.fft.ifft scales by 1/M by default. To approximate the continuous
        # convolution integral values, we multiply by h_f.
        # This specific scaling (jnp.fft.ifft * h_f) combined with ∫f=1 normalization
        # was successful in Inspiration 1.
        g = jnp.fft.ifft(fft_f * fft_f).real * h_f

        # 4. Calculate norms of the convolution g.
        # Step size for convolution domain [0, 2L] is (2L)/(2N) = L/N = h_f
        h_g = h_f 
        
        norm_inf = jnp.max(g)
        
        # Use piecewise linear integral formula for L2 norm squared (Simpson's rule variant)
        # This is more accurate than a simple Riemann sum.
        # y_points creates 2N+2 points (including 0 at boundaries), defining 2N+1 intervals
        # over the domain [0, 2L]. Thus, the integration step size is (2L) / (num_conv_points + 1).
        num_conv_points = len(g)
        h_conv_integr = (2 * L) / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), g, jnp.array([0.0], dtype=jnp.float64)])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_conv_integr / 3) * (y1**2 + y1 * y2 + y2**2))

        # 5. Calculate C2. Since ∫f=1, ||f★f||₁ = 1.
        # Add a small epsilon to denominator to prevent division by zero for norm_inf.
        norm_inf_guarded = jnp.maximum(norm_inf, jnp.finfo(jnp.float64).eps)
        c2 = l2_norm_squared / norm_inf_guarded
        loss = -c2

        # 6. Apply regularization during training
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
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values, apply_regularization=True)
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
        # Apply gradient clipping to stabilize training, especially with lower regularization.
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.gradient_clip_value),
            optax.adam(learning_rate=schedule)
        )

        # Set numpy random seed for reproducibility, as per requirements, for any numpy operations.
        np.random.seed(42)

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        L = self.hypers.domain_length
        # Ensure x_points are float64
        x_points = jnp.linspace(0.0, L, N, endpoint=False, dtype=jnp.float64)
        
        # Initialize with a sum of Gaussians + noise (inspired by Insp. 1)
        # to provide a complex, non-symmetric starting point, potentially avoiding
        # simple local optima.
        num_gaussians = 3
        # Scale means and std_devs to the new domain_length L
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
        f_values = jax.nn.relu(f_values) # Ensure initial non-negativity

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        
        # JIT compile the train step and the objective function without regularization for reporting
        train_step_jit = jax.jit(self.train_step)
        objective_fn_no_reg = jax.jit(lambda f: self._objective_fn(f, apply_regularization=False))

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # For printing, report the pure C2 value without the regularization penalty
                current_c2 = -objective_fn_no_reg(f_values)
                print(f"Step {step:5d} | C2 ≈ {current_c2:.8f} (Loss with reg: {loss:.8f})")
        
        final_c2 = -objective_fn_no_reg(f_values)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the final regularized loss, which is the actual loss minimized during training.
        final_regularized_loss = loss # 'loss' variable here holds the last regularized loss
        return jax.nn.relu(f_values), final_c2, final_regularized_loss


def run():
    """Entry point for running the optimization."""
    # numpy.random.seed(42) is now handled inside run_optimization for consistency.
    
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
