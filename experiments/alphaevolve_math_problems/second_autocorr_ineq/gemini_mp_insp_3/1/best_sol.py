# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Enable 64-bit floating-point precision for JAX computations
jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 8192  # Doubled resolution to capture extremely fine function details.
    learning_rate: float = 0.005  # Keep learning rate as it proved stable at lower resolutions.
    num_steps: int = 250000  # Increased steps for a more exhaustive search in the larger parameter space.
    warmup_steps: int = 25000  # Increased proportionally to num_steps.


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function C2 = ||f★f||₂² / (||f★f||₁ ||f★f||_{∞}).
        This implementation uses the crucial simplification ||f★f||₁ = (∫f)².
        """
        # Enforce f(x) >= 0
        f_non_negative = jax.nn.relu(f_values)

        # Discretization parameters for f(x) over [0, L_f]
        N = self.hypers.num_intervals
        L_f = 1.0  # Domain length of f, can be normalized to 1 due to scale-invariance
        
        # Define h_f such that N points span (N-1) intervals over L_f.
        # This means points are at 0, h_f, 2*h_f, ..., (N-1)*h_f = L_f.
        # This ensures the trapezoidal rule integrates over the full [0, L_f] domain.
        h_f = L_f / (N - 1) if N > 1 else L_f # Handle N=1 case, though N is typically much larger.

        # Normalize f such that its integral is 1 (due to C2's scale-invariance).
        # This simplifies the denominator as ||f★f||₁ becomes 1, and helps stabilize optimization
        # by removing the scale degree of freedom which does not affect C2.
        # Approximate ∫f using the trapezoidal rule for better accuracy.
        current_integral_f = h_f * (jnp.sum(f_non_negative) - 0.5 * (f_non_negative[0] + f_non_negative[-1]))
        # Add a small epsilon to prevent division by zero if f_non_negative becomes all zeros.
        f_normalized = f_non_negative / (current_integral_f + 1e-12)

        # 1. Calculate ||f★f||₁ = (∫f)²
        # After normalization, ∫f_normalized should be very close to 1.
        # For stability and precision, we assume it's exactly 1.0.
        norm_1_g = 1.0 

        # 2. Compute g = f★f using FFT-based convolution with normalized f
        # Pad f_normalized to length 2N for linear convolution, which has 2N-1 points.
        # Using 2N as FFT length is efficient if N is a power of 2.
        padded_f = jnp.pad(f_normalized, (0, N))
        # The result of discrete convolution needs to be scaled by h_f
        # to approximate the continuous convolution integral.
        g_values_unscaled = jnp.fft.ifft(jnp.fft.fft(padded_f) ** 2).real
        g_values = g_values_unscaled * h_f
        g_values = g_values[:2 * N - 1] # Convolution of N points with N points yields 2N-1 points.

        # 3. Calculate ||f★f||_∞
        norm_inf_g = jnp.max(g_values)
        # Guard against near-zero max to prevent division by zero and stabilize gradients (from Inspiration 1/2).
        norm_inf_g = jnp.where(norm_inf_g < 1e-12, 1e-12, norm_inf_g)

        # 4. Calculate ||f★f||₂² = ∫(f★f)² dx
        # Calculate step size for g_values. g is defined over [0, 2*L_f] with 2N-1 points.
        # These 2N-1 points define (2N-1) - 1 = 2N-2 intervals.
        # Consistent with h_f, h_g = (2 * L_f) / (2 * N - 2) = L_f / (N - 1) if N > 1.
        h_g = (2 * L_f) / jnp.maximum(1, 2 * N - 2) # jnp.maximum handles N=1 case for robustness.
        # Use Simpson's rule for piecewise linear functions: ∫y²dx ≈ ∑(h/3)(yᵢ²+yᵢyᵢ₊₁+yᵢ₊₁²)
        y1, y2 = g_values[:-1], g_values[1:]
        l2_norm_squared_g = jnp.sum((h_g / 3) * (y1**2 + y1 * y2 + y2**2))

        # 5. Calculate C2
        denominator = norm_1_g * norm_inf_g
        # Add a small epsilon for numerical stability, crucial for the first steps.
        c2 = l2_norm_squared_g / (denominator + 1e-12)

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        # Enforce non-negativity immediately after update to stabilize the optimization.
        # This ensures the parameters themselves always respect the f(x) >= 0 constraint.
        f_values = jax.nn.relu(f_values)
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

        # Initialize f_values with a triangular pulse, which is known to be a good starting point.
        # This provides a more informed initial guess than uniform random values.
        N = self.hypers.num_intervals
        x_coords = jnp.linspace(0, 1, N)
        # Create a trapezoidal pulse, which is closer to known good functions (flatter top).
        N = self.hypers.num_intervals
        # Create a trapezoidal pulse, which is closer to known good functions (flatter top).
        # Adjusting side_width to make the flat top relatively wider.
        # Reverting the initialization experiment. The N//3 ratio proved superior,
        # matching the high-performing inspiration programs.
        side_width = N // 3  # ~33% of the width for each ramp
        # Ensure top_width accounts for any remainder due to integer division
        top_width = N - 2 * side_width
        
        # Increase initial peak value slightly to give more room for optimization.
        initial_min_val = 0.005 # Slightly lower start
        initial_peak_val = 0.2  # Higher peak
        
        ramp_up = jnp.linspace(initial_min_val, initial_peak_val, side_width)
        flat_top = jnp.full((top_width,), initial_peak_val)
        # Ensure the last part has the correct length to sum up to N
        ramp_down = jnp.linspace(initial_peak_val, initial_min_val, N - side_width - top_width)
        
        f_values = jnp.concatenate([ramp_up, flat_top, ramp_down])
        # Ensure non-negativity and add a small epsilon to avoid zero sum if optimization drives values too low.
        f_values = jax.nn.relu(f_values) + 1e-6

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        best_c2 = -jnp.inf  # Initialize with negative infinity to track the maximum C2
        best_f_values = f_values  # Store the f_values corresponding to the best C2

        for step in range(self.hypers.num_steps):
            f_values, opt_state, current_loss = train_step_jit(f_values, opt_state)
            
            current_c2 = -current_loss # Convert current loss to C2 value
            if current_c2 > best_c2:
                best_c2 = current_c2
                # Store a copy of f_values that yielded this best C2.
                # f_values are already non-negative due to relu in train_step.
                best_f_values = f_values 

            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | Current C2 ≈ {current_c2:.8f} | Best C2: {best_c2:.8f}")

        # Return the best C2 found and the corresponding function values.
        # jax.nn.relu is applied for consistency, though best_f_values should already be non-negative.
        print(f"Final (best) C2 lower bound found: {best_c2:.8f}")
        return jax.nn.relu(best_f_values), best_c2


def run():
    """Entry point for running the optimization."""
    # Ensure full reproducibility for all stochastic components, a best practice from inspirations.
    np.random.seed(42)

    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
