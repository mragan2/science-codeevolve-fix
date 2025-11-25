# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np # numpy is imported, so its seed should be set
from dataclasses import dataclass

# Set random seed for numpy for reproducibility (as per problem statement)
np.random.seed(42)
# JAX uses PRNGKey objects for explicit stateful randomness,
# so a global seed is not typically set for JAX.
# Any JAX random operations will use a specific PRNGKey if needed.


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 800 # Keeping high resolution, proven effective
    learning_rate: float = 0.02 # Slightly increased peak learning rate for potentially faster/better convergence
    num_steps: int = 300000 # Maintaining increased steps for exhaustive search
    warmup_steps: int = 10000 # Maintaining adjusted warmup steps


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the simplified C2 formula:
        C₂ = ||f ★ f||₂² / ((∫f)² ||f ★ f||_{∞})
        """
        # Ensure f_values are non-negative
        f_non_negative = jax.nn.relu(f_values)

        N = self.hypers.num_intervals
        L_f = 0.5  # Fixed domain length for f, as per problem description
        dx_f = L_f / N  # Discretization step for f (N bins over L_f)

        # Calculate integral_f = ∫f dx
        # Using trapezoidal rule for integral_f, for improved accuracy and consistency.
        integral_f = jax.scipy.integrate.trapezoid(f_non_negative, dx=dx_f)

        # Handle integral_f being zero to avoid division by zero later
        # If integral_f is ~0, C2 is undefined or 0, leading to large negative C2,
        # which means large positive loss for minimization.
        integral_f_squared = integral_f**2

        # FFT-based convolution
        # f_values has N points. Convolution of two N-point functions results in 2N-1 points.
        # The support of f*f is [0, 2*L_f] = [0, 1].
        conv_output_len = 2 * N - 1
        
        # For FFT, pad to a length M >= conv_output_len.
        # A common choice is to pad to 2N for simpler indexing and power-of-2 like behavior.
        fft_len = 2 * N 
        
        # Pad f_non_negative to fft_len
        padded_f = jnp.pad(f_non_negative, (0, fft_len - N))
        
        fft_f = jnp.fft.fft(padded_f, n=fft_len)
        convolution_full = jnp.fft.ifft(fft_f * fft_f, n=fft_len).real

        # Take the relevant part of the convolution (2N-1 points)
        # The FFT-based convolution result needs to be truncated to the actual length.
        convolution = convolution_full[:conv_output_len] 
        
        # Discretization step for the convolution
        # As per problem statement: dx_conv = (2 * L_f) / (2N - 1)
        # This assumes conv_output_len points span the domain [0, 2*L_f] with equal spacing.
        dx_conv = (2 * L_f) / conv_output_len

        # Calculate L2-norm squared of the convolution: ||g||_2^2 = ∫ g(x)^2 dx
        # Using trapezoidal rule for L2-norm squared, as recommended for higher accuracy
        l2_norm_squared = jax.scipy.integrate.trapezoid(convolution**2, dx=dx_conv)

        # Calculate infinity-norm of the convolution: ||g||_inf = sup|g|
        # Since f >= 0, convolution >= 0, so jnp.max(convolution) is sufficient.
        norm_inf = jnp.max(convolution)

        # Define conditions for numerical stability checks using JAX-compatible operations.
        # These replace the Python 'if' statements to avoid TracerBoolConversionError in JIT.
        integral_f_is_problematic = integral_f_squared < 1e-12
        norm_inf_is_problematic = norm_inf < 1e-9
        
        # Calculate C2 ratio using the simplified formula
        # If either integral_f_squared or norm_inf is too small, the denominator would be problematic.
        # In such cases, we want to return a large positive loss to penalize these solutions.
        
        # Calculate the raw C2 ratio. This might involve division by a very small number,
        # but the subsequent jnp.where will handle the problematic cases.
        denominator = integral_f_squared * norm_inf
        raw_c2_ratio = l2_norm_squared / denominator
        
        # Use jnp.where to conditionally return the loss.
        # If problematic, return a large positive value (1e12) to indicate a very poor C2.
        # Otherwise, return the negative of the calculated C2 ratio (since we minimize loss to maximize C2).
        loss = jnp.where(
            jnp.logical_or(integral_f_is_problematic, norm_inf_is_problematic),
            jnp.array(1e12), # Large positive loss for minimization
            -raw_c2_ratio    # Negative C2 ratio for minimization
        )
        return loss

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
            end_value=self.hypers.learning_rate * 1e-6, # Even lower end_value for ultra-fine tuning
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        # The initial f_values are deterministic (all ones), so PRNGKey is not strictly needed for f_values init.
        # However, for general JAX usage, a key is good practice for any future random ops.
        # key = jax.random.PRNGKey(42) # Commented out as not used for f_values initialization

        # Initialize f_values to represent a triangular pulse.
        # This is a 'novel function' initial guess, as it's known to be optimal for C3,
        # and we are exploring if it yields a higher C2 than the rectangular pulse.
        N = self.hypers.num_intervals
        L_f = 0.5 # Fixed domain length for f, as per problem description
        
        # Create x values for the function domain [0, L_f]
        # We use endpoint=False to match the discretization for N bins over L_f.
        # This means f_values[i] corresponds to the center of each bin.
        x_points = jnp.linspace(0.0, L_f, N, endpoint=False) + (L_f / N / 2.0)
        
        # Define a symmetric triangular pulse peaking at L_f/2
        # f(x) = 1 - |x - L_f/2| / (L_f/2) for x in [0, L_f]
        f_values = 1.0 - jnp.abs(x_points - L_f / 2.0) / (L_f / 2.0)
        
        # Ensure non-negativity for values near the edges due to floating point or definition
        f_values = jnp.maximum(0.0, f_values)

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
