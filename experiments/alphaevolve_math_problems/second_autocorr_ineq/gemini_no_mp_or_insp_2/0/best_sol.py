# EVOLVE-BLOCK-START
import jax
# Enable 64-bit precision for higher accuracy in optimization.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1024  # Increased resolution to a power of 2 for FFT efficiency and finer detail
    learning_rate: float = 0.005  # Keep learning rate, schedule is robust
    num_steps: int = 750000  # Increased steps for convergence with more parameters and higher resolution
    warmup_steps: int = 15000  # Longer warmup for more stable start with higher complexity
    tv_reg_coeff: float = 1e-6 # Small coefficient for Total Variation regularization to encourage step-like functions


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        
        Assumes f is a piecewise constant function defined on [0, 1]
        with num_intervals segments, each of width h_f.
        """
        f_non_negative = jax.nn.relu(f_values)

        # 1. Calculate integral_f for L1-norm of convolution
        N = self.hypers.num_intervals
        h_f = 1.0 / N  # Width of each interval for f on domain [0, 1]
        integral_f = jnp.sum(f_non_negative) * h_f

        # 2. Unscaled discrete autoconvolution using FFT
        # padded_f will have length 2N, effectively representing f on [0, 2] for convolution.
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # 3. Calculate L2-norm squared of the convolution: ||f ★ f||₂² = ∫ (f★f)(x)² dx
        # The convolution f*f is supported on [0, 2].
        # The 'convolution' array has 2N points.
        # 3. Calculate L2-norm squared of the convolution: ||f ★ f||₂² = ∫ (f★f)(x)² dx
        # The convolution f*f is supported on [0, 2].
        # 'convolution' array has 2N points: (f*f)(0), (f*f)(1/N), ..., (f*f)((2N-1)/N).
        # We assume (f*f)(2) = 0.
        # This means f*f is represented by 2N+1 points (g_0, ..., g_{2N}) with step size 1/N.
        dx_conv = 1.0 / N # Step size for convolution points on domain [0, 2]
        
        # Create points for integration: g_0, ..., g_{2N-1}, g_{2N}=0
        g_values = jnp.concatenate([convolution, jnp.array([0.0])])
        
        # Integrate g(x)^2 using piecewise linear assumption (Simpson's 1/3 rule for quadratic)
        # For a linear segment g(x) from (x_k, g_k) to (x_{k+1}, g_{k+1}),
        # integral of g(x)^2 over [x_k, x_{k+1}] is (h/3) * (g_k^2 + g_k g_{k+1} + g_{k+1}^2)
        g1, g2 = g_values[:-1], g_values[1:]
        l2_norm_squared = jnp.sum((dx_conv / 3) * (g1**2 + g1 * g2 + g2**2))

        # 4. Calculate L1-norm of the convolution using the identity: ||f ★ f||₁ = (∫f)²
        # This is crucial for accuracy as per the problem statement.
        norm_1 = integral_f**2

        # 5. Calculate infinity-norm of the convolution
        # Since f(x) >= 0, f★f(x) >= 0, so abs() is not strictly needed.
        norm_inf = jnp.max(convolution)

        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        # Add a small epsilon to denominator to prevent division by zero,
        # especially during early optimization steps when f_values might be near zero.
        denominator = jnp.maximum(denominator, 1e-10) # Use 1e-10 to keep gradients flowing
        
        c2_ratio = l2_norm_squared / denominator

        # Calculate Total Variation (TV) regularization to encourage step-like functions.
        # This penalizes large differences between adjacent f_non_negative values, scaled by interval width.
        tv_loss = jnp.sum(jnp.abs(f_non_negative[1:] - f_non_negative[:-1])) * h_f
        
        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add TV loss to the objective function to guide the function towards step-like solutions.
        return -c2_ratio + self.hypers.tv_reg_coeff * tv_loss

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
        # Initialize f_values with a small uniform positive value plus noise.
        # This provides a less biased starting point than specific Gaussian shapes,
        # allowing the optimizer and TV regularization to sculpt the function shape
        # more freely, potentially discovering novel and optimal structures.
        initial_value_scale = 0.2 # A reasonable starting scale for f values
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,), minval=0.01, maxval=initial_value_scale)
        # Add a tiny bit of normal noise to ensure no perfect initial symmetries or flat spots,
        # encouraging early exploration.
        f_values += jax.random.normal(key, f_values.shape) * 0.005
        # Ensure non-negativity after adding noise, before the relu in the objective.
        f_values = jnp.maximum(f_values, 1e-3)

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

        # Calculate the final C2 value *without* the regularization term for accurate reporting.
        # This involves re-evaluating the C2 ratio components using the optimized f_values.
        f_non_negative_final = jax.nn.relu(f_values)
        N = self.hypers.num_intervals
        h_f = 1.0 / N
        integral_f = jnp.sum(f_non_negative_final) * h_f
        padded_f = jnp.pad(f_non_negative_final, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real
        dx_conv = 1.0 / N
        g_values = jnp.concatenate([convolution, jnp.array([0.0])])
        g1, g2 = g_values[:-1], g_values[1:]
        l2_norm_squared = jnp.sum((dx_conv / 3) * (g1**2 + g1 * g2 + g2**2))
        norm_1 = integral_f**2
        norm_inf = jnp.max(convolution)
        denominator = jnp.maximum(norm_1 * norm_inf, 1e-10) # Prevent division by zero
        
        final_c2_unregularized = l2_norm_squared / denominator
        
        print(f"Final C2 lower bound found (unregularized): {final_c2_unregularized:.8f}")
        return f_non_negative_final, final_c2_unregularized


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
