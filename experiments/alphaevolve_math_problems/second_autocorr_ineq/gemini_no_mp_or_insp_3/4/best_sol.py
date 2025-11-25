# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass

# Enable float64 for higher precision, crucial for sensitive mathematical constants
jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1024  # Further increased resolution for f to explore even finer structures
    learning_rate: float = 0.01
    num_steps: int = 120000  # Further increased steps for better convergence with higher resolution
    warmup_steps: int = 8000 # Adjusted warmup proportionally for more steps


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        """
        # Ensure f is non-negative by squaring the parameters.
        # This is a common way to enforce non-negativity during optimization.
        f_non_negative = jnp.square(params)
        # Add a small epsilon to prevent f_non_negative from being exactly zero everywhere,
        # which would make integral_f zero and lead to division by zero.
        f_non_negative = f_non_negative + 1e-9

        # Define discretization step sizes
        N = self.hypers.num_intervals
        h_f = 1.0 / N  # Step size for f, assuming f is defined on [0, 1]

        # The discrete convolution from FFT needs to be scaled by the step size h_f
        # to approximate the continuous convolution integral.
        # The relationship is: g_continuous(k*h) ≈ h_f * g_discrete[k]
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        # The result of FFT convolution is the unscaled discrete convolution
        g_unscaled = jnp.fft.ifft(fft_f * fft_f).real
        # Scale it to get the values of the continuous convolution function g = f*f
        g_scaled = g_unscaled * h_f

        # The step size for the convolution g, which is supported on [0, 2], is also h_f.
        # It has 2N points over a length of 2, so h_conv = 2.0 / (2*N) = 1.0/N = h_f.
        h_conv = h_f

        # Calculate integral of f for the L1-norm of g = f*f
        # Integral_f = sum(f_values) * h_f (Riemann sum for piecewise constant f)
        integral_f = jnp.sum(f_non_negative) * h_f
        # L1-norm of g: ||f*f||_1 = (Integral_f)^2 for non-negative f.
        norm_1 = integral_f**2
        norm_1 = jnp.maximum(norm_1, 1e-12)

        # Calculate L2-norm squared of the convolution, ||g||_2^2 = integral(g(x)^2 dx).
        # We use a high-accuracy numerical integration rule for a piecewise linear function
        # whose vertex values are given by the correctly scaled g_scaled.
        g_vals_i = g_scaled[:-1]
        g_vals_i_plus_1 = g_scaled[1:]
        l2_norm_squared = jnp.sum((h_conv / 3) * (g_vals_i**2 + g_vals_i * g_vals_i_plus_1 + g_vals_i_plus_1**2))

        # Calculate infinity-norm of the convolution, ||g||_inf = sup|g(x)|
        # This must be calculated on the correctly scaled convolution values.
        norm_inf = jnp.max(g_scaled)
        norm_inf = jnp.maximum(norm_inf, 1e-12)

        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, params: jnp.ndarray, opt_state: optax.OptState) -> tuple:
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

        key = jax.random.PRNGKey(42)
        # Initialize parameters for f. jnp.square will be applied in _objective_fn.
        # Initializing with uniform values ensures non-zero starting point for f.
        params = jax.random.uniform(key, (self.hypers.num_intervals,))

        opt_state = self.optimizer.init(params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            params, opt_state, loss = train_step_jit(params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")

        # Calculate final C2 using the optimized parameters, applying the non-negativity transform
        final_c2 = -self._objective_fn(params)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        # Return the actual function values (squared parameters) for analysis
        return jnp.square(params), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f_values, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f_values)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
