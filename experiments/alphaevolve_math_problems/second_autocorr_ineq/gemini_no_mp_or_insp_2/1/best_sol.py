# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 400  # Increased resolution for function representation
    learning_rate: float = 0.005  # Adjusted learning rate for more steps
    num_steps: int = 500000  # Significantly increased number of optimization steps for higher resolution
    warmup_steps: int = 5000  # Adjusted warmup steps


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

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

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
        # Initialize f_values with a Gaussian-like bump to provide a more structured starting point.
        # This can help the optimizer converge to better local minima than uniform random noise.
        x_points = jnp.linspace(0, 1, self.hypers.num_intervals, endpoint=False)
        center = 0.5
        width = 0.2  # Adjust width to control the spread of the initial bump
        f_values = jnp.exp(-((x_points - center) ** 2) / (2 * width**2))
        # Scale the initial values to prevent them from being too large or too small
        f_values = f_values / jnp.sum(f_values) * self.hypers.num_intervals / 5.0 + 0.1
        # Add a tiny bit of noise to break perfect symmetry and encourage exploration
        f_values += jax.random.uniform(key, f_values.shape) * 0.01

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
