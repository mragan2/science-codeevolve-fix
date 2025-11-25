# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 256  # Increased resolution for f(x)
    learning_rate: float = 0.005  # Slightly reduced learning rate for stability with higher N
    num_steps: int = 50000  # Increased training steps for better convergence
    warmup_steps: int = 5000  # Increased warmup steps for smoother learning rate transition


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
        """
        f_non_negative = jax.nn.relu(f_values)

        # Unscaled discrete autoconvolution
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate L2-norm squared of the convolution (rigorous method)
        num_conv_points = len(convolution)  # This is 2 * N
        # The convolution of a function on [0,1] with itself is on [0,2].
        # The y_points array (with added zeros at boundaries) has num_conv_points + 2 elements.
        # This effectively defines num_conv_points + 1 intervals for integration.
        # Thus, the step size h for integration over a domain of length 2 should be 2.0 / (num_conv_points + 1).
        h_conv_integral = 2.0 / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        # This formula approximates the integral of g(x)^2 using a Simpson-like rule for piecewise linear g(x).
        l2_norm_squared = jnp.sum((h_conv_integral / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate L1-norm of the convolution using the mathematical property ||f ★ f||₁ = (∫f)²
        # Assuming f_values represents samples of f(x) on [0, 1]
        h_f = 1.0 / N  # Step size for f on [0, 1]
        integral_f = jnp.sum(f_non_negative) * h_f
        norm_1_conv = integral_f**2

        # Calculate infinity-norm of the convolution
        # Since f(x) >= 0, f ★ f(x) >= 0, so jnp.abs is not strictly needed for convolution values.
        norm_inf_conv = jnp.max(convolution)

        # Ensure the denominator components are non-zero to avoid division by zero.
        # integral_f should be > 0 due to initialization.
        # norm_inf_conv should be > 0 if f is non-trivial.
        norm_inf_conv_stable = jnp.maximum(norm_inf_conv, 1e-10)

        # Calculate C2 ratio
        denominator = norm_1_conv * norm_inf_conv_stable
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
        f_values = jax.random.uniform(key, (self.hypers.num_intervals,))

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
