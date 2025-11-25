# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 512  # N points for f
    domain_length_f: float = 4.0  # Initial L: compact support of f is [-L/2, L/2]
    learning_rate: float = 5e-3
    num_steps: int = 20000
    warmup_steps: int = 1000


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant.
    Enforces ∫f = 1 and uses a consistent discretization for norms.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, params: dict) -> jnp.ndarray:
        """
        Computes the objective function for C2, enforcing ∫f = 1.
        The objective is C2 = ||f ★ f||₂² / ||f ★ f||_{∞},
        simplified because ||f ★ f||₁ = (∫f)² = 1 when ∫f = 1.
        """
        f_values = params["f_values"]
        # L is a learnable parameter. We optimize log(L) to ensure L > 0.
        L = jnp.exp(params["log_L"])

        N = self.hypers.num_intervals
        dx_f = L / N  # Step size for f on domain [-L/2, L/2]

        # 1. Enforce non-negativity and normalize f such that ∫f = 1
        f_non_negative = jax.nn.relu(f_values)
        integral_f = jnp.sum(f_non_negative) * dx_f
        # Add epsilon for numerical stability in case integral_f is zero
        f_normalized = f_non_negative / (integral_f + 1e-9)

        # 2. Compute autoconvolution f ★ f using FFT
        # f_normalized has length N and is supported on [-L/2, L/2].
        # Its autoconvolution (f ★ f) will be supported on [-L, L].
        # The output length of linear convolution is 2*N - 1.
        # Pad f_normalized to M points for FFT, where M >= 2*N - 1.
        # Using M = 2*N for simplicity, which is >= 2*N-1 for N>=1.
        M = 2 * N
        padded_f = jnp.pad(f_normalized, (0, M - N))

        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # The convolution output has M points and corresponds to a domain of length 2L.
        # So, the step size for convolution points is dx_conv.
        dx_conv = (2 * L) / M

        # 3. Calculate norms for C2 = ||f ★ f||₂² / ||f ★ f||_{∞}
        # L1-norm of convolution is 1 because ∫f = 1.

        # L2-norm squared of the convolution (using rectangular sum approximation)
        l2_norm_squared = jnp.sum(convolution**2) * dx_conv

        # Infinity-norm of the convolution (f*f is always non-negative)
        norm_inf = jnp.max(convolution)
        # Add epsilon for numerical stability if norm_inf is zero
        norm_inf = jnp.maximum(norm_inf, 1e-9)

        # Calculate C2 ratio
        c2_ratio = l2_norm_squared / norm_inf

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, params: dict, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def _calculate_c2_from_params(self, params: dict, use_trapezoidal: bool = True) -> jnp.ndarray:
        """
        Calculates C2 from a parameter dictionary, with an option for more accurate integration.
        This is used for final evaluation, not for gradient descent.
        """
        f_values = params["f_values"]
        L = jnp.exp(params["log_L"])
        N = self.hypers.num_intervals
        dx_f = L / N

        f_non_negative = jax.nn.relu(f_values)
        integral_f = jnp.sum(f_non_negative) * dx_f
        f_normalized = f_non_negative / (integral_f + 1e-9)

        M = 2 * N
        padded_f = jnp.pad(f_normalized, (0, M - N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real
        dx_conv = (2 * L) / M

        if use_trapezoidal:
            # More accurate trapezoidal rule for final evaluation
            l2_norm_squared = (jnp.sum(convolution**2) - 0.5 * (convolution[0]**2 + convolution[-1]**2)) * dx_conv
        else:
            # Rectangular rule (consistent with optimizer)
            l2_norm_squared = jnp.sum(convolution**2) * dx_conv

        norm_inf = jnp.max(convolution)
        norm_inf = jnp.maximum(norm_inf, 1e-9)
        c2_ratio = l2_norm_squared / norm_inf
        return c2_ratio

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

        # Initialize f_values to a triangle function and L to its initial value.
        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        L_initial = self.hypers.domain_length_f
        
        x_f = jnp.linspace(-L_initial / 2, L_initial / 2, N, endpoint=False)
        f_initial = jnp.maximum(0.0, 1.0 - jnp.abs(x_f) / (L_initial / 2))

        dx_f = L_initial / N
        initial_integral_f = jnp.sum(f_initial) * dx_f
        f_values_initial = f_initial / (initial_integral_f + 1e-9)

        # The state to be optimized is a dictionary (PyTree)
        params = {
            "f_values": f_values_initial,
            "log_L": jnp.log(L_initial),
        }

        opt_state = self.optimizer.init(params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Initial Domain Length (L): {L_initial:.2f}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            params, opt_state, loss = train_step_jit(params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                current_L = jnp.exp(params["log_L"])
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f} | L ≈ {current_L:.4f}")

        final_c2 = self._calculate_c2_from_params(params, use_trapezoidal=True)
        final_L = jnp.exp(params["log_L"])
        print(f"Final C2 lower bound found: {final_c2:.8f} with optimal L ≈ {final_L:.4f}")
        
        # Final normalization before returning
        f_values = jax.nn.relu(params["f_values"])
        dx_f_final = final_L / N
        integral_f_final = jnp.sum(f_values) * dx_f_final
        f_values_normalized = f_values / (integral_f_final + 1e-9)
        
        return f_values_normalized, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
