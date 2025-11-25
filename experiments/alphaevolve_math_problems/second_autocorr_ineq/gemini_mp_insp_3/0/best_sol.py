# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Any, Dict # Added for type hinting MLP parameters

# Set random seed for numpy operations, if any are used outside JAX's PRNG system.
np.random.seed(42)
# JAX's PRNG is handled via explicit keys; the key for f_values is set in run_optimization.


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Increased resolution to allow for finer function details.
    # Powers of 2 can be efficient for FFTs.
    num_intervals: int = 256
    learning_rate: float = 0.01
    # More steps for convergence with a larger parameter space.
    num_steps: int = 25000
    warmup_steps: int = 2500
    # MLP architecture for parameterizing f(x).
    # Input is 1 (x-coordinate), output is 1 (f(x)).
    mlp_layer_sizes: Sequence[int] = (64, 64, 1)


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    The function f(x) is parameterized by a small MLP.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _init_mlp_params(self, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Initializes MLP parameters."""
        params = []
        # Input layer dimension is 1 (for x-coordinate)
        input_dim = 1
        layer_sizes = (input_dim,) + self.hypers.mlp_layer_sizes

        for i in range(len(layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            # He initialization for ReLU activation
            limit = np.sqrt(2.0 / in_dim)
            w = jax.random.uniform(subkey, (in_dim, out_dim), minval=-limit, maxval=limit)
            b = jnp.zeros(out_dim)
            params.append({'w': w, 'b': b})
        return params

    def _apply_mlp(self, params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        """Applies the MLP to an input x."""
        # Ensure x is at least 1D for consistent matrix multiplication, e.g., (1,)
        x = jnp.atleast_1d(x)
        for i, p in enumerate(params):
            x = jnp.dot(x, p['w']) + p['b']
            if i < len(params) - 1: # Apply ReLU to hidden layers
                x = jax.nn.relu(x)
        # Output layer with ReLU to ensure f(x) >= 0. Squeeze to remove extra dimension.
        return jax.nn.relu(x.squeeze())

    def _objective_fn(self, mlp_params: Dict[str, Any]) -> jnp.ndarray:
        """
        Computes the objective function (negative C2 ratio) by evaluating
        the MLP-parameterized function f.
        """
        N = self.hypers.num_intervals
        dx = 1.0 / N
        
        # Generate x-coordinates for evaluating f(x) over [0, 1)
        x_coords = jnp.linspace(0.0, 1.0 - dx, N)

        # Evaluate f(x) using the MLP for each x-coordinate
        # jax.vmap efficiently applies _apply_mlp across the x_coords array
        f_values = jax.vmap(lambda x_val: self._apply_mlp(mlp_params, x_val))(x_coords)
        
        # Ensure f_values are non-negative (already done by _apply_mlp's final relu, but explicit for clarity)
        f_non_negative = f_values 

        # --- 1. Calculate integral of f ---
        # Using trapezoidal rule for ∫f for increased accuracy.
        if N > 1:
            integral_f = (jnp.sum(f_non_negative) - 0.5 * (f_non_negative[0] + f_non_negative[-1])) * dx
        else:
            integral_f = f_non_negative[0] * dx
        # Guard against integral_f being too small. This enforces ∫f > 0.
        integral_f = jnp.where(integral_f < 1e-6, 1e-6, integral_f)

        # --- 2. Compute the autoconvolution g = f ★ f ---
        # The result of FFT-based convolution is a sum, which needs to be scaled by dx
        # to approximate the integral definition: (f ★ f)(x) ≈ dx * Σ f(t_i)f(x-t_i).
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution_unscaled = jnp.fft.ifft(fft_f * fft_f).real
        # Scale by dx to get the correct magnitude for the continuous convolution `g`.
        g = convolution_unscaled * dx

        # --- 3. Calculate norms of the convolution g ---

        # L2-norm squared: ∫g² dx
        # Use the highly accurate piecewise-linear integral formula: ∫g²dx ≈ Σ (dx/3)(g_i² + g_i*g_{i+1} + g_{i+1}²).
        # The convolution `g` is supported on [0, 2). The `g` array contains samples
        # for x=0, dx, ..., (2N-1)dx. To integrate up to 2, we need g(2), which is 0.
        g_for_integral = jnp.concatenate([g, jnp.array([0.0])])
        g1, g2 = g_for_integral[:-1], g_for_integral[1:]
        l2_norm_squared = jnp.sum((dx / 3) * (g1**2 + g1 * g2 + g2**2))

        # L1-norm: ||g||₁ = ||f ★ f||₁ = (∫f)²
        # This is an analytical identity for f >= 0.
        norm_1 = integral_f**2

        # Infinity-norm: ||g||_{∞} = sup|g(x)|
        norm_inf = jnp.max(g)

        # --- 4. Calculate C2 and Loss ---
        # Guard against division by zero.
        norm_inf = jnp.where(norm_inf < 1e-6, 1e-6, norm_inf)
        denominator = norm_1 * norm_inf
        denominator = jnp.where(denominator < 1e-12, 1e-12, denominator)

        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, mlp_params: Dict[str, Any], opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(mlp_params)
        updates, opt_state = self.optimizer.update(grads, opt_state, mlp_params)
        mlp_params = optax.apply_updates(mlp_params, updates)
        return mlp_params, opt_state, loss

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

        # Initialize MLP parameters
        key = jax.random.PRNGKey(42)
        mlp_params = self._init_mlp_params(key)

        opt_state = self.optimizer.init(mlp_params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            mlp_params, opt_state, loss = train_step_jit(mlp_params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f}")

        final_c2 = -self._objective_fn(mlp_params)
        print(f"Final C2 lower bound found: {final_c2:.8f}")
        
        # Reconstruct the optimized f_values from the final MLP parameters for output
        N = self.hypers.num_intervals
        dx = 1.0 / N
        x_coords = jnp.linspace(0.0, 1.0 - dx, N)
        optimized_f_values = jax.vmap(lambda x_val: self._apply_mlp(mlp_params, x_val))(x_coords)

        return optimized_f_values, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
