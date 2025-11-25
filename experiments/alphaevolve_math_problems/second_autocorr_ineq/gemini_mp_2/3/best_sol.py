# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 400
    learning_rate: float = 0.005  # Slightly reduced for finer convergence
    num_steps: int = 40000
    warmup_steps: int = 3500  # Adjusted for new num_steps
    domain_length_init: float = 1.5  # Initial guess for the effective support length L


class C2Optimizer:
    """
    Optimizes a parameterized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    The function f is parameterized by a small MLP and a learnable domain length L.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.mlp_key = jax.random.PRNGKey(42)  # For MLP initialization

        # Define MLP architecture
        # Input is a single normalized x coordinate, output is a single f(x) value
        self.mlp_layers = [
            {'output_dim': 32, 'activation': jax.nn.relu},
            {'output_dim': 32, 'activation': jax.nn.relu},
            {'output_dim': 1, 'activation': None}  # No activation on final layer, relu applied later
        ]

    def _init_mlp_params(self, key):
        """Initializes weights and biases for the MLP."""
        params = []
        in_dim = 1  # Input dimension for x (normalized coordinate)
        for i, layer_def in enumerate(self.mlp_layers):
            out_dim = layer_def['output_dim']
            w_key, b_key = jax.random.split(key)
            # Xavier uniform initialization suitable for ReLU activations
            limit = np.sqrt(6 / (in_dim + out_dim))
            w = jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit)
            b = jnp.zeros(out_dim)
            params.append({'weights': w, 'biases': b})
            in_dim = out_dim
            key, _ = jax.random.split(key)  # Update key for next layer
        return params

    def _apply_mlp(self, params, x):
        """Applies the MLP forward pass."""
        # x can be a scalar or an array of x_coords
        # Ensure x is 2D for matmul: (N, 1) or (1, 1)
        x_reshaped = x.reshape(-1, 1)
        for i, layer_def in enumerate(self.mlp_layers):
            layer_params = params[i]
            y = jnp.dot(x_reshaped, layer_params['weights']) + layer_params['biases']
            if layer_def['activation'] is not None:
                y = layer_def['activation'](y)
            x_reshaped = y  # Output of current layer becomes input for next
        return x_reshaped.flatten() # Return 1D array of f(x) values

    def _objective_fn(self, params: dict) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        `params` now contains MLP weights and the log of the domain length L.
        """
        mlp_params = params['mlp_params']
        L = jnp.exp(params['log_L'])  # Ensure L is positive and directly optimized

        N = self.hypers.num_intervals
        dx_f = L / N  # Step size for function f over its domain [0, L]

        # 1. Define x_coords for f on [0, L)
        x_coords = jnp.linspace(0, L - dx_f, N)

        # 2. Get f_raw from MLP
        # Normalize x_coords to [0, 1] for MLP input
        normalized_x_coords = x_coords / L
        f_raw = self._apply_mlp(mlp_params, normalized_x_coords)

        # 3. Enforce non-negativity and normalize integral
        f_non_negative = jax.nn.relu(f_raw)
        integral_f = jnp.sum(f_non_negative) * dx_f
        # Add a small epsilon to avoid division by zero if f_values collapse to all zeros.
        f_normalized = f_non_negative / jnp.maximum(integral_f, 1e-6)

        # Unscaled discrete autoconvolution
        # Pad f_normalized to 2N points for convolution.
        # If f is on [0,L], f*f is on [0,2L].
        padded_f = jnp.pad(f_normalized, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # The convolution result 'convolution' has 2N points and approximates f*f on [0, 2L - dx_f].
        # For L2-norm, we integrate (f*f)^2 over [0, 2L].
        # We assume (f*f)(2L) = 0 for the piecewise linear integration.
        y_points = jnp.concatenate([convolution, jnp.array([0.0])]) # Now 2N+1 points
        # The integration step size is dx_f, consistent with the sampling of convolution.
        h_integration = dx_f

        # Calculate L2-norm squared of the convolution using the piecewise linear integral formula.
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h_integration / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate infinity-norm of the convolution.
        norm_inf = jnp.max(jnp.abs(convolution))

        # Calculate C2 ratio.
        # Since we normalized f such that ∫f = 1, then ||f ★ f||₁ = (∫f)² = 1.
        # Therefore, the objective simplifies to C₂ = ||f ★ f||₂² / ||f ★ f||_{∞}.
        denominator = norm_inf
        # Add a small epsilon to the denominator to prevent division by zero
        # if norm_inf becomes zero (e.g., if f collapses to all zeros).
        c2_ratio = l2_norm_squared / jnp.maximum(denominator, 1e-6)

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, params: dict, opt_state: optax.OptState) -> tuple:
        """Performs a single training step, optimizing MLP params and log_L."""
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

        # Initialize MLP parameters
        mlp_params = self._init_mlp_params(self.mlp_key)

        # Initialize learnable domain length L (optimize log(L) to ensure L > 0)
        log_L_init = jnp.array(jnp.log(self.hypers.domain_length_init), dtype=jnp.float32)

        # Combine all learnable parameters
        params = {'mlp_params': mlp_params, 'log_L': log_L_init}

        opt_state = self.optimizer.init(params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            params, opt_state, loss = train_step_jit(params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                current_L = jnp.exp(params['log_L'])
                print(f"Step {step:5d} | C2 ≈ {-loss:.8f} | L ≈ {current_L:.4f}")

        final_c2 = -self._objective_fn(params)
        final_L = jnp.exp(params['log_L'])
        print(f"Final C2 lower bound found: {final_c2:.8f} with L={final_L:.4f}")

        # To return the optimized function values, we need to evaluate the MLP
        # with the final L and MLP params.
        N = self.hypers.num_intervals
        dx_f = final_L / N
        x_coords_final = jnp.linspace(0, final_L - dx_f, N)
        normalized_x_coords_final = x_coords_final / final_L
        final_f_raw = self._apply_mlp(params['mlp_params'], normalized_x_coords_final)
        optimized_f_values = jax.nn.relu(final_f_raw)

        return optimized_f_values, final_c2


def run():
    """Entry point for running the optimization."""
    # Set numpy random seed for overall reproducibility outside JAX.
    np.random.seed(42)
    hypers = Hyperparameters(domain_length_init=1.5) # Initialize with a reasonable L
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
