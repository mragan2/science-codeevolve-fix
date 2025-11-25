# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 512  # Further increased resolution for better approximation and FFT efficiency
    learning_rate: float = 0.005 # Reduced learning rate for finer convergence
    num_steps: int = 60000  # Increased steps for convergence with higher N and lower LR
    warmup_steps: int = 3000 # Increased warmup steps proportional to total steps
    initial_domain_length: float = 1.0  # Initial guess for the domain length L, now optimized
    l2_reg_coeff: float = 1e-6  # Adjusted L2 regularization, might need less with more points/longer runs


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, params: dict) -> jnp.ndarray:
        """
        Computes the objective function for C2, with domain_length as an optimizable parameter.
        """
        f_values = params['f_values']
        # Ensure domain_length is positive using jnp.exp
        domain_length = jnp.exp(params['log_domain_length']) 

        f_non_negative = jax.nn.relu(f_values)

        N = self.hypers.num_intervals
        # Discretization step size for f and g, incorporating optimized domain_length L
        h_step = domain_length / N

        # Calculate sum of f_values for the integral term (∫f)²
        # The problem statement simplifies ||f ★ f||₁ = (∫f)²
        sum_f_values = jnp.sum(f_non_negative)

        # Compute autoconvolution using FFT
        # Pad f_non_negative to 2N length for linear convolution
        # (convolution of two N-length arrays is 2N-1 long, 2N is a common padding choice)
        padded_f = jnp.pad(f_non_negative, (0, N))  # Array length becomes 2N
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real  # Resulting g_values, length 2N

        # L2-norm squared of the convolution (using Riemann sum as specified in prompt)
        # The prompt's simplified C2 formula uses jnp.sum(g_values**2) in the numerator
        sum_g_squared = jnp.sum(convolution**2)

        # L-infinity norm of the convolution
        norm_g_L_inf = jnp.max(convolution)

        # Calculate C2 ratio based on the prompt's simplified formula:
        # C₂ = jnp.sum(g_values**2) / (h_step * (jnp.sum(f_values))**2 * jnp.max(g_values))
        # Add small epsilon for numerical stability if denominator terms are near zero
        epsilon = 1e-12 # Increased robustness of epsilon
        # Robust denominator handling: ensure sum_f_values is not zero before squaring
        denominator_sum_f_squared = (jnp.sum(f_non_negative) + epsilon)**2 
        denominator = (h_step * denominator_sum_f_squared) * (norm_g_L_inf + epsilon)
        
        # Prevent division by zero if convolution is all zeros (unlikely with good init)
        c2_ratio = sum_g_squared / (denominator + epsilon) 

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # Add L2 regularization to f_values to encourage smoother functions
        l2_regularization_term = self.hypers.l2_reg_coeff * jnp.sum(f_non_negative**2)
        # Add a small regularization for log_domain_length to prevent it from exploding
        # or collapsing, encouraging it to stay within a reasonable range.
        # This can be seen as a prior or a soft constraint.
        domain_length_reg_term = self.hypers.l2_reg_coeff * 0.1 * (jnp.log(domain_length) - jnp.log(self.hypers.initial_domain_length))**2
        
        return -c2_ratio + l2_regularization_term + domain_length_reg_term

    def train_step(self, params: dict, opt_state: optax.OptState) -> tuple:
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
            end_value=self.hypers.learning_rate * 1e-5, # Even smaller end_value for finer tuning
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        # Initialize f_values with a Gaussian pulse for better starting point
        N = self.hypers.num_intervals
        L_init = self.hypers.initial_domain_length
        # Generate N points from 0 to L (exclusive of L)
        x_coords = jnp.linspace(0, L_init, N, endpoint=False) 
        
        # Gaussian parameters: centered in the middle, width is a fraction of L
        mu = L_init / 2  
        sigma = L_init / 8  

        # Initial f_values as a Gaussian pulse
        f_values_init = jnp.exp(-((x_coords - mu) ** 2) / (2 * sigma ** 2))
        # Normalize the initial pulse and add small uniform noise for perturbation
        f_values_init = f_values_init / jnp.max(f_values_init) * 0.5 + jax.random.uniform(key, (N,)) * 0.1

        # Initialize log_domain_length for optimization
        log_domain_length_init = jnp.log(L_init)

        initial_params = {
            'f_values': f_values_init,
            'log_domain_length': log_domain_length_init
        }

        opt_state = self.optimizer.init(initial_params)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        params = initial_params # Initialize params for the loop
        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            params, opt_state, loss = train_step_jit(params, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                # Recalculate C2 for logging, making sure to use the current params
                current_c2 = -self._objective_fn(params) 
                current_domain_length = jnp.exp(params['log_domain_length'])
                print(f"Step {step:5d} | C2 ≈ {current_c2:.8f} | L ≈ {current_domain_length:.4f}")

        final_c2 = -self._objective_fn(params)
        final_f_values = params['f_values']
        final_domain_length = jnp.exp(params['log_domain_length'])
        print(f"Final C2 lower bound found: {final_c2:.8f} with L={final_domain_length:.4f}")
        return jax.nn.relu(final_f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END
