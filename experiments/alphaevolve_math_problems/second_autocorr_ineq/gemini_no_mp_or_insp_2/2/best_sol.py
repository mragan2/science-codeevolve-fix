# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    # Core learning parameters
    learning_rate: float = 0.005
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0  # NEW: Added gradient clipping norm for stability

    # Multi-stage configuration for adaptive resolution and steps
    resolutions: tuple[int, ...] = (256, 1024, 4096, 8192)  # EVOLVED: Increased final resolution to 8192
    steps_per_resolution: tuple[int, ...] = (25000, 50000, 100000, 150000)  # EVOLVED: Increased steps for highest resolution
    
    # Initial function values generation and noise for basin hopping
    initial_scale_factor: float = 1.0
    noise_amplitude_factor: float = 1e-3


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        # self.current_num_intervals will be dynamically set by run_optimization
        self.current_num_intervals = None 
        self.optimizer = None # Optimizer will be initialized per stage

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        """
        f_non_negative = jax.nn.relu(f_values)

        # 1. Define discretization parameters
        L_f = 1.0  # Domain length for f(x). C2 is scale-invariant, so L_f=1 is fine.
        N_f = self.current_num_intervals # Use the current resolution set by run_optimization
        dx = L_f / N_f  # Step size for f(x) and for g(x) = (f*f)(x)

        # Calculate integral_f early
        integral_f = jnp.sum(f_non_negative) * dx

        # Check for trivial function case (f=0) using JAX-compatible operations
        integral_f_is_trivial = integral_f < 1e-9

        # 2. Calculate the convolution g(x) = (f * f)(x)
        # Pad f for linear convolution using FFT.
        # The length of f_non_negative is N_f. Padded to 2*N_f for convolution.
        padded_f = jnp.pad(f_non_negative, (0, N_f))
        fft_f = jnp.fft.fft(padded_f)
        convolution_discrete_unscaled = jnp.fft.ifft(fft_f * fft_f).real

        # Scale discrete convolution to approximate continuous convolution integral.
        # If f_values are f(k*dx), then discrete convolution needs to be multiplied by dx.
        g_x = convolution_discrete_unscaled * dx

        # Check for trivial convolution case (e.g., if f_values are all zeros after relu)
        # Since f(x) >= 0, g(x) = (f*f)(x) >= 0. So jnp.abs is not needed for norm_inf.
        norm_inf_g = jnp.max(g_x)
        norm_inf_g_is_trivial = norm_inf_g < 1e-9

        # 3. Calculate norms for C2
        # ||f * f||_2^2 = integral (g(x))^2 dx
        # g_x represents (f*f)(x_k) values. Since f is piecewise constant, f*f is piecewise linear.
        # The formula (dx / 3) * (y1^2 + y1*y2 + y2^2) is exact for integrating y(x)^2 where y(x) is linear over the interval.
        y1_g, y2_g = g_x[:-1], g_x[1:]
        l2_norm_squared_g = jnp.sum((dx / 3) * (y1_g**2 + y1_g * y2_g + y2_g**2))

        # (integral f)^2, which is ||f * f||_1 for non-negative f (as per problem statement simplification)
        integral_f_squared = integral_f**2

        # Calculate C2 ratio
        denominator = integral_f_squared * norm_inf_g
        c2_ratio = l2_norm_squared_g / denominator

        # Combine triviality conditions for JAX compatibility
        is_problematic = integral_f_is_trivial | norm_inf_g_is_trivial

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        # If problematic, return a large positive loss (effectively minimizing -C2 to -inf)
        return jnp.where(is_problematic, jnp.array(1e9), -c2_ratio)

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step. The optimizer and its schedule are set in run_optimization."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full multi-stage optimization process."""
        
        optimized_f = None
        final_c2 = jnp.array(-jnp.inf) # Initialize with negative infinity for max
        
        for stage, (num_intervals, num_steps_stage) in enumerate(zip(self.hypers.resolutions, self.hypers.steps_per_resolution)):
            self.current_num_intervals = num_intervals # Set current resolution for the objective function
            
            # Use a consistent key for reproducibility across stages, but vary it slightly
            # for different stages to ensure different noise patterns for basin hopping.
            key = jax.random.PRNGKey(42 + stage) 
            
            if optimized_f is None: # First stage: Initialize randomly
                f_values = jax.random.uniform(key, (num_intervals,)) * self.hypers.initial_scale_factor
            else: # Subsequent stages: Interpolate from previous stage's optimized function
                old_num_intervals = optimized_f.shape[0]
                
                # Create grid points for interpolation. Using endpoint=False to match [0, L_f) domain.
                x_old = jnp.linspace(0.0, 1.0, old_num_intervals, endpoint=False)
                x_new = jnp.linspace(0.0, 1.0, num_intervals, endpoint=False)
                
                # Interpolate the optimized function values to the new resolution
                f_values = jnp.interp(x_new, x_old, optimized_f)
                # Add some small noise to interpolated values to help escape local minima (basin hopping),
                # scaled by the initial factor to maintain relative magnitude.
                f_values += jax.random.uniform(key, (num_intervals,)) * (self.hypers.initial_scale_factor * self.hypers.noise_amplitude_factor)
                f_values = jax.nn.relu(f_values) # Ensure non-negativity after noise addition
            
            # Initialize optimizer and its learning rate schedule for the current stage.
            # The schedule is a function that optax.adam will call at each step.
            schedule_fn = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.hypers.learning_rate,
                warmup_steps=self.hypers.warmup_steps,
                decay_steps=num_steps_stage - self.hypers.warmup_steps,
                end_value=self.hypers.learning_rate * 1e-4,
            )
            # EVOLVED: Added gradient clipping for improved stability during optimization.
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.hypers.max_grad_norm),
                optax.adam(learning_rate=schedule_fn)
            )
            opt_state = self.optimizer.init(f_values)
            
            print(f"\n--- Stage {stage + 1}/{len(self.hypers.resolutions)} ---")
            print(f"Number of intervals (N): {num_intervals}, Steps: {num_steps_stage}")
            
            # JIT compile train_step for the current f_values shape for efficiency
            train_step_jit = jax.jit(self.train_step)

            loss = jnp.inf
            for step_in_stage in range(num_steps_stage):
                f_values, opt_state, loss = train_step_jit(f_values, opt_state)
                
                # Periodically print the current C2 value for monitoring progress
                if (step_in_stage % (num_steps_stage // 10)) == 0 or step_in_stage == num_steps_stage - 1:
                    current_c2 = -self._objective_fn(f_values) # Re-calculate C2 with current f_values
                    print(f"  Stage {stage+1}, Step {step_in_stage:5d} | C2 ≈ {current_c2:.8f}")
            
            optimized_f = jax.nn.relu(f_values) # Ensure final function is non-negative after stage
            final_c2 = -self._objective_fn(optimized_f) # Calculate C2 for the final function of this stage
            print(f"Stage {stage+1} Final C2 lower bound: {final_c2:.8f}")

        print(f"\nOverall Final C2 lower bound found: {final_c2:.8f}")
        return optimized_f, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    # The N_points should reflect the final resolution from the last stage
    return f_values_np, float(final_c2_val), float(loss_val), hypers.resolutions[-1]


# EVOLVE-BLOCK-END
