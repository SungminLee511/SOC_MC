# Category (a) Samplers: Architectures and Loss Functions

This document briefly describes the category (a) SOC samplers used as experimental test-beds for `experimental_plan.md`. All samplers covered here share the category (a) loss structure:
$$\ell^{(\mathrm{a})}(X_{[0,1]}, \theta) = \int_0^1 |u_\theta(X_t, t) - u^**{\mathrm{target}}(X*{[0,1]}, t)|^2,dt$$
where $u^**{\mathrm{target}}$ is a target drift computed from the trajectory. Samplers differ in how $u^**{\mathrm{target}}$ is defined and in controller parameterization details.

-----

## Common Elements

All category (a) samplers considered here operate on a controlled SDE
$$dX_t = b(X_t, t),dt + g(t)^2,u_\theta(X_t, t),dt + g(t),dW_t$$
with a learnable drift $u_\theta$ and a base diffusion schedule $g(t)$. The sampling goal is $X_1 \sim p$ for a Boltzmann target $p(x) \propto \exp(-E(x))$.

**Controller architecture (typical).** A time-conditioned neural network $u_\theta(x, t)$, usually implemented as an MLP with:

- Time embedding (sinusoidal features or learned projection).
- Concatenation of spatial input $x$ and time features.
- Several hidden layers with smooth activations (SiLU, GELU).
- Output layer producing a $d$-dimensional drift correction.

Width and depth depend on problem complexity; for 2D benchmarks, depth 3–5 with width 64–256 is typical.

-----

## Adjoint Sampling (AS)

**Loss formulation.** The regression target is the negative rescaled adjoint:
$$u^**{\mathrm{target,AS}}(X*{[0,1]}, t) = -g(t)^2 Y_t$$
where $Y_t$ is the backward adjoint process satisfying a BSDE terminating in $Y_1 = -\nabla E(X_1)$. In practice, $Y_t$ is computed by propagating $-\nabla E(X_1)$ backward along the sampled trajectory using the adjoint ODE/BSDE for the controlled SDE.

**Loss integrand.**
$$\ell^{(\mathrm{a}), \mathrm{AS}}(X_{[0,1]}, \theta) = \int_0^1 |u_\theta(X_t, t) + g(t)^2 Y_t|^2,dt$$

**Training procedure.** Trajectories are sampled from $Q_\theta$ (current policy), adjoints are computed along them, and the loss is optimized. Standard implementation uses Adam with cosine learning-rate schedule.

**Why category (a).** The loss has explicit $|u_\theta - u^*|^2$ form. The target $u^*$ is constructed from $\nabla E(X_1)$, so on surviving basins (where $\nabla E = \nabla E_S$) the target equals its restricted counterpart, satisfying Assumption L2.

-----

## Adjoint Schrödinger Bridge Sampler (ASBS)

**Loss formulation.** The regression target is derived from the Schrödinger Bridge optimality condition between source $\mu$ and target $p$. The bridge drift at a trajectory point is:
$$u^**{\mathrm{target,ASBS}}(X*{[0,1]}, t) = \text{bridge-matching quantity involving endpoints}$$
often expressible through an adjoint equation similar to AS, but with a different terminal condition accounting for the source-target symmetry of the bridge.

**Loss integrand.** Same category (a) form:
$$\ell^{(\mathrm{a}), \mathrm{ASBS}}(X_{[0,1]}, \theta) = \int_0^1 |u_\theta(X_t, t) - u^*_{\mathrm{target,ASBS}}|^2,dt$$

**Training procedure.** Iterative bridge refinement: trajectories are sampled, bridge targets are computed, and the loss is optimized. Often uses an iterative outer loop alternating between forward and backward bridge directions.

**Why category (a).** Same reasoning as AS — explicit squared-residual form with a target derived from energy gradients.

**Differences from AS.** ASBS’s bridge target incorporates source-distribution information through the endpoint coupling; AS targets depend only on the terminal energy. In practice, ASBS often exhibits smoother training dynamics at the cost of more complex target computation.

-----

## Other Potential Category (a) Samplers

The category (a) family includes any sampler whose loss takes the squared-residual form with a trajectory-derived target. Additional examples that could serve as test-beds:

**Denoising-Energy Matching.** Target is a denoised score quantity: $u^*_{\mathrm{target}} = -\nabla \log q_t^\theta(x)$ where $q_t^\theta$ is a time-$t$ smoothed version of the target. Related to diffusion samplers.

**Iterative Denoising Samplers.** Variants where the regression target is computed via iterative refinement rather than a single BSDE pass.

The experimental plan does not require any specific choice beyond testing on at least two independent category (a) samplers for cross-sampler consistency. AS and ASBS are the recommended primary test-beds given their prominence in the literature and well-understood training procedures.

-----

## What the Experimental Plan Needs from Each Sampler

For each category (a) sampler used as a test-bed, the experimental plan requires:

1. **A controller parameterization** $u_\theta(x, t)$ — any architecture satisfying Assumption L3 (bounded first/second variations, bounded score Hessian and Fisher info on typical support).
1. **A loss function** $\ell^{(\mathrm{a})}$ of category (a) form — any regression target derived from the energy (satisfying Assumption L2).
1. **A training procedure** — standard gradient-based optimization. The plan refers to “epochs” defined by each sampler’s reference configuration.
1. **Sampling capability** — ability to generate trajectories $X_{[0,1]}$ from $Q_\theta$ for mode-weight estimation and Monte Carlo integration in Goals 3-5.
1. **Differentiability through the controller** — autodiff support for computing $\nabla_\theta u_\theta$ (Goals 3-4) and Hessian-vector products (Goal 3).

All of these are standard features of any neural-network-based SOC sampler implementation.

-----

## Theoretical Assumption Compatibility

All category (a) samplers satisfy the theoretical assumptions by construction:

- **L1 (Novikov):** Holds for bounded controllers — standard for neural networks.
- **L2 (Restricted-compatible loss):** Holds whenever $u^*_{\mathrm{target}}$ depends on $E$ only through its gradient $\nabla E$. Both AS and ASBS satisfy this.
- **L3 (Parameterization regularity):** Holds for smooth network architectures with bounded weights over a training neighborhood.
- **L4 (Hessian boundary regularity):** Holds for smooth parameterizations — not automatic but standard.
- **L5 (Revival coordinate change):** Holds generically; Goal 4 verifies its consequence empirically.

No sampler-specific theoretical modifications are needed. The theorem applies to all category (a) samplers as stated.