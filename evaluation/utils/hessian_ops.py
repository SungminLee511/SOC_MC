"""Hessian-vector product and Lanczos iteration for spectral analysis.

Computes extreme eigenvalues of the training-loss Hessian at checkpoints
using HVP via double-backward autodiff and Lanczos tridiagonalization.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    """Flatten list of parameter tensors into a single 1D vector."""
    return torch.cat([p.reshape(-1) for p in params])


def unflatten_params(flat: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Unflatten a 1D vector back into list of tensors with given shapes."""
    tensors = []
    offset = 0
    for shape in shapes:
        numel = 1
        for s in shape:
            numel *= s
        tensors.append(flat[offset:offset + numel].reshape(shape))
        offset += numel
    return tensors


def hvp(loss_fn, params: List[torch.Tensor], vector: List[torch.Tensor]) -> List[torch.Tensor]:
    """Compute Hessian-vector product: H @ v = d/dθ (∇L · v).

    Args:
        loss_fn: callable that returns scalar loss (must be called fresh each time)
        params: list of parameter tensors (requires_grad=True)
        vector: list of tensors with same shapes as params

    Returns:
        list of tensors: H @ v, same shapes as params
    """
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    # Replace None grads (unused params) with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

    # Compute dot product grad · v
    grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vector))

    # Differentiate again to get Hv
    Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=False, allow_unused=True)
    Hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(Hv, params)]
    return [h.detach() for h in Hv]



def hvp_flat(loss_fn, params: List[torch.Tensor], v_flat: torch.Tensor) -> torch.Tensor:
    """HVP with flat vector interface.

    Args:
        loss_fn: callable returning scalar loss
        params: list of parameter tensors
        v_flat: 1D vector of length sum(p.numel() for p in params)

    Returns:
        1D tensor: H @ v_flat
    """
    shapes = [p.shape for p in params]
    v_list = unflatten_params(v_flat, shapes)
    Hv_list = hvp(loss_fn, params, v_list)
    return flatten_params(Hv_list)


def make_averaged_loss_fn(sampler, energy_fn, batch_size: int, n_avg: int, device: torch.device):
    """Create a loss function that averages over multiple batches for stable HVP.

    Returns:
        callable: () -> scalar loss tensor with grad graph
    """
    def loss_fn():
        total = 0.0
        for _ in range(n_avg):
            total = total + sampler.train_step(batch_size, energy_fn, device)
        return total / n_avg
    return loss_fn


def lanczos(hvp_fn, dim: int, T: int, device: torch.device,
            v0: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Lanczos tridiagonalization for extreme eigenvalues.

    Args:
        hvp_fn: callable(v: Tensor) -> Tensor, the HVP operator
        dim: parameter dimension
        T: number of Lanczos steps
        device: torch device
        v0: optional starting vector (random if None)

    Returns:
        eigenvalues: sorted array of T eigenvalues from tridiagonal matrix
        eigenvectors: (dim, T) array of Ritz vectors in parameter space
    """
    # Lanczos vectors
    V = torch.zeros(T, dim, device=device)
    alphas = torch.zeros(T, device=device)
    betas = torch.zeros(T, device=device)

    # Initial vector
    if v0 is None:
        v = torch.randn(dim, device=device)
    else:
        v = v0.to(device)
    v = v / v.norm()
    V[0] = v

    # First step
    w = hvp_fn(v)
    alpha = w.dot(v)
    alphas[0] = alpha
    w = w - alpha * v

    for j in range(1, T):
        beta = w.norm()
        betas[j] = beta

        if beta < 1e-10:
            # Early termination — invariant subspace found
            T_actual = j
            break

        v_prev = V[j - 1]
        v = w / beta
        V[j] = v

        # HVP
        w = hvp_fn(v)
        alpha = w.dot(v)
        alphas[j] = alpha
        w = w - alpha * v - beta * v_prev

        # Full reorthogonalization (critical for numerical stability)
        for k in range(j + 1):
            coeff = w.dot(V[k])
            w = w - coeff * V[k]
    else:
        T_actual = T

    # Build tridiagonal matrix and solve
    alphas_np = alphas[:T_actual].cpu().numpy()
    betas_np = betas[1:T_actual].cpu().numpy()

    # Tridiagonal eigendecomposition
    T_mat = np.diag(alphas_np) + np.diag(betas_np, 1) + np.diag(betas_np, -1)
    eig_vals, eig_vecs_tri = np.linalg.eigh(T_mat)

    # Convert Ritz vectors back to parameter space
    V_np = V[:T_actual].cpu().numpy()  # (T_actual, dim)
    ritz_vecs = eig_vecs_tri.T @ V_np  # (T_actual, dim)

    return eig_vals, ritz_vecs


def compute_extreme_eigenvalues(sampler, energy_fn, batch_size: int,
                                 n_hvp_avg: int, T_lanczos: int,
                                 device: torch.device):
    """Full pipeline: compute lambda_min and lambda_max via Lanczos.

    Args:
        sampler: loaded CategoryASampler at checkpoint
        energy_fn: benchmark energy function
        batch_size: HVP batch size (e.g. 1024)
        n_hvp_avg: number of batches to average per HVP (e.g. 8)
        T_lanczos: Lanczos steps (e.g. 50)
        device: cuda device

    Returns:
        dict with lambda_min, lambda_max, all eigenvalues, top Ritz vectors
    """
    params = list(sampler.parameters())
    dim = sum(p.numel() for p in params)

    def hvp_fn(v_flat):
        loss_fn = make_averaged_loss_fn(sampler, energy_fn, batch_size, n_hvp_avg, device)
        return hvp_flat(loss_fn, params, v_flat)

    eig_vals, ritz_vecs = lanczos(hvp_fn, dim, T_lanczos, device)

    return {
        "lambda_min": float(eig_vals[0]),
        "lambda_max": float(eig_vals[-1]),
        "eigenvalues": eig_vals.tolist(),
        "dim": dim,
        "T_lanczos": T_lanczos,
        # Store top-5 smallest and largest Ritz vectors
        "ritz_vecs_min5": ritz_vecs[:5].tolist(),
        "ritz_vecs_max5": ritz_vecs[-5:].tolist(),
    }
