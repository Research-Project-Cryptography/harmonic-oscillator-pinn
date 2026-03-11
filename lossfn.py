"""
Loss functions for the physics-informed harmonic oscillator model.

All loss functions assume **m = 1** (unit mass).  The governing ODE is:

    d²x/dt² + mu * dx/dt + k * x = 0

where  k = w0²  and  mu = 2*d.

The composite loss used during training is:

    L = lambda1 * L_bc1 + lambda2 * L_bc2 + lambda3 * L_phys + lambda4 * L_data

    L_bc1   - boundary condition  x(0) = 1
    L_bc2   - boundary condition  dx/dt(0) = 0
    L_phys  - PDE residual on collocation points
    L_data  - MSE between prediction and noisy observations
"""

from __future__ import annotations

import torch


def mse(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between targets and predictions."""
    return torch.mean((y - y_pred) ** 2)


def boundary_loss(
    prediction: torch.Tensor,
    t_boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boundary / initial-condition loss.

    Enforces the two initial conditions of the underdamped harmonic oscillator:
        x(t=0) = 1     (displacement)
        dx/dt(t=0) = 0 (velocity)

    Parameters
    ----------
    prediction : torch.Tensor
        Model output at t_boundary, shape (1, 1).
    t_boundary : torch.Tensor
        The boundary time point (t=0), shape (1, 1), requires_grad=True.

    Returns
    -------
    loss_x0 : torch.Tensor
        Squared error on x(0) = 1.
    loss_dxdt0 : torch.Tensor
        Squared error on dx/dt(0) = 0.
    """
    loss_x0 = (torch.squeeze(prediction) - 1.0) ** 2

    dxdt = torch.autograd.grad(
        prediction, t_boundary,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
    )[0]
    loss_dxdt0 = (torch.squeeze(dxdt) - 0.0) ** 2

    return loss_x0, loss_dxdt0


def physics_loss(
    prediction: torch.Tensor,
    t_physics: torch.Tensor,
    mu: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """PDE residual loss (m=1).

    Computes  mean[ (d²x/dt² + mu * dx/dt + k * x)² ]  over collocation pts.

    Derivatives are obtained via torch.autograd so that they flow through
    classical layers and through PennyLane's TorchLayer (which implements
    the parameter-shift rule internally).

    Parameters
    ----------
    prediction : torch.Tensor
        Model output at collocation points, shape (N, 1).
    t_physics : torch.Tensor
        Collocation time points, shape (N, 1), requires_grad=True.
    mu : torch.Tensor
        Damping coefficient (may be a trainable parameter).
    k : float
        Spring constant  (k = w0²).

    Returns
    -------
    torch.Tensor
        Scalar PDE residual loss.
    """
    dxdt = torch.autograd.grad(
        prediction, t_physics,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
    )[0]

    d2xdt2 = torch.autograd.grad(
        dxdt, t_physics,
        grad_outputs=torch.ones_like(dxdt),
        create_graph=True,
    )[0]

    residual = d2xdt2 + mu * dxdt + k * prediction
    return torch.mean(residual ** 2)
