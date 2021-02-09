import torch
import numpy as np

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.lowlevel.low_level_controller import LowLevelController


# AeroBench dynamics for the F16 fighter jet
n_dims = 16
n_controls = 4


def control_affine_dynamics(x):
    """
    Return the control-affine dynamics evaluated at the given state
    """
    # The f16 model is not batched, so we need to compute f and g for each x separately
    n_batch = x.size()[0]
    f = torch.zeros_like(x)
    g = torch.zeros(n_batch, n_dims, n_controls, dtype=x.dtype)

    # Convert input to numpy
    x = x.cpu().detach().numpy()
    for batch in range(n_batch):
        # Get the derivatives at each of n_controls + 1 linearly independent points (plus zero)
        # to fit control-affine dynamics
        u = np.zeros((1, n_controls))
        for i in range(n_controls):
            u_i = np.zeros((1, n_controls))
            u_i[0, i] = 1.0
            u = np.vstack((u, u_i))

        # Compute derivatives at each of these points
        llc = LowLevelController()
        model = "stevens"  # look-up table
        # model = "morelli"  # polynomial fit
        t = 0.0
        xdot = np.zeros((n_controls + 1, n_dims))
        for i in range(n_controls + 1):
            xdot[i, :], _, _, _, _ = controlled_f16(t, x[batch, :], u[i, :], llc, f16_model=model)

        # Run a least-squares regression to fit control-affine dynamics
        # We want a relationship of the form xdot = f(x) + g(x)*u, or xdot = [f, g]*[1, u]
        # Augment the inputs with a one column for the control-independent part
        regressors = np.hstack((np.ones((n_controls + 1, 1)), u))
        # Compute the least-squares fit and find A^T such that xdot = [1, u] A^T
        A, residuals, _, _ = np.linalg.lstsq(regressors, xdot, rcond=None)
        A = A.T
        # Extract the control-affine fit
        f[batch, :] = torch.tensor(A[:, 0])
        g[batch, :] = torch.tensor(A[:, 1:])

    return f, g


def u_nominal(x):
    """
    Return the nominal controller for the system at state x, given by LQR
    """
    pass
