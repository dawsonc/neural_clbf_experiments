import torch
import numpy as np

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.lowlevel.low_level_controller import LowLevelController


# AeroBench dynamics for the F16 fighter jet
n_dims = 16
n_controls = 4


def dynamics(x, u, return_Nz=False):
    """
    Return the dynamics evaluated at the given state and control input

    Returns a tuple of xdot and Nz (true vertical acceleration) if return_Nz is True
    """
    # The f16 model is not batched, so we need to compute f and g for each x separately
    n_batch = x.size()[0]
    xdot = torch.zeros_like(x)
    Nz = torch.zeros((n_batch, 1))

    # Convert input to numpy
    x = x.cpu().detach().numpy()
    u = u.cpu().detach().numpy()
    for batch in range(n_batch):
        # Compute derivatives at this point points
        llc = LowLevelController()
        model = "stevens"  # look-up table
        # model = "morelli"  # polynomial fit
        t = 0.0
        xdot_i, _, Nz_i, _, _ = controlled_f16(t, x[batch, :], u[batch, :],
                                               llc, f16_model=model)
        xdot[batch, :] = torch.tensor(xdot_i)

        # Save the true z acceleration
        Nz[batch, 0] = Nz_i

    if return_Nz:
        return xdot, Nz
    else:
        return xdot


def u_nominal(x, alt_setpoint=500, vt_setpoint=500):
    """
    Return the nominal controller for the system at state x
    """
    gcas = GcasAutopilot()

    # The autopilot is not meant to be run on batches so we need to get control inputs separately
    n_batch = x.size()[0]
    x = x.cpu().detach().numpy()
    u = torch.zeros((n_batch, n_controls))

    for batch in range(n_batch):
        # The GCAS autopilot is implemented as a state machine that first rolls and then pulls up.
        # Here we unwrap the state machine logic to get a simpler mapping from state to control

        # If the plane is not hurtling towards the ground, no need to do anything.
        if gcas.is_nose_high_enough(x[batch, :]) or gcas.is_above_flight_deck(x[batch, :]):
            continue

        # If we are hurtling towards the ground and the plane isn't level, we need to try to
        # roll to get level
        if not gcas.is_roll_rate_low(x[batch, :]) or not gcas.are_wings_level(x[batch, :]):
            u[batch, :] = torch.tensor(gcas.roll_wings_level(x[batch, :]))
            continue

        # If we are hurtling towards the ground and the plane IS level, then we need to pull up
        u[batch, :] = torch.tensor(gcas.pull_nose_level())

    return u


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
