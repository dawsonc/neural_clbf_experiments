import torch
import numpy as np

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.lowlevel.low_level_controller import LowLevelController


# AeroBench dynamics for the F16 fighter jet, restricted to longitudinal motion
# This includes state variables:
#   Vt: airspeed
#   alpha: angle of attack
#   theta: pitch angle
#   Q: pitch rate
#   alt: altitude
#   pow: engine power lag
#   Nz control integrator
n_dims = 7
# The controls are:
#   Nz: desired z axis acceleration
#   dt: throttle command
n_controls = 2

# All other states and inputs are zero in the longitudinal model
n_dims_full = 16
n_controls_full = 4


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
    x_i = np.zeros(n_dims_full)
    xdot_i = np.zeros(n_dims_full)
    u_i = np.zeros(n_controls_full)
    for batch in range(n_batch):
        # Copy the longitudinal state into the full state vector
        x_i[0] = x[batch, 0]  # airspeed Vt
        x_i[1] = x[batch, 1]  # angle of attack alpha
        x_i[4] = x[batch, 2]  # pitch angle theta
        x_i[7] = x[batch, 3]  # pitch rate Q
        x_i[11] = x[batch, 4]  # altitude alt
        x_i[12] = x[batch, 5]  # pow: engine power lag
        x_i[13] = x[batch, 6]  # control integrator Nz
        # Same for the control
        u_i[0] = u[batch, 0]  # Nz
        u_i[3] = u[batch, 1]  # Nz

        # Compute derivatives at this point points
        llc = LowLevelController()
        model = "stevens"  # look-up table
        # model = "morelli"  # polynomial fit
        t = 0.0
        xdot_i, _, Nz_i, _, _ = controlled_f16(t, x_i, u_i,
                                               llc, f16_model=model)

        # Extract just the longitudinal state
        xdot[batch, 0] = xdot_i[0]  # airspeed Vt
        xdot[batch, 1] = xdot_i[1]  # angle of attack alpha
        xdot[batch, 2] = xdot_i[4]  # pitch angle theta
        xdot[batch, 3] = xdot_i[7]  # pitch rate Q
        xdot[batch, 4] = xdot_i[11]  # altitude alt
        xdot[batch, 5] = xdot_i[12]  # pow: engine power lag
        xdot[batch, 6] = xdot_i[13]  # control integrator Nz

        # Save the true z acceleration
        Nz[batch, 0] = Nz_i

    if return_Nz:
        return xdot, Nz
    else:
        return xdot


def u_nominal(x, alt_setpoint=3600, vt_setpoint=1500):
    """
    Return the nominal controller for the system at state x

    # Taken from the original AeroBench straight and level example. Does PD control on altitude.
    Original code at:
    https://github.com/stanleybak/AeroBenchVVPython/blob/master/code/aerobench/examples/...
        straight_and_level/run.py
    """
    # todo: torch-ify
    airspeed = x[0]        # Vt            (ft/sec)
    alpha = x[1]           # AoA           (rad)
    theta = x[2]           # Pitch angle   (rad)
    gamma = theta - alpha  # Path angle    (rad)
    h = x[4]               # Altitude      (feet)

    # Proportional Control
    k_alt = 0.025
    h_error = alt_setpoint - h
    Nz = k_alt * h_error  # Allows stacking of cmds

    # (Psuedo) Derivative control using path angle
    k_gamma = 25
    Nz = Nz - k_gamma*gamma

    # try to maintain a fixed airspeed near trim point
    K_vt = 0.25
    throttle = -K_vt * (airspeed - vt_setpoint)

    return Nz, 0, 0, throttle


# def control_affine_dynamics(x):
#     """
#     Return the control-affine dynamics evaluated at the given state
#     """
#     # The f16 model is not batched, so we need to compute f and g for each x separately
#     n_batch = x.size()[0]
#     f = torch.zeros_like(x)
#     g = torch.zeros(n_batch, n_dims, n_controls, dtype=x.dtype)

#     # Convert input to numpy
#     x = x.cpu().detach().numpy()
#     for batch in range(n_batch):
#         # Get the derivatives at each of n_controls + 1 linearly independent points (plus zero)
#         # to fit control-affine dynamics
#         u = np.zeros((1, n_controls))
#         for i in range(n_controls):
#             u_i = np.zeros((1, n_controls))
#             u_i[0, i] = 1.0
#             u = np.vstack((u, u_i))

#         # Compute derivatives at each of these points
#         llc = LowLevelController()
#         model = "stevens"  # look-up table
#         # model = "morelli"  # polynomial fit
#         t = 0.0
#         xdot = np.zeros((n_controls + 1, n_dims))
#         for i in range(n_controls + 1):
#             xdot[i, :], _, _, _, _ = controlled_f16(t, x[batch, :], u[i, :], llc, f16_model=model)

#         # Run a least-squares regression to fit control-affine dynamics
#         # We want a relationship of the form xdot = f(x) + g(x)*u, or xdot = [f, g]*[1, u]
#         # Augment the inputs with a one column for the control-independent part
#         regressors = np.hstack((np.ones((n_controls + 1, 1)), u))
#         # Compute the least-squares fit and find A^T such that xdot = [1, u] A^T
#         A, residuals, _, _ = np.linalg.lstsq(regressors, xdot, rcond=None)
#         A = A.T
#         # Extract the control-affine fit
#         f[batch, :] = torch.tensor(A[:, 0])
#         g[batch, :] = torch.tensor(A[:, 1:])

#     return f, g
