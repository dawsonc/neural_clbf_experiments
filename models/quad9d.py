import torch
import numpy as np

from models.utils import lqr

# Continuous time 9-dof quadcopter control-affine dynamics are given by x_dot = f(x) + g(x) * u
# Oddly enough, it has 10 state variables (but 9 dof in the sense of 3D position, velocity, and
# orientation).
#
# Also oddly, z is defined pointing downwards.

# For the planar vertical takeoff and landing system (PVTOL), the state variables are
#   x, y, theta, vx, vy, thetadot

# Dynamics from Sun et al. C3M.

# Define parameters of the inverted pendulum
g = 9.81  # gravity

n_dims = 10
n_controls = 4


class StateIndex:
    'list of static state indices'

    PX = 0
    PY = 1
    PZ = 2

    VX = 3
    VY = 4
    VZ = 5

    F = 6

    PHI = 7
    THETA = 8
    PSI = 9


def f_func(x, **kwargs):
    """
    Return the state-dependent part of the continuous-time dynamics for the pvtol system

    x = [[px, py, pz, vx, vy, vz, f, phi, theta, psi]_1, ...]
    """
    # x is batched, so has dimensions [n_batches, n_dims]. Compute x_dot for each bit
    f = torch.zeros_like(x)

    # Derivatives of positions are just velocities
    f[:, StateIndex.PX] = x[:, StateIndex.VX]  # x
    f[:, StateIndex.PY] = x[:, StateIndex.VY]  # y
    f[:, StateIndex.PZ] = x[:, StateIndex.VZ]  # z

    # Derivatives of velocities depend on thrust f
    s_theta = torch.sin(x[:, StateIndex.THETA])
    c_theta = torch.cos(x[:, StateIndex.THETA])
    s_phi = torch.sin(x[:, StateIndex.PHI])
    c_phi = torch.cos(x[:, StateIndex.PHI])
    f[:, StateIndex.VX] = -x[:, StateIndex.F] * s_theta
    f[:, StateIndex.VY] = x[:, StateIndex.F] * c_theta * s_phi
    f[:, StateIndex.VZ] = g - x[:, StateIndex.F] * c_theta * c_phi

    # Thrust derivative and orientation velocities are directly actuated

    return f


def g_func(x, **kwargs):
    """
    Return the state-dependent coefficient of the control input for the pvtol system.
    """
    n_batch = x.size()[0]
    g = torch.zeros(n_batch, n_dims, n_controls, dtype=x.dtype)

    # Derivatives of thrust and all orientations are control variables
    g[:, StateIndex.F:, :] = torch.eye(n_controls)

    return g


def control_affine_dynamics(x, **kwargs):
    """
    Return the control-affine dynamics evaluated at the given state

    x = [[x, z, theta, vx, vz, theta_dot]_1, ...]
    """
    return f_func(x), g_func(x)


def u_nominal(x, **kwargs):
    """
    Return the nominal controller for the system at state x, given by LQR
    """
    # Linearize the system about the x = 0 (except f = g), u = 0
    A = np.zeros((n_dims, n_dims))
    A[StateIndex.PX, StateIndex.VX] = 1.0
    A[StateIndex.PY, StateIndex.VY] = 1.0
    A[StateIndex.PZ, StateIndex.VZ] = 1.0

    A[StateIndex.VX, StateIndex.THETA] = -g
    A[StateIndex.VY, StateIndex.PHI] = g
    A[StateIndex.VZ, StateIndex.F] = -1

    B = np.zeros((n_dims, n_controls))
    B[StateIndex.F:, :] = torch.eye(n_controls)

    # Define cost matrices as identity
    Q = np.eye(n_dims)
    R = np.eye(n_controls)

    # Get feedback matrix
    K = torch.tensor(lqr(A, B, Q, R), dtype=x.dtype)

    # Compute nominal control from feedback + equilibrium control
    x_eq = torch.zeros_like(x)
    x_eq[:, StateIndex.F] = g
    u_nominal = -(K @ (x - x_eq).T).T
    u_eq = torch.zeros_like(u_nominal)

    return u_nominal + u_eq
