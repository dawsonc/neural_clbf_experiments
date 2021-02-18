import torch
import numpy as np

from models.utils import lqr

# Continuous time 9-dof quadcopter control-affine dynamics are given by x_dot = f(x) + g(x) * u
#
# Also oddly, z is defined pointing downwards.

# For the planar vertical takeoff and landing system (PVTOL), the state variables are
#   x, y, theta, vx, vy, thetadot

# Dynamics from Sun et al. C3M. but without force as a state

# Define parameters of the inverted pendulum
g = 9.81  # gravity

n_dims = 9
n_controls = 4


class StateIndex:
    'list of static state indices'

    PX = 0
    PY = 1
    PZ = 2

    VX = 3
    VY = 4
    VZ = 5

    PHI = 6
    THETA = 7
    PSI = 8


def f_func(x, **kwargs):
    """
    Return the state-dependent part of the continuous-time dynamics for the pvtol system

    x = [[px, py, pz, vx, vy, vz, phi, theta, psi]_1, ...]
    """
    # x is batched, so has dimensions [n_batches, n_dims]. Compute x_dot for each bit
    f = torch.zeros_like(x)

    # Derivatives of positions are just velocities
    f[:, StateIndex.PX] = x[:, StateIndex.VX]  # x
    f[:, StateIndex.PY] = x[:, StateIndex.VY]  # y
    f[:, StateIndex.PZ] = x[:, StateIndex.VZ]  # z

    # Constant acceleration in z due to gravity
    f[:, StateIndex.VZ] = g

    # Orientation velocities are directly actuated

    return f


def g_func(x, **kwargs):
    """
    Return the state-dependent coefficient of the control input for the pvtol system.
    """
    n_batch = x.size()[0]
    g = torch.zeros(n_batch, n_dims, n_controls, dtype=x.dtype)

    # Derivatives of linear velocities depend on thrust f
    s_theta = torch.sin(x[:, StateIndex.THETA])
    c_theta = torch.cos(x[:, StateIndex.THETA])
    s_phi = torch.sin(x[:, StateIndex.PHI])
    c_phi = torch.cos(x[:, StateIndex.PHI])
    g[:, StateIndex.VX, 0] = -s_theta
    g[:, StateIndex.VY, 0] = c_theta * s_phi
    g[:, StateIndex.VZ, 0] = -c_theta * c_phi

    # Derivatives of all orientations are control variables
    g[:, StateIndex.PHI:, 1:] = torch.eye(n_controls - 1)

    return g


def control_affine_dynamics(x, **kwargs):
    """
    Return the control-affine dynamics evaluated at the given state

    x = [[x, z, theta, vx, vz, theta_dot]_1, ...]
    """
    return f_func(x), g_func(x)


# Linearize the system about the x = 0, u = [g, 0, 0, 0]
A = np.zeros((n_dims, n_dims))
A[StateIndex.PX, StateIndex.VX] = 1.0
A[StateIndex.PY, StateIndex.VY] = 1.0
A[StateIndex.PZ, StateIndex.VZ] = 1.0

A[StateIndex.VX, StateIndex.THETA] = -g
A[StateIndex.VY, StateIndex.PHI] = g

B = np.zeros((n_dims, n_controls))
B[StateIndex.VZ, 0] = -1.0
B[StateIndex.PHI:, 1:] = torch.eye(n_controls - 1)

# Define cost matrices as identity
Q = np.eye(n_dims)
R = np.eye(n_controls)

# Get feedback matrix
K_np = lqr(A, B, Q, R)


def u_nominal(x, **kwargs):
    """
    Return the nominal controller for the system at state x, given by LQR
    """
    # Compute nominal control from feedback + equilibrium control
    K = torch.tensor(K_np, dtype=x.dtype)
    u_nominal = -(K @ x.T).T
    u_eq = torch.zeros_like(u_nominal)
    u_eq[:, 0] = g

    return u_nominal + u_eq
