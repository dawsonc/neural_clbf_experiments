import torch
import numpy as np

from models.utils import lqr

# Continuous time planar quadcopter control-affine dynamics are given by x_dot = f(x) + g(x) * u

# For the planar vertical takeoff and landing system (PVTOL), the state variables are
#   x, y, theta, vx, vy, thetadot

# Dynamics from Dawei's paper

# Define parameters of the inverted pendulum
g = 9.81  # gravity
# copter mass lower and upper bounds
# low_m = 0.2
# high_m = 0.4
# # moment of inertia lower and upper bounds
# low_I = 0.1
# high_I = 0.2
low_m = 0.486
high_m = 0.486 * 1.5
# moment of inertia lower and upper bounds
low_I = 0.00383
high_I = 0.00383 * 1.5
r = 0.25  # lever arm
n_dims = 6
n_controls = 2

# Define maximum control input
max_u = 100
# Express this as a matrix inequality G * u <= h
G = torch.tensor([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
])
h = torch.tensor([max_u, max_u, max_u, max_u]).T


def f_func(x, m=low_m, inertia=low_I):
    """
    Return the state-dependent part of the continuous-time dynamics for the pvtol system

    x = [[px, pz, phi, vx, pz, phi_dot]_1, ...]
    """
    # x is batched, so has dimensions [n_batches, 2]. Compute x_dot for each bit
    f = torch.zeros_like(x)

    f[:, 0] = x[:, 3] * torch.cos(x[:, 2]) - x[:, 4] * torch.sin(x[:, 2])
    f[:, 1] = x[:, 3] * torch.sin(x[:, 2]) + x[:, 4] * torch.cos(x[:, 2])
    f[:, 2] = x[:, 5]
    f[:, 3] = x[:, 4] * x[:, 5] - g * torch.sin(x[:, 2])
    f[:, 4] = -x[:, 3] * x[:, 5] - g * torch.cos(x[:, 2])
    f[:, 5] = 0.0

    return f


def g_func(x, m=low_m, inertia=low_I):
    """
    Return the state-dependent coefficient of the control input for the pvtol system.
    """
    n_batch = x.size()[0]
    g = torch.zeros(n_batch, n_dims, n_controls, dtype=x.dtype)

    # Effect on z acceleration
    g[:, 4, 0] = 1 / m
    g[:, 4, 1] = 1 / m

    # Effect on heading from rotors
    g[:, 5, 0] = r / inertia
    g[:, 5, 1] = -r / inertia

    return g


def u_nominal(x, m=low_m, inertia=low_I):
    """
    Return the nominal controller for the system at state x, given by LQR
    """
    # Linearize the system about the origin
    A = np.zeros((n_dims, n_dims))
    A[0, 3] = 1.0
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 2] = -g

    B = np.zeros((n_dims, n_controls))
    B[4, 0] = 1.0 / m
    B[4, 1] = 1.0 / m
    B[5, 0] = r / m
    B[5, 1] = -r / m

    # Define cost matrices as identity
    Q = np.eye(n_dims)
    R = np.eye(n_controls)

    # Get feedback matrix
    K = torch.tensor(lqr(A, B, Q, R), dtype=x.dtype)

    # Compute nominal control from feedback
    u_nominal = (K @ x.T).T

    return u_nominal
