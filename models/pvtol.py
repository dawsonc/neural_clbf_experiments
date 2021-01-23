import torch

# Continuous time planar quadcopter control-affine dynamics are given by x_dot = f(x) + g(x) * u

# For the planar vertical takeoff and landing system (PVTOL), the state variables are
#   x, y, theta, vx, vy, thetadot

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

    x = [[x, y, theta, vx, vy, thetadot]_1, ...]
    """
    # x is batched, so has dimensions [n_batches, 2]. Compute x_dot for each bit
    f = torch.zeros(x.size())

    f[:, 0] = x[:, 3]
    f[:, 1] = x[:, 4]
    f[:, 2] = x[:, 5]
    f[:, 3] = 0.0
    f[:, 4] = - g
    f[:, 5] = 0.0

    return f


def g_func(x, m=low_m, inertia=low_I):
    """
    Return the state-dependent coefficient of the control input for the pvtol system.
    """
    n_batch = x.size()[0]
    n_state_dim = x.size()[1]
    g = torch.zeros(n_batch, n_state_dim, n_controls)

    # Effect on x acceleration
    g[:, 3, 0] = -torch.sin(x[:, 2]) / m
    g[:, 3, 1] = -torch.sin(x[:, 2]) / m

    # Effect on y acceleration
    g[:, 4, 0] = torch.cos(x[:, 2]) / m
    g[:, 4, 1] = torch.cos(x[:, 2]) / m

    # Effect on heading from rotors
    g[:, 5, 0] = r / inertia
    g[:, 5, 1] = -r / inertia

    return g
