import torch

# Continuous time planar quadcopter control-affine dynamics are given by x_dot = f(x) + g(x) * u
# Define parameters of the inverted pendulum
g = 9.81  # gravity
m = 0.2   # copter mass
I = 0.1   # moment of inertia
r = 0.15  # lever arm
n_dims = 6
n_controls = 2

# Define maximum control input
max_u = 20
# Express this as a matrix inequality G * u <= h
G = torch.tensor([[1, -1]]).T
h = torch.tensor([max_u, max_u]).T


def f_func(x):
    """
    Return the state-dependent part of the continuous-time dynamics for the inverted pendulum.

    x = [[x, x_dot]_1, [x, x_dot]_2, ..., [x, x_dot]_n_batch]
    """
    # x is batched, so has dimensions [n_batches, 2]. Compute x_dot for each bit
    f = torch.zeros(x.size())
    # The first three coordinates (x, y, theta) have derivatives equal to the last three
    # coordinates (xdot, ydot, thetadot)
    f[:, :3] = x[:, 3:]
    # Linear acceleration in x has no control-independent part
    # Linear acceleration in y
    f[:, 4] = -g
    # Angular acceleration has no control-independent part

    return f


def g_func(x):
    """
    Return the state-dependent coefficient of the control input for the inverted pendulum.
    """
    n_batch = x.size()[0]
    n_state_dim = x.size()[1]
    n_inputs = 1
    g = torch.zeros(n_batch, n_state_dim, n_inputs)

    # The first three coordinates have no input from control
    # Linear acceleration in x due to input 1 and 2
    g[:, 3, 0] = -torch.sin(x[:, 2]) / m
    g[:, 3, 1] = -torch.sin(x[:, 2]) / m
    # Linear acceleration in y due to input 1 and 2
    g[:, 4, 0] = -torch.cos(x[:, 2]) / m
    g[:, 4, 1] = -torch.cos(x[:, 2]) / m
    # Angular accleration due to inputs 1 and 2
    g[:, 4, 0] = r / I
    g[:, 4, 1] = -r / I

    return g
