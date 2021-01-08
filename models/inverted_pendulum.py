import torch

# Continuous time inverted pendulum control-affine dynamics are given by x_dot = f(x) + g(x) * u
# Define parameters of the inverted pendulum (from Chang et al's Neural Lyapunov example)
g = 9.81  # gravity
L = 0.5   # length of the pole
m = 0.15  # ball mass
b = 0.1   # friction
n_dims = 2
n_controls = 1

# Define maximum control input
max_u = 50
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
    f[:, 0] = x[:, 1]
    f[:, 1] = m*g*L*torch.sin(x[:, 0]) - b*x[:, 1]
    f[:, 1] /= m*L**2

    return f


def g_func(x):
    """
    Return the state-dependent coefficient of the control input for the inverted pendulum.
    """
    n_batch = x.size()[0]
    n_state_dim = x.size()[1]
    n_inputs = 1
    g = torch.zeros(n_batch, n_state_dim, n_inputs)
    g[:, 1, :] = 1 / (m*L**2)

    return g
