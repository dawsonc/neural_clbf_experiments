import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qpth.qp import QPFunction


torch.set_default_dtype(torch.float64)

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
h = torch.tensor([[max_u, max_u]]).T


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
    g[:, 1, :] = 1 / m*L**2

    return g


def test(x):
    delta_x = 0.01 * x
    x2 = x + delta_x

    fc_layer_1 = nn.Linear(2, 32)
    fc_layer_2 = nn.Linear(32, 32)

    def d_tanh_dx(tanh):
        return torch.diag_embed(1 - tanh**2)

    # Use the first two layers to compute the Lyapunov function
    sigmoid = nn.Tanh()
    fc1_act = sigmoid(fc_layer_1(x))
    # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
    Dfc1_act = torch.matmul(d_tanh_dx(fc1_act), fc_layer_1.weight)
    fc1_act2 = sigmoid(fc_layer_1(x2))
    # print(fc1_act2 - fc1_act)
    # print(Dfc1_act @ delta_x.T)
    print("this should be 0")
    print(torch.norm((fc1_act2 - fc1_act) - (Dfc1_act @ delta_x.T).squeeze()))

    fc2_act = sigmoid(fc_layer_2(fc1_act))
    fc2_act2 = sigmoid(fc_layer_2(fc1_act2))

    # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
    Dfc2_act = torch.bmm(torch.matmul(d_tanh_dx(fc2_act), fc_layer_2.weight), Dfc1_act)

    # print(fc2_act2 - fc2_act)
    # print(Dfc2_act @ delta_x.T)
    print("this should be 0")
    print(torch.norm((fc2_act2 - fc2_act) - (Dfc2_act @ delta_x.T).squeeze()))

    V = 0.5 * (fc2_act * fc2_act).sum(1)
    V2 = 0.5 * (fc2_act2 * fc2_act2).sum(1)

    # Gradient of V wrt input (n_batch x 1 x n_input)
    grad_V = torch.bmm(fc2_act.unsqueeze(1), Dfc2_act)
    print(V2 - V)
    print(grad_V @ delta_x.T)
    print("this should be 0")
    print(torch.norm((V2 - V) - (grad_V @ delta_x.T).squeeze()))

    # L_f_V = torch.bmm(grad_V, f_func(x).unsqueeze(-1))
    # L_g_V = torch.bmm(grad_V, g_func(x))

    # return V, grad_V


test(torch.tensor([[2.0, 1.1]]))
