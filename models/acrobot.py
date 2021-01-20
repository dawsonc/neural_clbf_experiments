import torch
import numpy as np

# Continuous time control-affine dynamics are given by x_dot = f(x) + g(x) * u

# Define parameters of the acrobot
g = 9.81   # gravitational acceleration
l1 = 1     # Length of link 1
lc1 = 0.5  # Distance from pivot to link 1 center of mass
m1 = 0.5   # Mass of link 1
I1 = 1     # Rotational inertia of link 1
lc2 = 0.5  # Distance from joint to link 2 center of mass
m2 = 0.5   # Mass of link 2
I2 = 1     # Rotational inertia of link 2
n_dims = 4
n_controls = 1

# Define maximum control input
max_u = 20
# Express this as a matrix inequality G * u <= h
G = torch.tensor([[1, -1]]).T
h = torch.tensor([max_u, max_u]).T


def f_func(x):
    """
    Return the state-dependent part of the continuous-time dynamics for the inverted pendulum.

    x = [[theta1, theta1, omega1, omega2]_1, ..., [theta1, theta1, omega1, omega2]_n_batch]
    """
    # x is batched, so has dimensions [n_batches, 4]. Unpack the state variables
    # Unpack state variables.
    # The dynamics here assume that q1=q2=0 is pointing straight down, so add pi
    # to the angle of the first link.
    q1 = x[:, 0] + np.pi  # angle of link 1 w.r.t. vertical
    q2 = x[:, 1]          # angle of link 2 w.r.t. link 1
    q1_dot = x[:, 2]      # link 1 angular velocity
    q2_dot = x[:, 3]      # link 2 angular velocity

    # Construct manipulator equation matrices
    # Mass matrix
    n_batches = x.size(0)
    M = torch.zeros(n_batches, 2, 2)
    M[:, 0, 0] = I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*torch.cos(q2)
    M[:, 0, 1] = I2 + m2*l1*lc2*torch.cos(q2)
    M[:, 1, 0] = I2 + m2*l1*lc2*torch.cos(q2)
    M[:, 1, 1] = I2
    # Coriolis/damping matrix
    C = torch.zeros(n_batches, 2, 2)
    C[:, 0, 0] = -2*m2*l1*lc2*torch.sin(q2)*q2_dot
    C[:, 0, 1] = -m2*l1*lc2*torch.sin(q2)
    C[:, 1, 0] = m2*l1*lc2*torch.sin(q2)*q1_dot
    # Gravitational torque
    tau_g = torch.zeros(n_batches, 2, 1)
    tau_g[:, 0, 0] = -m1*g*lc1*torch.sin(q1) - m2*g*(l1*torch.sin(q1) + lc2*torch.sin(q1+q2))
    tau_g[:, 1, 0] = -m2*g*lc2*torch.sin(q1+q2)

    # Now construct the state-dependent part of the derivative x_dot
    f = torch.zeros(x.size())
    # The derivatives of the first two coordinates (the joint angles) are simply the last two
    # coordinates (the angular velocities)
    f[:, 0] = x[:, 2]
    f[:, 1] = x[:, 3]
    # The angular accelerations are computed using the manipulator equation matrices
    f[:, 2:] = torch.matmul(torch.inverse(M),
                            tau_g - torch.matmul(C, x[:, 2:].unsqueeze(-1))).squeeze()

    return f


def g_func(x):
    """
    Return the state-dependent coefficient of the control input for the inverted pendulum.
    """
    # x is batched, so has dimensions [n_batches, 4]. Unpack the state variables that we need
    # (for this part, we only need q2)
    q2 = x[:, 1]      # angle of link 2 w.r.t. link 1

    # Construct manipulator equation matrices
    # Mass matrix
    n_batches = x.size(0)
    M = torch.zeros(n_batches, 2, 2)
    M[:, 0, 0] = I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*torch.cos(q2)
    M[:, 0, 1] = I2 + m2*l1*lc2*torch.cos(q2)
    M[:, 1, 0] = I2 + m2*l1*lc2*torch.cos(q2)
    M[:, 1, 1] = I2
    # Actuation matrix
    B = torch.tensor([
        [0.0],
        [1.0]
    ])

    # Now construct the control-affine part of the derivative x_dot
    g = torch.zeros(n_batches, n_dims, n_controls)
    g[:, 2:] = torch.matmul(torch.inverse(M), B)

    return g
