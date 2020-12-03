import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qpth.qp import QPFunction

import matplotlib.pyplot as plt
from matplotlib import cm

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


# Define a neural network class for simultaneously computing the Lyapunov function and the control
# input (a neural net makes the Lyapunov function, and the control input is computed by solving
# a QP)
class CLF_QP_Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                 G_u=torch.tensor([]), h_u=torch.tensor([])):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            clf_lambda: desired exponential convergence rate
            relaxation_penalty: the penalty for relaxing the control lyapunov constraint
            G_u: a matrix of constraints on fesaible control inputs (n_constraints x n_controls)
            h_u: a vector of constraints on fesaible control inputs (n_constraints x 1)
                Given G_u and h_u, the CLF QP will additionally enforce G_u u <= h_u
        """
        super(CLF_QP_Net, self).__init__()

        # The network will have the following architecture
        #
        # n_input -> FC1 (n_input x n_hidden) -> FC2 (n_hidden, n_hidden) -> V = x^T x --> QP -> u
        self.fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.fc_layer_2 = nn.Linear(n_hidden, n_hidden)

        self.n_controls = n_controls
        self.clf_lambda = clf_lambda
        self.relaxation_penalty = relaxation_penalty
        assert G_u.size(0) == h_u.size(0), "G_u and h_u must have consistent dimensions"
        self.G_u = G_u.double()
        self.h_u = h_u.double()

    def forward(self, x):
        """
        Compute the forward pass of the controller

        args:
            x: the state at the current timestep [n_batch, 2]
        returns:
            u: the input at the current state [n_batch, 1]
            r: the relaxation required to satisfy the CLF inequality
            V: the value of the Lyapunov function at a given point
            Vdot: the time derivative of the Lyapunov function
        """
        # Use the first two layers to compute the Lyapunov function
        sigmoid = nn.Tanh()
        fc1_act = sigmoid(self.fc_layer_1(x))
        fc2_act = sigmoid(self.fc_layer_2(fc1_act))
        V = 0.5 * (fc2_act * fc2_act).sum(1)

        # We also need to calculate the Lie derivative of V along f and g
        #
        # L_f V = \grad V * f
        # L_g V = \grad V * g
        #
        # Since V = tanh(w2 * tanh(w1*x + b1) + b1),
        # grad V = d_tanh_dx(V) * w2 * d_tanh_dx(tanh(w1*x + b1)) * w1
        def d_tanh_dx(tanh):
            return torch.diag_embed(1 - tanh**2)

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        Dfc1_act = torch.matmul(d_tanh_dx(fc1_act), self.fc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        Dfc2_act = torch.bmm(torch.matmul(d_tanh_dx(fc2_act), self.fc_layer_2.weight), Dfc1_act)
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(fc2_act.unsqueeze(1), Dfc2_act)

        L_f_V = torch.bmm(grad_V, f_func(x).unsqueeze(-1))
        L_g_V = torch.bmm(grad_V, g_func(x))

        # To find the control input, we want to solve a QP constrained by
        #
        # L_f V + L_g V u + lambda V <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        #
        # and add the cost term relaxation_penalty * r.
        #
        # The decision variables here are z=[u r], so our quadratic cost is 1/2 z^T Q z + p^T z,
        # where
        #           Q = I
        #           p = [0 relaxation_penalty]
        #
        # Expressing the constraints formally:
        #
        #       Gz <= h
        #
        # where h = [-L_f V - lambda V, 0]^T and G = [L_g V, -1
        #                                             0,     -1]
        # We also add the user-specified inequality constraints
        Q = torch.eye(self.n_controls + 1).double()
        p = torch.zeros(Q.size(0)).double()
        p[-1] = self.relaxation_penalty
        G = torch.zeros(x.size(0), 2 + self.G_u.size(0), self.n_controls + 1).double()
        G[:, 0, :self.n_controls] = L_g_V.squeeze(1)
        G[:, 0, -1] = -1
        G[:, 1, -1] = -1
        G[:, 2:, :self.n_controls] = self.G_u
        h = torch.zeros(x.size(0), 2 + self.h_u.size(0), 1)
        h[:, 0, 0] = -L_f_V.squeeze() - self.clf_lambda * V
        h[:, 1, 0] = 0.0
        h[:, 2:, 0] = self.h_u.view(1, self.h_u.size(0))
        h = h.squeeze()
        # No equality constraints
        A = torch.tensor([])
        b = torch.tensor([])

        # Solve the QP!
        result = QPFunction(verbose=False)(Q, p, G, h, A, b)
        # Extract the results
        u = result[:, :self.n_controls]
        relaxation = result[:, -1]

        Vdot = L_f_V.squeeze() + L_g_V.squeeze(1) * u

        return u, relaxation, V, Vdot


# Now it's time to learn. First, sample training data uniformly from the state space
N_train = 1000
theta = torch.Tensor(N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_train, 1).uniform_(-2*np.pi, 2*np.pi)
x_train = torch.cat((theta, theta_dot), 1)

# Also get some testing data, just to be principled
N_test = 500
theta = torch.Tensor(N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_test, 1).uniform_(-2*np.pi, 2*np.pi)
x_test = torch.cat((theta, theta_dot), 1)

# Create a tensor for the origin as well
x0 = torch.zeros(1, 2)

# Define hyperparameters
relaxation_penalty = 10
clf_lambda = 1
n_hidden = 64
learning_rate = 0.0001
momentum = 0.1
epochs = 100
batch_size = 64

# Instantiate the network
clf_net = CLF_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty, G, h)

# Initialize the optimizer
optimizer = optim.SGD(clf_net.parameters(), lr=learning_rate, momentum=momentum)

# Train!
epoch_test_losses = []
epoch_test_max_relaxation = []
for epoch in range(epochs):
    # Randomize presentation order
    permutation = torch.randperm(N_train)

    loss_acumulated = 0.0
    for i in range(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Forward pass: compute the control input and required Lyapunov relaxation
        u, r, V, _ = clf_net(x)
        # Also get the Lyapunov function value at the origin
        _, _, V0, _ = clf_net(x0)

        # Compute loss based on...
        #   1.) mean Lyapunov relaxation
        #   2.) squared value of the Lyapunov function at the origin
        #   3.) ReLU to encourage V >= x^Tx
        #   4.) Maximum Lyapunov relaxation
        loss = 0.0
        loss += r.mean()
        loss += V0.pow(2).squeeze()
        loss += F.relu(0.1*(x*x).sum(1) - V).max()
        loss += F.relu(0.1*(x*x).sum(1) - V).mean()
        loss += r.max()

        # Accumulate loss from this epoch and do backprop
        loss_acumulated += loss
        loss.backward()

        # Update the parameters
        optimizer.step()

    # Print progress on each epoch, then re-zero accumulated loss for the next epoch
    print(f'epoch {epoch + 1}, training loss: {loss_acumulated / (N_train / batch_size)}')
    loss_acumulated = 0.0

    # Get loss on test set
    with torch.no_grad():
        u, r, V, _ = clf_net(x_test)
        _, _, V0, _ = clf_net(x0)
        loss = 0.0
        loss += r.mean()
        loss += V0.pow(2).squeeze()
        loss += F.relu(0.1*(x_test*x_test).sum(1) - V).max()
        loss += F.relu(0.1*(x_test*x_test).sum(1) - V).mean()
        loss += r.max()
        print(f"\tTest loss: {loss}")
        print(f"\tTest max relaxation: {r.max()}")
        print(f"\tTest max quadratic violation: {F.relu(0.1*(x_test*x_test).sum(1) - V).max()}")
        epoch_test_losses.append(loss.item())
        epoch_test_max_relaxation.append(r.max().item())

print("Done! Test loss sequence: ")
print(epoch_test_losses)
print("Test max relaxation sequence: ")
print(epoch_test_max_relaxation)

# Make a grid and plot the relaxation
with torch.no_grad():
    n_theta = 30
    n_theta_dot = 30
    theta = torch.linspace(-np.pi, np.pi, n_theta)
    theta_dot = torch.linspace(-np.pi, np.pi, n_theta_dot)
    grid_theta, grid_theta_dot = torch.meshgrid(theta, theta_dot)
    residuals = torch.zeros(n_theta, n_theta_dot)
    V_values = torch.zeros(n_theta, n_theta_dot)
    V_dot_values = torch.zeros(n_theta, n_theta_dot)
    for i in range(n_theta):
        print(i)
        for j in range(n_theta_dot):
            # Get the residual from running the model
            x = torch.tensor([[theta[i], theta_dot[j]]])
            _, r, V, V_dot = clf_net(x)
            residuals[i, j] = r
            V_values[i, j] = V
            V_dot_values[i, j] = V_dot
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, residuals.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('Residual')
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, V_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$V$')
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, V_dot_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$\\dot{V}$')
    #
    plt.show()

# Simulate some results
with torch.no_grad():
    N_sim = 1
    theta = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    theta_dot = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    x_sim_start = torch.cat((theta, theta_dot), 1)
    #
    t_sim = 0.5
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)
    x_sim = torch.zeros(num_timesteps, N_sim, 2)
    u_sim = torch.zeros(num_timesteps, N_sim, 1)
    V_sim = torch.zeros(num_timesteps, N_sim, 1)
    Vd_sim = torch.zeros(num_timesteps, N_sim, 1)
    x_sim[0, :, :] = x_sim_start
    clf_net.relaxation_penalty = 500
    for tstep in range(1, num_timesteps):
        print(tstep)
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state
        u, r, V, Vdot = clf_net(x_current)
        # Get the dynamics
        f_val = f_func(x_current)
        g_val = g_func(x_current)
        # Take one step to the future
        xdot = f_val.unsqueeze(-1) + torch.bmm(g_val, u.unsqueeze(-1))
        x_sim[tstep, :, :] = x_current + delta_t * xdot.squeeze()
        u_sim[tstep, :, 0] = u
        V_sim[tstep, :, 0] = V
        Vd_sim[tstep, :, 0] = Vdot
    #
    t = np.linspace(0, t_sim, num_timesteps)
    ax = plt.subplot(5, 1, 1)
    ax.plot(t, x_sim[:, :, 0])
    ax.set_xlabel("timestep")
    ax.set_ylabel("$\\theta$")
    ax = plt.subplot(5, 1, 2)
    ax.plot(t, x_sim[:, :, 1])
    ax.set_xlabel("timestep")
    ax.set_ylabel("$\\dot{\\theta}$")
    ax = plt.subplot(5, 1, 3)
    ax.plot(t, u_sim[:, :, 0])
    ax.set_xlabel("timestep")
    ax.set_ylabel("$u$")
    ax = plt.subplot(5, 1, 4)
    ax.plot(t, V_sim[:, :, 0], 'x')
    ax.set_xlabel("timestep")
    ax.set_ylabel("$V$")
    ax = plt.subplot(5, 1, 5)
    ax.plot(t, Vd_sim[:, :, 0], 'x')
    ax.set_xlabel("timestep")
    ax.set_ylabel("$Vdot$")
    plt.show()
