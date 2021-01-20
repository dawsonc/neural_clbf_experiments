import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from tqdm import trange


from models.acrobot import (
    f_func,
    g_func,
    n_controls,
    n_dims,
    G,
    h,
)


torch.set_default_dtype(torch.float64)


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
        self.fc_layer_3 = nn.Linear(n_hidden, n_hidden)

        self.n_controls = n_controls
        self.clf_lambda = clf_lambda
        self.relaxation_penalty = relaxation_penalty
        assert G_u.size(0) == h_u.size(0), "G_u and h_u must have consistent dimensions"
        self.G_u = G_u.double()
        self.h_u = h_u.double()

        # To find the control input, we want to solve a QP, so we need to define the QP layer here
        # The decision variables are the control input and relaxation of the CLF condition
        u = cp.Variable(self.n_controls)
        r = cp.Variable(1)
        # And it's parameterized by the lie derivatives and value of the Lyapunov function
        L_f_V = cp.Parameter(1)  # this is a scalar
        L_g_V = cp.Parameter(self.n_controls)  # this is an n_controls-length vector
        V = cp.Parameter(1)  # also a scalar

        # The QP is constrained by
        #
        # L_f V + L_g V u + lambda V <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        #
        # and later add the cost term relaxation_penalty * r.
        constraints = [
            L_f_V + L_g_V.T @ u + self.clf_lambda * V - r <= 0,
            r >= 0
        ]
        # We also add the user-supplied constraints, if provided
        if len(self.G_u) > 0:
            constraints.append(self.G_u @ u <= self.h_u)

        # The cost is quadratic in the controls and linear in the relaxation
        objective = cp.Minimize(0.5 * cp.sum_squares(u) + self.relaxation_penalty * cp.square(r))

        # Finally, create the optimization problem and the layer based on that
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        self.qp_layer = CvxpyLayer(problem, parameters=[L_f_V, L_g_V, V], variables=[u, r])

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
        fc3_act = self.fc_layer_3(fc2_act)
        V = 0.5 * (fc3_act * fc3_act).sum(1)

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
        # Jacobian of third layer wrt input (n_batch x n_hidden x n_input)
        Dfc3_act = torch.matmul(self.fc_layer_3.weight, Dfc2_act)
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(fc3_act.unsqueeze(1), Dfc3_act)

        L_f_V = torch.bmm(grad_V, f_func(x).unsqueeze(-1))
        L_g_V = torch.bmm(grad_V, g_func(x))

        # To find the control input, we need to solve a QP
        u, relaxation = self.qp_layer(L_f_V.squeeze(-1), L_g_V.squeeze(-1), V.unsqueeze(-1))

        Vdot = L_f_V.squeeze() + L_g_V.squeeze() * u.squeeze()

        return u, relaxation, V, Vdot


if __name__ == "__main__":
    # Now it's time to learn. First, sample training data uniformly from the state space
    N_train = 100000
    theta = torch.Tensor(N_train, 2).uniform_(-np.pi, np.pi)
    theta_dot = torch.Tensor(N_train, 2).uniform_(-2*np.pi, 2*np.pi)
    x_train = torch.cat((theta, theta_dot), 1)

    # Also get some testing data, just to be principled
    N_test = 20000
    theta = torch.Tensor(N_test, 2).uniform_(-np.pi, np.pi)
    theta_dot = torch.Tensor(N_test, 2).uniform_(-2*np.pi, 2*np.pi)
    x_test = torch.cat((theta, theta_dot), 1)

    # Create a tensor for the origin as well
    x0 = torch.zeros(1, 4)

    # Define hyperparameters
    relaxation_penalty = 10
    clf_lambda = 1
    n_hidden = 64
    learning_rate = 0.001
    epochs = 10
    batch_size = 32

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = learning_rate * (0.1 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(lr, 1e-6)

    # Instantiate the network
    clf_net = CLF_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty, G, h)

    # Initialize the optimizer
    optimizer = optim.Adam(clf_net.parameters(), lr=learning_rate)

    # Train!
    test_losses = []
    test_max_relaxation = []
    for epoch in range(epochs):
        # Randomize presentation order
        permutation = torch.randperm(N_train)

        # Cool learning rate
        adjust_learning_rate(optimizer, epoch)

        loss_acumulated = 0.0
        batches = trange(0, N_train, batch_size)
        for i in batches:
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
            loss_acumulated += loss.detach()
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Print progress on each epoch, then re-zero accumulated loss for the next epoch
        print(f'Epoch {epoch + 1} training loss: {loss_acumulated / (N_train / batch_size)}')
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
            print(f"Epoch {epoch + 1} test loss: {loss.item()}")

            # Save the model if it's the best yet
            if not test_losses or loss.item() < min(test_losses):
                filename = 'logs/acrobot_model_best.pth.tar'
                torch.save({'n_hidden': n_hidden,
                            'relaxation_penalty': relaxation_penalty,
                            'G': G,
                            'h': h,
                            'clf_lambda': clf_lambda,
                            'clf_net': clf_net.state_dict(),
                            'test_losses': test_losses}, filename)
            test_losses.append(loss.item())
            test_max_relaxation.append(r.max().item())
