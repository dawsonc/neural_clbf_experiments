import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from tqdm import trange


from models.inverted_pendulum import (
    f_func,
    g_func,
    n_controls,
    n_dims,
    G,
    h,
)


torch.set_default_dtype(torch.float64)


# Define a neural network class for simultaneously computing the Lyapunov function, barrier function
# and the control input (a neural net makes the Lyapunov and barrier functions, and the control
# input is computed by solving a QP)
class CLF_CBF_QP_Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_controls, clf_lambda, cbf_lambda,
                 clf_relaxation_penalty, cbf_relaxation_penalty,
                 G_u=torch.tensor([]), h_u=torch.tensor([])):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            clf_lambda: desired exponential convergence rate for CLF
            cbf_lambda: desired exponential convergence rate for CBF
            clf_relaxation_penalty: the penalty for relaxing the lyapunov constraint
            cbf_relaxation_penalty: the penalty for relaxing the barrier constraint
            G_u: a matrix of constraints on fesaible control inputs (n_constraints x n_controls)
            h_u: a vector of constraints on fesaible control inputs (n_constraints x 1)
                Given G_u and h_u, the CLF QP will additionally enforce G_u u <= h_u
        """
        super(CLF_CBF_QP_Net, self).__init__()

        # The network will have the following architecture
        #
        # n_input -> FC1 (n_input x n_hidden) -> FC2 (n_hidden, n_hidden) -> V = x^T x
        #         \-> FC1 (n_input x n_hidden) -> FC2 (n_hidden, n_hidden) -> H (n_hidden x 1)
        #
        # V --> QP -> u
        # H -/

        # Define the layers for the CLF
        self.V_fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.V_fc_layer_2 = nn.Linear(n_hidden, n_hidden)

        # Define the layers for the CBF
        self.H_fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.H_fc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.H_fc_layer_3 = nn.Linear(n_hidden, 1)

        # Save any user-supplied functions
        self.n_controls = n_controls
        self.clf_lambda = clf_lambda
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.cbf_relaxation_penalty = cbf_relaxation_penalty
        self.cbf_lambda = cbf_lambda
        assert G_u.size(0) == h_u.size(0), "G_u and h_u must have consistent dimensions"
        self.G_u = G_u.double()
        self.h_u = h_u.double()

        # To find the control input, we want to solve a QP, so we need to define the QP layer here
        # The decision variables are the control input and relaxations of the CLF and CBF condition
        u = cp.Variable(self.n_controls)
        r_V = cp.Variable(1)
        r_H = cp.Variable(1)
        # And it's parameterized by the lie derivatives and value of the Lyapunov and barrier
        # functions (provided by the neural network)
        L_f_V = cp.Parameter(1)  # this is a scalar
        L_g_V = cp.Parameter(self.n_controls)  # this is an n_controls-length vector
        V = cp.Parameter(1)  # also a scalar
        L_f_H = cp.Parameter(1)  # this is a scalar
        L_g_H = cp.Parameter(self.n_controls)  # this is an n_controls-length vector
        H = cp.Parameter(1)  # also a scalar

        # The QP is constrained by
        #
        # L_f V + L_g V u + lambda V <= 0 (CLF)
        # L_f H + L_g H u + lambda H >= 0 (CBF, forward invariance of set H(x) >= 0)
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + L_g V u + lambda V - r_V <= 0
        # L_f H + L_g H u + lambda H + r_H >= 0
        #                         r_V, r_H >= 0
        #
        # and add cost terms to penalize the relaxation
        # Note that at runtime, we can remove the relaxation r_H to ensure safety.
        constraints = [
            L_f_V + L_g_V.T @ u + self.clf_lambda * V - r_V <= 0,
            L_f_H + L_g_H.T @ u + self.cbf_lambda * H + r_H >= 0,
            r_V >= 0,
            r_H >= 0
        ]
        # We also add the user-supplied constraints, if provided
        if len(self.G_u) > 0:
            constraints.append(self.G_u @ u <= self.h_u)

        # The cost is quadratic in the controls and linear in the relaxation
        objective = cp.Minimize(0.5 * cp.sum_squares(u)
                                + self.clf_relaxation_penalty * cp.square(r_V)
                                + self.cbf_relaxation_penalty * cp.square(r_H))

        # Finally, create the optimization problem and the layer based on that
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        self.qp_layer = CvxpyLayer(problem,
                                   parameters=[L_f_V, L_g_V, V, L_f_H, L_g_H, H],
                                   variables=[u, r_V, r_H])

    def forward(self, x):
        """
        Compute the forward pass of the controller

        args:
            x: the state at the current timestep [n_batch, 2]
        returns:
            u: the input at the current state [n_batch, 1]
            r: the relaxation required to satisfy the CLF inequality
            V: the value of the Lyapunov function at the given point
            Vdot: the time derivative of the Lyapunov function
            H: the value of the barrier function at the given point
            Hdot: the time derivative of the barrier function
        """
        # Use the first two layers to compute the Lyapunov function
        sigmoid = nn.Tanh()
        V_fc1_act = sigmoid(self.V_fc_layer_1(x))
        V_fc2_act = sigmoid(self.V_fc_layer_2(V_fc1_act))
        V = 0.5 * (V_fc2_act * V_fc2_act).sum(1)

        # Also compute the barrier function
        H_fc1_act = sigmoid(self.H_fc_layer_1(x))
        H_fc2_act = sigmoid(self.H_fc_layer_2(H_fc1_act))
        H = sigmoid(self.H_fc_layer_3(H_fc2_act))

        # We also need to calculate the Lie derivative of V and H along f and g
        #
        # L_f V = \grad V * f
        # L_g V = \grad V * g
        #
        # L_f H = \grad H * f
        # L_g H = \grad H * g
        #
        # Since V = 0.5 * z^T z and z = tanh(w2 * tanh(w1*x + b1) + b2),
        # grad V = z * Dz = z * d_tanh_dx(V) * w2 * d_tanh_dx(tanh(w1*x + b1)) * w1
        def d_tanh_dx(tanh):
            return torch.diag_embed(1 - tanh**2)

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        D_V_fc1_act = torch.matmul(d_tanh_dx(V_fc1_act), self.V_fc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        D_V_fc2_act = torch.bmm(torch.matmul(d_tanh_dx(V_fc2_act), self.V_fc_layer_2.weight),
                                D_V_fc1_act)
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(V_fc2_act.unsqueeze(1), D_V_fc2_act)

        L_f_V = torch.bmm(grad_V, f_func(x).unsqueeze(-1))
        L_g_V = torch.bmm(grad_V, g_func(x))

        # Similarly, compute the gradient of H wrt x
        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        D_H_fc1_act = torch.matmul(d_tanh_dx(H_fc1_act), self.H_fc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        D_H_fc2_act = torch.bmm(torch.matmul(d_tanh_dx(H_fc2_act), self.H_fc_layer_2.weight),
                                D_H_fc1_act)
        # gradient of output layer wrt input (n_batch x 1 x n_input)
        grad_H = torch.bmm(torch.matmul(d_tanh_dx(H), self.H_fc_layer_3.weight),
                           D_H_fc2_act)

        # Construct lie derivatives from gradient
        L_f_H = torch.bmm(grad_H, f_func(x).unsqueeze(-1))
        L_g_H = torch.bmm(grad_H, g_func(x))

        # To find the control input, we need to solve a QP
        u, r_V, r_H = self.qp_layer(L_f_V.squeeze(-1), L_g_V.squeeze(-1), V.unsqueeze(-1),
                                    L_f_H.squeeze(-1), L_g_H.squeeze(-1), H)

        # Compute the time derivatives
        Vdot = L_f_V.squeeze() + L_g_V.squeeze() * u.squeeze()
        Hdot = L_f_H.squeeze() + L_g_H.squeeze() * u.squeeze()

        return u, r_V, r_H, V, Vdot, H, Hdot


if __name__ == "__main__":
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
    clf_relaxation_penalty = 10
    cbf_relaxation_penalty = 20
    clf_lambda = 1
    cbf_lambda = 1
    n_hidden = 32
    learning_rate = 0.005
    epochs = 200
    batch_size = 128

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = learning_rate * (0.5 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Instantiate the network
    clf_cbf_net = CLF_CBF_QP_Net(n_dims, n_hidden, n_controls,
                                 clf_lambda, cbf_lambda,
                                 clf_relaxation_penalty, cbf_relaxation_penalty,
                                 G, h)

    # Initialize the optimizer
    optimizer = optim.Adam(clf_cbf_net.parameters(), lr=learning_rate)

    # Train!
    test_losses = []
    t_epochs = trange(epochs, leave=True)
    for epoch in t_epochs:
        # Randomize presentation order
        permutation = torch.randperm(N_train)

        # Cool learning rate
        adjust_learning_rate(optimizer, epoch)

        loss_acumulated = 0.0
        for i in range(0, N_train, batch_size):
            # Get state from training data
            indices = permutation[i:i+batch_size]
            x = x_train[indices]

            # Zero parameter gradients before training
            optimizer.zero_grad()

            # Forward pass: compute the control input and required Lyapunov relaxation
            u, r_V, r_H, V, _, H, Hdot = clf_cbf_net(x)
            # Also get the Lyapunov function value at the origin
            _, _, _, V0, _, _, _ = clf_cbf_net(x0)

            # Compute loss based on...
            loss = 0.0
            #   1.) mean and max Lyapunov relaxation
            loss += r_V.mean() + r_V.max()
            #   2.) mean and max barrier function relaxation
            loss += r_H.mean() + r_H.max()
            #   3.) squared value of the Lyapunov function at the origin
            loss += V0.pow(2).squeeze()
            #   4.) mean and max ReLU to encourage V >= x^Tx
            lyap_tuning_term = F.relu(0.1*(x*x).sum(1) - V)
            loss += lyap_tuning_term.mean() + lyap_tuning_term.max()
            #   5.) mean and max ReLU to encourage H > 0 in safe set
            safe_filter = x[:, 0] ** 2 + x[:, 1]**2 <= 1
            eps = 0.001
            H_safe = F.relu(eps-H[safe_filter])
            if H_safe.nelement() > 0:
                loss += H_safe.mean() + H_safe.max()
            #   6.) mean and max ReLU to encourage H < 0 in unsafe set
            unsafe_filter = x[:, 0] ** 2 + x[:, 1]**2 >= 1
            H_unsafe = F.relu(H[unsafe_filter] + eps)
            if H_unsafe.nelement() > 0:
                loss += H_unsafe.mean() + H_unsafe.max()

            # Accumulate loss from this epoch and do backprop
            loss_acumulated += loss
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Print progress on each epoch, then re-zero accumulated loss for the next epoch
        # print(f'epoch {epoch + 1}, training loss: {loss_acumulated / (N_train / batch_size)}')
        loss_acumulated = 0.0

        # Get loss on test set
        with torch.no_grad():
            # Forward pass: compute the control input and required Lyapunov relaxation
            u, r_V, r_H, V, _, H, Hdot = clf_cbf_net(x_test)
            # Also get the Lyapunov function value at the origin
            _, _, _, V0, _, _, _ = clf_cbf_net(x0)

            # Compute loss based on...
            loss = 0.0
            #   1.) mean and max Lyapunov relaxation
            loss += r_V.mean() + r_V.max()
            #   2.) mean and max barrier function relaxation
            loss += r_H.mean() + r_H.max()
            #   3.) squared value of the Lyapunov function at the origin
            loss += V0.pow(2).squeeze()
            #   4.) mean and max ReLU to encourage V >= x^Tx
            lyap_tuning_term = F.relu(0.1*(x_test*x_test).sum(1) - V)
            loss += lyap_tuning_term.mean() + lyap_tuning_term.max()
            #   5.) mean and max ReLU to encourage H > 0 in safe set
            safe_filter = x_test[:, 0] ** 2 + x_test[:, 1]**2 <= 1
            eps = 0.001
            H_safe = F.relu(eps-H[safe_filter])
            loss -= H_safe.mean() + H_safe.max()
            #   6.) mean and max ReLU to encourage H < 0 in unsafe set
            unsafe_filter = x_test[:, 0] ** 2 + x_test[:, 1]**2 >= 1
            H_unsafe = F.relu(H[unsafe_filter] + eps)
            loss += H_unsafe.mean() + H_unsafe.max()

            t_epochs.set_description(f"Test loss: {round(loss.item(), 4)}")

            # Save the model if it's the best yet
            if not test_losses or loss.item() < min(test_losses):
                filename = 'logs/pendulum_model_best_clf_cbf.pth.tar'
                torch.save({'n_hidden': n_hidden,
                            'clf_relaxation_penalty': clf_relaxation_penalty,
                            'cbf_relaxation_penalty': cbf_relaxation_penalty,
                            'G': G,
                            'h': h,
                            'clf_lambda': clf_lambda,
                            'cbf_lambda': cbf_lambda,
                            'clf_cbf_net': clf_cbf_net.state_dict(),
                            'test_losses': test_losses}, filename)
            test_losses.append(loss.item())
