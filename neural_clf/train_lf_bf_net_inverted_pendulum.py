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
# and the control input (a neural net makes the Lyapunov, barrier functions, AND the control input
class LF_BF_QP_Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_controls, lf_lambda, bf_lambda):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            lf_lambda: desired exponential convergence rate for LF
            bf_lambda: desired exponential convergence rate for BF
        """
        super(LF_BF_QP_Net, self).__init__()

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

        # Define the layers for the controller
        self.u_fc_layer_1 = nn.Linear(n_input, n_hidden)
        self.u_fc_layer_2 = nn.Linear(n_hidden, 1)

        # Save any user-supplied functions
        self.n_controls = n_controls
        self.lf_lambda = lf_lambda
        self.bf_lambda = bf_lambda

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

        # And the controlller
        u_fc1_act = sigmoid(self.u_fc_layer_1(x))
        u = sigmoid(self.u_fc_layer_2(u_fc1_act))

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

        # Compute the time derivatives
        Vdot = L_f_V.squeeze() + L_g_V.squeeze() * u.squeeze()
        Hdot = L_f_H.squeeze() + L_g_H.squeeze() * u.squeeze()

        return u, V, Vdot, H, Hdot


if __name__ == "__main__":
    # Now it's time to learn. First, sample training data uniformly from the state space
    N_train = 10000
    theta = torch.Tensor(N_train, 1).uniform_(-np.pi, np.pi)
    theta_dot = torch.Tensor(N_train, 1).uniform_(-2*np.pi, 2*np.pi)
    x_train = torch.cat((theta, theta_dot), 1)

    # Also get some testing data, just to be principled
    N_test = 5000
    theta = torch.Tensor(N_test, 1).uniform_(-np.pi, np.pi)
    theta_dot = torch.Tensor(N_test, 1).uniform_(-2*np.pi, 2*np.pi)
    x_test = torch.cat((theta, theta_dot), 1)

    # Create a tensor for the origin as well
    x0 = torch.zeros(1, 2)

    # Define hyperparameters
    lf_lambda = 1
    bf_lambda = 1
    n_hidden = 64
    learning_rate = 0.0001
    epochs = 500
    batch_size = 256

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = learning_rate * (0.8 ** (epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(lr, 5e-6)

    # Instantiate the network
    lf_bf_net = LF_BF_QP_Net(n_dims, n_hidden, n_controls,
                             lf_lambda, bf_lambda)

    # Initialize the optimizer
    optimizer = optim.Adam(lf_bf_net.parameters(), lr=learning_rate)

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
            u, V, Vdot, H, Hdot = lf_bf_net(x)
            # Also get the Lyapunov function value at the origin
            _, V0, _, _, _ = lf_bf_net(x0)

            # Compute loss based on...
            loss = 0.0
            #   1.) squared value of the Lyapunov function at the origin
            loss += V0.pow(2).squeeze()
            #   2.) mean and max ReLU to encourage V >= x^Tx
            lyap_tuning_term = F.relu(0.1*(x*x).sum(1) - V)
            loss += lyap_tuning_term.mean() + lyap_tuning_term.max()
            #   3.) mean and max violation of the Lyapunov condition
            lyap_violation = F.relu(Vdot + lf_lambda * V)
            loss += lyap_violation.mean() + lyap_violation.max()
            #   4.) mean and max ReLU to encourage H > 0 in safe set
            safe_filter = x[:, 0] ** 2 + x[:, 1]**2 <= 1
            eps = 0.0
            H_safe = F.relu(eps-H[safe_filter])
            if H_safe.nelement() > 0:
                loss += H_safe.mean() + H_safe.max()
            #   5.) mean and max ReLU to encourage H < 0 in unsafe set
            unsafe_filter = x[:, 0] ** 2 + x[:, 1]**2 >= 1
            H_unsafe = F.relu(H[unsafe_filter] + eps)
            if H_unsafe.nelement() > 0:
                loss += H_unsafe.mean() + H_unsafe.max()
            #   6.) mean and max violation of the barrier function condition
            barrier_violation = F.relu(-Hdot - bf_lambda * H)
            loss += barrier_violation.mean() + barrier_violation.max()

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
            u, V, Vdot, H, Hdot = lf_bf_net(x_test)
            # Also get the Lyapunov function value at the origin
            _, V0, _, _, _ = lf_bf_net(x0)

            # Compute loss based on...
            loss = 0.0
            #   1.) squared value of the Lyapunov function at the origin
            loss += V0.pow(2).squeeze()
            #   2.) mean and max ReLU to encourage V >= x^Tx
            lyap_tuning_term = F.relu(0.1*(x_test*x_test).sum(1) - V)
            loss += lyap_tuning_term.mean() + lyap_tuning_term.max()
            #   3.) mean and max violation of the Lyapunov condition
            lyap_violation = F.relu(Vdot + lf_lambda * V)
            loss += lyap_violation.mean() + lyap_violation.max()
            #   4.) mean and max ReLU to encourage H > 0 in safe set
            safe_filter = x_test[:, 0] ** 2 + x_test[:, 1]**2 <= 1
            eps = 0.0
            H_safe = F.relu(eps-H[safe_filter])
            if H_safe.nelement() > 0:
                loss += H_safe.mean() + H_safe.max()
            #   5.) mean and max ReLU to encourage H < 0 in unsafe set
            unsafe_filter = x_test[:, 0] ** 2 + x_test[:, 1]**2 >= 1
            H_unsafe = F.relu(H[unsafe_filter] + eps)
            if H_unsafe.nelement() > 0:
                loss += H_unsafe.mean() + H_unsafe.max()
            #   6.) mean and max violation of the barrier function condition
            barrier_violation = F.relu(-Hdot - bf_lambda * H)
            loss += barrier_violation.mean() + barrier_violation.max()

            t_epochs.set_description(f"Test loss: {round(loss.item(), 4)}")

            # Save the model if it's the best yet
            if not test_losses or loss.item() < min(test_losses):
                filename = 'logs/pendulum_model_best_lf_bf.pth.tar'
                torch.save({'n_hidden': n_hidden,
                            'G': G,
                            'h': h,
                            'lf_lambda': lf_lambda,
                            'bf_lambda': bf_lambda,
                            'lf_bf_net': lf_bf_net.state_dict(),
                            'test_losses': test_losses}, filename)
            test_losses.append(loss.item())
