import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.clf_qp_net import CLF_QP_Net
from models.pvtol import (
    f_func,
    g_func,
    u_nominal,
    n_controls,
    n_dims,
    G,
    h,
    low_m,
    high_m,
    low_I,
    high_I,
)


torch.set_default_dtype(torch.float64)

# First, sample training data uniformly from the state space
N_train = 1000
xy = torch.Tensor(N_train, 2).uniform_(-4, 4)
xydot = torch.Tensor(N_train, 2).uniform_(-10, 10)
theta = torch.Tensor(N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_train, 1).uniform_(-2*np.pi, 2*np.pi)
x_train = torch.cat((xy, theta, xydot, theta_dot), 1)

# Also get some testing data, just to be principled
N_test = 500
xy = torch.Tensor(N_test, 2).uniform_(-4, 4)
xydot = torch.Tensor(N_test, 2).uniform_(-10, 10)
theta = torch.Tensor(N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_test, 1).uniform_(-2*np.pi, 2*np.pi)
x_test = torch.cat((xy, theta, xydot, theta_dot), 1)

# Create a tensor for the origin as well
x0 = torch.zeros(1, 6)

# Define the scenarios
nominal_scenario = {"m": low_m, "inertia": low_I}
scenarios = [
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": high_I},
    {"m": high_m, "inertia": low_I},
    {"m": high_m, "inertia": high_I},
]

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 1.0
clf_lambda = 1
n_hidden = 48
learning_rate = 0.001
epochs = 1000
batch_size = 1  # 64


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF condition, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(clf_net, epoch):
    penalty = relaxation_penalty * (2 ** (epoch // 3))
    clf_net.relaxation_penalty = penalty


# Instantiate the network
clf_net = CLF_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                     f_func, g_func, u_nominal, scenarios, nominal_scenario,
                     allow_relax=False)

# Initialize the optimizer
optimizer = optim.Adam(clf_net.parameters(), lr=learning_rate)

# Train!
test_losses = []
for epoch in range(epochs):
    # Randomize presentation order
    permutation = torch.randperm(N_train)

    # Cool learning rate
    adjust_learning_rate(optimizer, epoch)
    # And follow the relaxation penalty schedule
    adjust_relaxation_penalty(clf_net, epoch)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Forward pass: compute the control input and required Lyapunov relaxation
        u, r, V, Vdot = clf_net(x)
        # Also get the Lyapunov function value at the origin
        _, _, V0, _ = clf_net(x0)

        # Compute loss based on...
        loss = 0.0
        #   1.) mean and max Lyapunov relaxation
        loss += r.mean()
        #   3.) squared value of the Lyapunov function at the origin
        loss += V0.pow(2).squeeze()
        #   4.) term to encourage satisfaction of CLF condition
        lyap_descent_term = F.relu(Vdot.squeeze() + clf_lambda * V)
        loss += lyap_descent_term.mean()

        # Accumulate loss from this epoch and do backprop
        loss.backward()
        loss_acumulated += loss.detach()

        # Update the parameters
        optimizer.step()

    # Print progress on each epoch, then re-zero accumulated loss for the next epoch
    print(f'Epoch {epoch + 1} training loss: {loss_acumulated / (N_train / batch_size)}')
    loss_acumulated = 0.0

    # Get loss on test set
    with torch.no_grad():
        u, r, V, Vdot = clf_net(x_test)
        _, _, V0, _ = clf_net(x0)

        # Compute loss based on...
        loss = 0.0
        #   1.) mean and max Lyapunov relaxation
        loss += r.mean()
        #   3.) squared value of the Lyapunov function at the origin
        loss += V0.pow(2).squeeze()
        #   4.) term to encourage satisfaction of CLF condition
        lyap_descent_term = F.relu(Vdot.squeeze() + clf_lambda * V)
        loss += lyap_descent_term.mean()

        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")
        print(f"                     relaxation: {r.mean().item()}")
        print(f"                         origin: {V0.pow(2).squeeze().item()}")
        print(f"                   descent term: {lyap_descent_term.mean().item()}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            filename = 'logs/pvtol_robust_clf_qp.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'relaxation_penalty': relaxation_penalty,
                        'G': G,
                        'h': h,
                        'clf_lambda': clf_lambda,
                        'clf_net': clf_net.state_dict()}, filename)
        test_losses.append(loss.item())
