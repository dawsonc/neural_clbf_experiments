import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.lf_net_f16_longitudinal import (
    LF_Net,
    lyapunov_loss,
    controller_loss,
)
from models.longitudinal_f16 import (
    dynamics,
    u_nominal,
    n_controls,
    n_dims,
)


torch.set_default_dtype(torch.float64)

# Define the operational domain
vt_min, vt_max = (400, 600)
alpha_min, alpha_max = (-1, 1)
theta_min, theta_max = (-1, 1)
Q_min, Q_max = (-5, 5)
alt_min, alt_max = (200, 800)
pow_min, pow_max = (0, 10)
nz_int_min, nz_int_max = (-20, 20)

# First, sample training data uniformly from the state space
N_train = 100000
x_train = torch.Tensor(N_train, n_dims).uniform_(0, 1)
x_train[:, 0] = x_train[:, 0] * (vt_max - vt_min) + vt_min
x_train[:, 1] = x_train[:, 1] * (alpha_max - alpha_min) + alpha_min
x_train[:, 2] = x_train[:, 2] * (theta_max - theta_min) + theta_min
x_train[:, 3] = x_train[:, 3] * (Q_max - Q_min) + Q_min
x_train[:, 4] = x_train[:, 4] * (alt_max - alt_min) + alt_min
x_train[:, 5] = x_train[:, 5] * (pow_max - pow_min) + pow_min
x_train[:, 6] = x_train[:, 6] * (nz_int_max - nz_int_min) + nz_int_min

# Also get some testing data, just to be principled
N_test = 10000
x_test = torch.Tensor(N_test, n_dims).uniform_(0, 1)
x_test[:, 0] = x_test[:, 0] * (vt_max - vt_min) + vt_min
x_test[:, 1] = x_test[:, 1] * (alpha_max - alpha_min) + alpha_min
x_test[:, 2] = x_test[:, 2] * (theta_max - theta_min) + theta_min
x_test[:, 3] = x_test[:, 3] * (Q_max - Q_min) + Q_min
x_test[:, 4] = x_test[:, 4] * (alt_max - alt_min) + alt_min
x_test[:, 5] = x_test[:, 5] * (pow_max - pow_min) + pow_min
x_test[:, 6] = x_test[:, 6] * (nz_int_max - nz_int_min) + nz_int_min

# Create a tensor for the origin as well, which is our goal
x0 = torch.zeros(1, n_dims)
x0[0, 0] = 500  # Vt
x0[0, 4] = 500  # alt

# Safe and unsafe regions are defined in lf_net_f16 loss functions

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 1.0
clf_lambda = 0.0
safe_level = 1.0
timestep = 0.001
n_hidden = 64
learning_rate = 0.001
epochs = 1000
batch_size = 64


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF condition, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(lf_net, epoch):
    penalty = relaxation_penalty * (2 ** (epoch // 2))
    lf_net.relaxation_penalty = penalty


# Instantiate the network
# filename = "logs/f16_lf_longitudinal.pth.tar"
# checkpoint = torch.load(filename)
lf_net = LF_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                dynamics, u_nominal)
# lf_net.load_state_dict(checkpoint['lf_net'])

# Initialize the optimizer
optimizer = optim.Adam(lf_net.parameters(), lr=learning_rate)

# Train!
test_losses = []
for epoch in range(epochs):
    # Randomize presentation order
    permutation = torch.randperm(N_train)

    # Cool learning rate
    adjust_learning_rate(optimizer, epoch)
    # And follow the relaxation penalty schedule
    adjust_relaxation_penalty(lf_net, epoch)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Compute loss
        loss = 0.0
        loss += lyapunov_loss(x,
                              x0,
                              lf_net,
                              clf_lambda,
                              safe_level,
                              timestep,
                              print_loss=False)
        loss += controller_loss(x, lf_net, print_loss=False)

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
        # Compute loss
        loss = 0.0
        loss += lyapunov_loss(x_test,
                              x0,
                              lf_net,
                              clf_lambda,
                              safe_level,
                              timestep,
                              print_loss=True)
        loss += controller_loss(x_test, lf_net, print_loss=True)
        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            filename = 'logs/f16_lf_longitudinal.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'relaxation_penalty': relaxation_penalty,
                        'safe_level': safe_level,
                        'clf_lambda': clf_lambda,
                        'lf_net': lf_net.state_dict()}, filename)
        test_losses.append(loss.item())
