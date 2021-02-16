import numpy as np
import torch
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.lf_net_f16_gcas import (
    LF_Net,
    lyapunov_loss,
    controller_loss,
)
from models.f16_full_gcas import (
    dynamics,
    control_affine_dynamics,
    u_nominal,
    n_controls,
    n_dims,
)
from aerobench.util import StateIndex


torch.set_default_dtype(torch.float64)

# Define the operational domain
vt_min, vt_max = (400, 600)
alpha_min, alpha_max = (-np.pi, np.pi)
beta_min, beta_max = (-np.pi, np.pi)
phi_min, phi_max = (-np.pi, np.pi)
theta_min, theta_max = (-np.pi, np.pi)
psi_min, psi_max = (-np.pi, np.pi)
P_min, P_max = (-2 * np.pi, 2 * np.pi)
Q_min, Q_max = (-2 * np.pi, 2 * np.pi)
R_min, R_max = (-2 * np.pi, 2 * np.pi)
pos_n_min, pos_n_max = (0, 0)  # translation invariant
pos_e_min, pos_e_max = (0, 0)  # translation invariant
alt_min, alt_max = (-500, 1100)
pow_min, pow_max = (0, 10)
nz_int_min, nz_int_max = (-20, 20)  # low-level controller internal states
ps_int_min, ps_int_max = (-20, 20)
nyr_int_min, nyr_int_max = (-20, 20)

# First, sample training data uniformly from the state space
N_train = 100000
x_train = torch.Tensor(N_train, n_dims).uniform_(0, 1)
x_train[:, StateIndex.VT] = x_train[:, StateIndex.VT] * (vt_max - vt_min) + vt_min
x_train[:, StateIndex.ALPHA] = x_train[:, StateIndex.ALPHA] * (alpha_max - alpha_min) + alpha_min
x_train[:, StateIndex.BETA] = x_train[:, StateIndex.BETA] * (beta_max - beta_min) + beta_min
x_train[:, StateIndex.PHI] = x_train[:, StateIndex.PHI] * (phi_max - phi_min) + phi_min
x_train[:, StateIndex.THETA] = x_train[:, StateIndex.THETA] * (theta_max - theta_min) + theta_min
x_train[:, StateIndex.PSI] = x_train[:, StateIndex.PSI] * (psi_max - psi_min) + psi_min
x_train[:, StateIndex.P] = x_train[:, StateIndex.P] * (P_max - P_min) + P_min
x_train[:, StateIndex.Q] = x_train[:, StateIndex.Q] * (Q_max - Q_min) + Q_min
x_train[:, StateIndex.R] = x_train[:, StateIndex.R] * (R_max - R_min) + R_min
x_train[:, StateIndex.POSN] = x_train[:, StateIndex.POSN] * (pos_n_max - pos_n_min) + pos_n_min
x_train[:, StateIndex.POSE] = x_train[:, StateIndex.POSE] * (pos_e_max - pos_e_min) + pos_e_min
x_train[:, StateIndex.ALT] = x_train[:, StateIndex.ALT] * (alt_max - alt_min) + alt_min
x_train[:, StateIndex.POW] = x_train[:, StateIndex.POW] * (pow_max - pow_min) + pow_min
x_train[:, 13] = x_train[:, 13] * (nz_int_max - nz_int_min) + nz_int_min
x_train[:, 14] = x_train[:, 14] * (ps_int_max - ps_int_min) + ps_int_min
x_train[:, 15] = x_train[:, 15] * (nyr_int_max - nyr_int_min) + nyr_int_min

# Also get some testing data, to be principled
N_test = 50000
x_test = torch.Tensor(N_test, n_dims).uniform_(0, 1)
x_test[:, StateIndex.VT] = x_test[:, StateIndex.VT] * (vt_max - vt_min) + vt_min
x_test[:, StateIndex.ALPHA] = x_test[:, StateIndex.ALPHA] * (alpha_max - alpha_min) + alpha_min
x_test[:, StateIndex.BETA] = x_test[:, StateIndex.BETA] * (beta_max - beta_min) + beta_min
x_test[:, StateIndex.PHI] = x_test[:, StateIndex.PHI] * (phi_max - phi_min) + phi_min
x_test[:, StateIndex.THETA] = x_test[:, StateIndex.THETA] * (theta_max - theta_min) + theta_min
x_test[:, StateIndex.PSI] = x_test[:, StateIndex.PSI] * (psi_max - psi_min) + psi_min
x_test[:, StateIndex.P] = x_test[:, StateIndex.P] * (P_max - P_min) + P_min
x_test[:, StateIndex.Q] = x_test[:, StateIndex.Q] * (Q_max - Q_min) + Q_min
x_test[:, StateIndex.R] = x_test[:, StateIndex.R] * (R_max - R_min) + R_min
x_test[:, StateIndex.POSN] = x_test[:, StateIndex.POSN] * (pos_n_max - pos_n_min) + pos_n_min
x_test[:, StateIndex.POSE] = x_test[:, StateIndex.POSE] * (pos_e_max - pos_e_min) + pos_e_min
x_test[:, StateIndex.ALT] = x_test[:, StateIndex.ALT] * (alt_max - alt_min) + alt_min
x_test[:, StateIndex.POW] = x_test[:, StateIndex.POW] * (pow_max - pow_min) + pow_min
x_test[:, 13] = x_test[:, 13] * (nz_int_max - nz_int_min) + nz_int_min
x_test[:, 14] = x_test[:, 14] * (ps_int_max - ps_int_min) + ps_int_min
x_test[:, 15] = x_test[:, 15] * (nyr_int_max - nyr_int_min) + nyr_int_min

# Create a bunch of example goal states
N_goal = 100
x_goal = torch.Tensor(N_goal, n_dims).uniform_(0, 1)
x_goal[:, StateIndex.VT] = x_goal[:, StateIndex.VT] * (vt_max - vt_min) + vt_min
x_goal[:, StateIndex.ALPHA] = 0.0
x_goal[:, StateIndex.BETA] = 0.0
x_goal[:, StateIndex.PHI] = 0.0
x_goal[:, StateIndex.THETA] = 0.0
x_goal[:, StateIndex.PSI] = x_goal[:, StateIndex.PSI] * (psi_max - psi_min) + psi_min
x_goal[:, StateIndex.P] = 0.0
x_goal[:, StateIndex.Q] = 0.0
x_goal[:, StateIndex.R] = 0.0
x_goal[:, StateIndex.POSN] = x_goal[:, StateIndex.POSN] * (pos_n_max - pos_n_min) + pos_n_min
x_goal[:, StateIndex.POSE] = x_goal[:, StateIndex.POSE] * (pos_e_max - pos_e_min) + pos_e_min
x_goal[:, StateIndex.ALT] = x_goal[:, StateIndex.ALT] * (alt_max - 500) + 500
x_goal[:, StateIndex.POW] = x_goal[:, StateIndex.POW] * (pow_max - pow_min) + pow_min
x_goal[:, 13] = x_goal[:, 13] * (nz_int_max - nz_int_min) + nz_int_min
x_goal[:, 14] = x_goal[:, 14] * (ps_int_max - ps_int_min) + ps_int_min
x_goal[:, 15] = x_goal[:, 15] * (nyr_int_max - nyr_int_min) + nyr_int_min

# Safe and unsafe regions are defined in lf_net_f16_gcas loss functions

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 1.0
clf_lambda = 0.0
safe_level = 1.0
timestep = 0.001
n_hidden = 64
learning_rate = 0.001
epochs = 500
batch_size = 64
controller_penalty = 1e-5  # coefficient for loss for matching nominal controller


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
filename = "logs/f16_lf_gcas.pth.tar"
checkpoint = torch.load(filename)
lf_net = LF_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                dynamics, control_affine_dynamics, u_nominal)
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
    # and gradually decrease the penalty for matching the nominal controller
    controller_penalty *= 0.1 ** (epoch // 4)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Compute loss
        loss = 0.0
        loss += controller_loss(x, lf_net, controller_penalty, print_loss=False)
        loss += lyapunov_loss(x,
                              x_goal,
                              lf_net,
                              clf_lambda,
                              safe_level,
                              timestep,
                              print_loss=False)

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
        loss += controller_loss(x_test, lf_net, controller_penalty, print_loss=True)
        loss += lyapunov_loss(x_test,
                              x_goal,
                              lf_net,
                              clf_lambda,
                              safe_level,
                              timestep,
                              print_loss=True)
        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            filename = 'logs/f16_lf_gcas.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'relaxation_penalty': relaxation_penalty,
                        'controller_penalty': controller_penalty,
                        'safe_level': safe_level,
                        'clf_lambda': clf_lambda,
                        'lf_net': lf_net.state_dict()}, filename)
        test_losses.append(loss.item())
