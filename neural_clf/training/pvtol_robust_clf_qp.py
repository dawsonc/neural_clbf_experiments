import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.clf_qp_net import (
    CLF_QP_Net,
    lyapunov_loss,
    controller_loss,
)
from models.pvtol import (
    control_affine_dynamics,
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
from neural_clf.training.simulation import simulate_rollout


torch.set_default_dtype(torch.float64)

# Define the region of interest
x_min, x_max = (-4.0, 4.0)
y_min, y_max = (-4.0, 4.0)
theta_min, theta_max = (-np.pi, np.pi)
xdot_min, xdot_max = (-8.0, 8.0)
ydot_min, ydot_max = (-8.0, 8.0)
thetadot_min, thetadot_max = (-2 * np.pi, 2 * np.pi)

# First, sample some training data uniformly from the state space
# We'll augment this later with rollout episodes
N_train = 100000
x_train = torch.Tensor(N_train, n_dims).uniform_(0.0, 1.0)
x_train[:, 0] = x_train[:, 0] * (x_max - x_min) + x_min
x_train[:, 1] = x_train[:, 1] * (y_max - y_min) + y_min
x_train[:, 2] = x_train[:, 2] * (theta_max - theta_min) + theta_min
x_train[:, 3] = x_train[:, 3] * (xdot_max - xdot_min) + xdot_min
x_train[:, 4] = x_train[:, 4] * (ydot_max - ydot_min) + ydot_min
x_train[:, 5] = x_train[:, 5] * (thetadot_max - thetadot_min) + thetadot_min

# Also get some testing data, just to be principled
N_test = 100000
x_test = torch.Tensor(N_test, n_dims).uniform_(0.0, 1.0)
x_test[:, 0] = x_test[:, 0] * (x_max - x_min) + x_min
x_test[:, 1] = x_test[:, 1] * (y_max - y_min) + y_min
x_test[:, 2] = x_test[:, 2] * (theta_max - theta_min) + theta_min
x_test[:, 3] = x_test[:, 3] * (xdot_max - xdot_min) + xdot_min
x_test[:, 4] = x_test[:, 4] * (ydot_max - ydot_min) + ydot_min
x_test[:, 5] = x_test[:, 5] * (thetadot_max - thetadot_min) + thetadot_min

# Create a tensor for the origin as well, which is our goal
x0 = torch.zeros(1, 6)

# Also define the safe and unsafe regions
safe_z = -0.1
unsafe_z = -0.5
safe_xz_radius = 3
unsafe_xz_radius = 3.5
# Compute mask of safe and unsafe test data
safe_mask_test = torch.logical_and(x_test[:, 1] >= safe_z,
                                   x_test[:, :2].norm(dim=-1) <= safe_xz_radius)
unsafe_mask_test = torch.logical_or(x_test[:, 1] <= unsafe_z,
                                    x_test[:, :2].norm(dim=-1) >= unsafe_xz_radius)

# Define the scenarios
nominal_scenario = {"m": low_m, "inertia": low_I}
scenarios = [
    {"m": low_m, "inertia": low_I},
    # {"m": low_m, "inertia": high_I},
    # {"m": high_m, "inertia": low_I},
    # {"m": high_m, "inertia": high_I},
]

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 1.0
clf_lambda = 0.1
safe_level = 1.0
timestep = 0.001
n_hidden = 64
learning_rate = 0.001
epochs = 1000
batch_size = 64
N_rollouts = 100
rollout_length = 1.0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF condition, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(clf_net, epoch):
    penalty = relaxation_penalty * (2 ** (epoch // 2))
    clf_net.relaxation_penalty = penalty


# Instantiate the network
filename = "logs/pvtol_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
clf_net = CLF_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                     control_affine_dynamics, u_nominal, scenarios, nominal_scenario)
# clf_net.load_state_dict(checkpoint['clf_net'])

# Initialize the optimizer
optimizer = optim.Adam(clf_net.parameters(), lr=learning_rate)

# Train!
test_losses = []
for epoch in range(epochs):
    # Randomize presentation order
    N_train = x_train.shape[0]
    permutation = torch.randperm(N_train)

    # Cool learning rate
    adjust_learning_rate(optimizer, epoch)
    # And follow the relaxation penalty schedule
    adjust_relaxation_penalty(clf_net, epoch)

    loss_acumulated = 0.0
    print("Training...")
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Segment into safe/unsafe
        safe_mask = torch.logical_and(x[:, 1] >= safe_z,
                                      x[:, :2].norm(dim=-1) <= safe_xz_radius)
        unsafe_mask = torch.logical_or(x[:, 1] <= unsafe_z,
                                       x[:, :2].norm(dim=-1) >= unsafe_xz_radius)

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Compute loss
        loss = 0.0
        loss += lyapunov_loss(x,
                              x0,
                              safe_mask,
                              unsafe_mask,
                              clf_net,
                              clf_lambda,
                              safe_level,
                              timestep,
                              print_loss=False)
        loss += controller_loss(x, clf_net, print_loss=False)

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
                              safe_mask_test,
                              unsafe_mask_test,
                              clf_net,
                              clf_lambda,
                              safe_level,
                              timestep,
                              print_loss=True)
        loss += controller_loss(x_test, clf_net, print_loss=True)
        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            filename = 'logs/pvtol_robust_clf_qp.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'relaxation_penalty': relaxation_penalty,
                        'G': G,
                        'h': h,
                        'safe_z': safe_z,
                        'unsafe_z': unsafe_z,
                        'safe_xz_radius': safe_xz_radius,
                        'unsafe_xz_radius': unsafe_xz_radius,
                        'safe_level': safe_level,
                        'clf_lambda': clf_lambda,
                        'clf_net': clf_net.state_dict()}, filename)
        test_losses.append(loss.item())

    # Conduct rollouts to augment the training and test data
    with torch.no_grad():
        rollout_init_x = torch.Tensor(N_rollouts, n_dims).uniform_(0.0, 1.0)
        rollout_init_x[:, 0] = rollout_init_x[:, 0] * (x_max - x_min) + x_min
        rollout_init_x[:, 1] = rollout_init_x[:, 1] * (y_max - y_min) + y_min
        rollout_init_x[:, 2] = rollout_init_x[:, 2] * (theta_max - theta_min) + theta_min
        rollout_init_x[:, 3] = rollout_init_x[:, 3] * (xdot_max - xdot_min) + xdot_min
        rollout_init_x[:, 4] = rollout_init_x[:, 4] * (ydot_max - ydot_min) + ydot_min
        rollout_init_x[:, 5] = rollout_init_x[:, 5] * (thetadot_max - thetadot_min) + thetadot_min

        rolled_out_x = simulate_rollout(rollout_init_x,
                                        clf_net,
                                        control_affine_dynamics,
                                        timestep,
                                        rollout_length,
                                        nominal_scenario)
        # Unwrap to give a bunch of samples of the state space
        rolled_out_x = rolled_out_x.view(1, -1, n_dims).squeeze()

        # Add the rollouts to the training data
        x_train = torch.vstack((x_train, rolled_out_x))
