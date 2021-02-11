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


torch.set_default_dtype(torch.float64)

# First, sample training data uniformly from the state space
N_train = 1000000
xy = torch.Tensor(N_train, 2).uniform_(-4, 4)
xydot = torch.Tensor(N_train, 2).uniform_(-10, 10)
theta = torch.Tensor(N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_train, 1).uniform_(-2 * np.pi, 2 * np.pi)
x_train = torch.cat((xy, theta, xydot, theta_dot), 1)
# Add some training data just around the origin
xz = torch.Tensor(2 * N_train, 2).uniform_(-1, 1)
xzdot = torch.Tensor(2 * N_train, 2).uniform_(-10, 10)
theta = torch.Tensor(2 * N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(2 * N_train, 1).uniform_(-2 * np.pi, 2 * np.pi)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_train = torch.cat((x_train, x_near_origin), 0)
# And some more, to make sure the stabilization is good
xz = torch.Tensor(N_train, 2).uniform_(-1, 1)
xzdot = torch.Tensor(N_train, 2).uniform_(-1, 1)
theta = torch.Tensor(N_train, 1).uniform_(-0.2 * np.pi, 0.2 * np.pi)
theta_dot = torch.Tensor(N_train, 1).uniform_(-0.5 * np.pi, 0.5 * np.pi)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_train = torch.cat((x_train, x_near_origin), 0)

# Also get some testing data, just to be principled
N_test = 10000
xy = torch.Tensor(N_test, 2).uniform_(-4, 4)
xydot = torch.Tensor(N_test, 2).uniform_(-10, 10)
theta = torch.Tensor(N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_test, 1).uniform_(-2*np.pi, 2*np.pi)
x_test = torch.cat((xy, theta, xydot, theta_dot), 1)
# Also add some test data just around the origin
xz = torch.Tensor(2 * N_test, 2).uniform_(-1, 1)
xzdot = torch.Tensor(2 * N_test, 2).uniform_(-10, 10)
theta = torch.Tensor(2 * N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(2 * N_test, 1).uniform_(-2 * np.pi, 2 * np.pi)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_test = torch.cat((x_test, x_near_origin), 0)
# And some more, to make sure the stabilization is good
xz = torch.Tensor(N_test, 2).uniform_(-1, 1)
xzdot = torch.Tensor(N_test, 2).uniform_(-1, 1)
theta = torch.Tensor(N_test, 1).uniform_(-0.2 * np.pi, 0.2 * np.pi)
theta_dot = torch.Tensor(N_test, 1).uniform_(-0.5 * np.pi, 0.5 * np.pi)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_test = torch.cat((x_test, x_near_origin), 0)

# Create a tensor for the origin as well, which is our goal
x0 = torch.zeros(1, 6)

# Also define the safe and unsafe regions
safe_z = -0.1
unsafe_z = -0.5
safe_xz_radius = 3
unsafe_xz_radius = 3.5
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
n_hidden = 48
learning_rate = 0.001
epochs = 1000
batch_size = 64


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 1))
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
