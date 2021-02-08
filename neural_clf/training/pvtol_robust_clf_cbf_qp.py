import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.clf_cbf_qp_net import (
    CLF_CBF_QP_Net,
    lyapunov_loss,
    barrier_loss,
    controller_loss,
)
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
N_train = 10000
xz = torch.Tensor(N_train, 2).uniform_(-3, 3)
xzdot = torch.Tensor(N_train, 2).uniform_(-3, 3)
theta = torch.Tensor(N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_train, 1).uniform_(-2*np.pi, 2*np.pi)
x_train = torch.cat((xz, theta, xzdot, theta_dot), 1)
# Take some extra samples near the origin to make sure the stabilization is smooth
xz = torch.Tensor(N_train, 2).uniform_(-0.5, 0.5)
xzdot = torch.Tensor(N_train, 2).uniform_(-1, 1)
theta = torch.Tensor(N_train, 1).uniform_(-1, 1)
theta_dot = torch.Tensor(N_train, 1).uniform_(-1, 1)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_train = torch.cat((x_train, x_near_origin), 0)
# Also take some extra samples at the safe/unsafe barrier
x = torch.Tensor(3 * N_train, 1).uniform_(-3, 3)
z = torch.Tensor(3 * N_train, 1).uniform_(-3.0, -1)
xzdot = torch.Tensor(3 * N_train, 2).uniform_(-3, 3)
theta = torch.Tensor(3 * N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(3 * N_train, 1).uniform_(-2*np.pi, 2*np.pi)
x_near_border = torch.cat((x, z, theta, xzdot, theta_dot), 1)
x_train = torch.cat((x_train, x_near_border), 0)

# Also get some testing data, just to be principled
N_test = 1000
xz = torch.Tensor(N_test, 2).uniform_(-3, 3)
xzdot = torch.Tensor(N_test, 2).uniform_(-3, 3)
theta = torch.Tensor(N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_test, 1).uniform_(-2*np.pi, 2*np.pi)
x_test = torch.cat((xz, theta, xzdot, theta_dot), 1)
# Take some extra samples near the origin to make sure the stabilization is smooth
xz = torch.Tensor(N_test, 2).uniform_(-0.5, 0.5)
xzdot = torch.Tensor(N_test, 2).uniform_(-1, 1)
theta = torch.Tensor(N_test, 1).uniform_(-1, 1)
theta_dot = torch.Tensor(N_test, 1).uniform_(-1, 1)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_test = torch.cat((x_test, x_near_origin), 0)
# Also take some extra samples in the unsafe region
x = torch.Tensor(3 * N_test, 1).uniform_(-3, 3)
z = torch.Tensor(3 * N_test, 1).uniform_(-3, -1)
xzdot = torch.Tensor(3 * N_test, 2).uniform_(-3, 3)
theta = torch.Tensor(3 * N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(3 * N_test, 1).uniform_(-2*np.pi, 2*np.pi)
x_near_border = torch.cat((x, z, theta, xzdot, theta_dot), 1)
x_test = torch.cat((x_test, x_near_border), 0)

# Segment the test set into safe and unsafe regions
# (z >= -0.25 is safe, z <= -0.5 is unsafe)
safe_z = -0.1
unsafe_z = -1
safe_mask_test = x_test[:, 1] >= safe_z
unsafe_mask_test = x_test[:, 1] <= unsafe_z

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
n_scenarios = len(scenarios)

# Define hyperparameters and define the learning rate and penalty schedule
clf_relaxation_penalty = 1.0
cbf_relaxation_penalty = 10.0
clf_lambda = 1.0
cbf_lambda = 10.0
timestep = 0.001
n_hidden = 32
learning_rate = 1e-3
epochs = 1000
batch_size = 64


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.5 ** (epoch // 10))
    # print(f"Learning rate: {round(max(lr, 1e-5), 6)}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF/CBF conditions, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(cbf_net, epoch):
    penalty = clf_relaxation_penalty * (2 ** (epoch // 3))
    cbf_net.clf_relaxation_penalty = min(penalty, 1000.0)
    penalty = cbf_relaxation_penalty * (2 ** (epoch // 3))
    cbf_net.cbf_relaxation_penalty = min(penalty, 1000.0)


# Instantiate the network
filename = "logs/pvtol_robust_clf_cbf_qp.pth.tar"
checkpoint = torch.load(filename)
clf_cbf_net = CLF_CBF_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, cbf_lambda,
                             clf_relaxation_penalty, cbf_relaxation_penalty,
                             f_func, g_func, u_nominal, scenarios, nominal_scenario,
                             allow_cbf_relax=False)
# clf_cbf_net.load_state_dict(checkpoint['clf_cbf_net'])

# Initialize the optimizer
optimizer = optim.Adam(clf_cbf_net.parameters(), lr=learning_rate)

# Train!
test_losses = []
for epoch in range(epochs):
    # Randomize presentation order
    permutation = torch.randperm(N_train)

    # Cool learning rate
    adjust_learning_rate(optimizer, epoch)
    adjust_relaxation_penalty(clf_cbf_net, epoch)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Segment into safe/unsafe
        safe_mask = x[:, 1] >= safe_z
        unsafe_mask = x[:, 1] <= unsafe_z

        # Zero parameter gradients before training
        optimizer.zero_grad()

        # Compute loss
        loss = 0.0
        loss += lyapunov_loss(x, x0, clf_cbf_net, clf_lambda, timestep, print_loss=False)
        loss += barrier_loss(x,
                             safe_mask,
                             unsafe_mask,
                             clf_cbf_net,
                             cbf_lambda,
                             timestep,
                             print_loss=False)
        loss += controller_loss(x, clf_cbf_net, print_loss=False)

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
        # Compute loss...
        loss = 0.0
        loss += lyapunov_loss(x_test, x0, clf_cbf_net, clf_lambda, timestep, print_loss=True)
        loss += barrier_loss(x_test,
                             safe_mask_test,
                             unsafe_mask_test,
                             clf_cbf_net,
                             cbf_lambda,
                             timestep,
                             print_loss=True)
        loss += controller_loss(x, clf_cbf_net, print_loss=True)

        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() <= min(test_losses):
            print("saving new model")
            filename = 'logs/pvtol_robust_clf_cbf_qp.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'clf_relaxation_penalty': clf_relaxation_penalty,
                        'cbf_relaxation_penalty': cbf_relaxation_penalty,
                        'clf_lambda': clf_lambda,
                        'cbf_lambda': cbf_lambda,
                        'safe_z': safe_z,
                        'unsafe_z': unsafe_z,
                        'clf_cbf_net': clf_cbf_net.state_dict()}, filename)
        test_losses.append(loss.item())
