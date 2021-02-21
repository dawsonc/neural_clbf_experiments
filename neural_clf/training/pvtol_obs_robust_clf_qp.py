import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.clf_uK_qp_net import (
    CLF_K_QP_Net,
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
N_train = 10000000
xy = torch.Tensor(N_train, 2).uniform_(-4, 4)
xydot = torch.Tensor(N_train, 2).uniform_(-10, 10)
theta = torch.Tensor(N_train, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_train, 1).uniform_(-2 * np.pi, 2 * np.pi)
x_train = torch.cat((xy, theta, xydot, theta_dot), 1)
# Add some training data just around the origin
xz = torch.Tensor(2 * N_train, 2).uniform_(-1.0, 1.0)
xzdot = torch.Tensor(2 * N_train, 2).uniform_(-1.0, 1.0)
theta = torch.Tensor(2 * N_train, 1).uniform_(-0.3 * np.pi, 0.3 * np.pi)
theta_dot = torch.Tensor(2 * N_train, 1).uniform_(-1 * np.pi, 1 * np.pi)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_train = torch.cat((x_train, x_near_origin), 0)
N_train = x_train.shape[0]

# Also get some testing data, just to be principled
N_test = 50000
xy = torch.Tensor(N_test, 2).uniform_(-4, 4)
xydot = torch.Tensor(N_test, 2).uniform_(-10, 10)
theta = torch.Tensor(N_test, 1).uniform_(-np.pi, np.pi)
theta_dot = torch.Tensor(N_test, 1).uniform_(-2*np.pi, 2*np.pi)
x_test = torch.cat((xy, theta, xydot, theta_dot), 1)
# Also add some test data just around the origin
xz = torch.Tensor(2 * N_test, 2).uniform_(-1.0, 1.0)
xzdot = torch.Tensor(2 * N_test, 2).uniform_(-1.0, 1.0)
theta = torch.Tensor(2 * N_test, 1).uniform_(-0.3 * np.pi, 0.3 * np.pi)
theta_dot = torch.Tensor(2 * N_test, 1).uniform_(-1 * np.pi, 1 * np.pi)
x_near_origin = torch.cat((xz, theta, xzdot, theta_dot), 1)
x_test = torch.cat((x_test, x_near_origin), 0)
N_test = x_test.shape[0]

# Create a tensor for the origin as well, which is our goal
x0 = torch.zeros(1, 6)
u_eq = u_nominal(x0)


# Also define the safe and unsafe regions
def safe_mask_fn(x):
    """Return the mask of x indicating safe regions"""
    safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

    # We have a floor at z=-0.1 that we need to avoid
    safe_z = -0.1
    floor_mask = x[:, 1] >= safe_z
    safe_mask.logical_and_(floor_mask)

    # We also have a block obstacle to the left at ground level
    obs1_min_x, obs1_max_x = (-1.1, -0.4)
    obs1_min_z, obs1_max_z = (-0.5, 0.6)
    obs1_mask_x = torch.logical_or(x[:, 0] <= obs1_min_x, x[:, 0] >= obs1_max_x)
    obs1_mask_z = torch.logical_or(x[:, 1] <= obs1_min_z, x[:, 1] >= obs1_max_z)
    obs1_mask = torch.logical_or(obs1_mask_x, obs1_mask_z)
    safe_mask.logical_and_(obs1_mask)

    # We also have a block obstacle to the right in the air
    obs2_min_x, obs2_max_x = (-0.1, 1.1)
    obs2_min_z, obs2_max_z = (0.7, 1.5)
    obs2_mask_x = torch.logical_or(x[:, 0] <= obs2_min_x, x[:, 0] >= obs2_max_x)
    obs2_mask_z = torch.logical_or(x[:, 1] <= obs2_min_z, x[:, 1] >= obs2_max_z)
    obs2_mask = torch.logical_or(obs2_mask_x, obs2_mask_z)
    safe_mask.logical_and_(obs2_mask)

    # Also constrain to be within a norm bound
    norm_mask = x.norm(dim=-1) <= 4.5
    safe_mask.logical_and_(norm_mask)

    return safe_mask


# Also define the safe and unsafe regions
def unsafe_mask_fn(x):
    """Return the mask of x indicating safe regions"""
    unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

    # We have a floor at z=-0.1 that we need to avoid
    unsafe_z = -0.3
    floor_mask = x[:, 1] <= unsafe_z
    unsafe_mask.logical_or_(floor_mask)

    # We also have a block obstacle to the left at ground level
    obs1_min_x, obs1_max_x = (-1.0, -0.5)
    obs1_min_z, obs1_max_z = (-0.4, 0.5)
    obs1_mask_x = torch.logical_and(x[:, 0] >= obs1_min_x, x[:, 0] <= obs1_max_x)
    obs1_mask_z = torch.logical_and(x[:, 1] >= obs1_min_z, x[:, 1] <= obs1_max_z)
    obs1_mask = torch.logical_and(obs1_mask_x, obs1_mask_z)
    unsafe_mask.logical_or_(obs1_mask)

    # We also have a block obstacle to the right in the air
    obs2_min_x, obs2_max_x = (0.0, 1.0)
    obs2_min_z, obs2_max_z = (0.8, 1.4)
    obs2_mask_x = torch.logical_and(x[:, 0] >= obs2_min_x, x[:, 0] <= obs2_max_x)
    obs2_mask_z = torch.logical_and(x[:, 1] >= obs2_min_z, x[:, 1] <= obs2_max_z)
    obs2_mask = torch.logical_and(obs2_mask_x, obs2_mask_z)
    unsafe_mask.logical_or_(obs2_mask)

    # Also constrain with a norm bound
    norm_mask = x.norm(dim=-1) >= 5.0
    unsafe_mask.logical_or_(norm_mask)

    return unsafe_mask


safe_mask_train = safe_mask_fn(x_train)
unsafe_mask_train = unsafe_mask_fn(x_train)
safe_mask_test = safe_mask_fn(x_test)
unsafe_mask_test = unsafe_mask_fn(x_test)

# Define the scenarios
nominal_scenario = {"m": low_m, "inertia": low_I}
scenarios = [
    {"m": low_m, "inertia": low_I},
    # {"m": low_m, "inertia": high_I},
    # {"m": high_m, "inertia": low_I},
    # {"m": high_m, "inertia": high_I},
]

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 10.0
clf_lambda = 0.1
safe_level = 1.0
timestep = 0.01
n_hidden = 48
learning_rate = 0.0001
epochs = 1000
batch_size = 64


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.9 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF condition, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(clf_net, epoch):
    penalty = relaxation_penalty * (2 ** (epoch // 2))
    clf_net.relaxation_penalty = penalty


# Instantiate the network
filename = "logs/pvtol_obs_clf.pth.tar"
checkpoint = torch.load(filename)
clf_net = CLF_K_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                       control_affine_dynamics, u_nominal, scenarios, nominal_scenario,
                       x0, u_eq)
clf_net.load_state_dict(checkpoint['clf_net'])
clf_net.use_QP = False

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
        safe_mask = safe_mask_train[indices]
        unsafe_mask = unsafe_mask_train[indices]

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
        loss += controller_loss(x, clf_net,
                                print_loss=False, use_nominal=True, loss_coeff=1e0)

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
        test_batch_size = 20 * batch_size
        for i in range(0, N_test, test_batch_size):
            loss += lyapunov_loss(x_test[i:i+test_batch_size],
                                  x0,
                                  safe_mask_test[i:i+test_batch_size],
                                  unsafe_mask_test[i:i+test_batch_size],
                                  clf_net,
                                  clf_lambda,
                                  safe_level,
                                  timestep,
                                  print_loss=(i == 0))
            # loss += controller_loss(x_test[i:i+test_batch_size], clf_net,
            #                         print_loss=(i == 0), use_nominal=True, loss_coeff=1e0)

        print(f"Epoch {epoch + 1}     test loss: {loss.item() / (N_test / test_batch_size)}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            filename = 'logs/pvtol_obs_clf.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'relaxation_penalty': clf_net.relaxation_penalty,
                        'G': G,
                        'h': h,
                        'safe_level': safe_level,
                        'clf_lambda': clf_lambda,
                        'x_goal': x0,
                        'u_eq': u_eq,
                        'clf_net': clf_net.state_dict()}, filename)
        test_losses.append(loss.item())
