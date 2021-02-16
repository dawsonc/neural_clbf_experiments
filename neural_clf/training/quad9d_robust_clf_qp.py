import numpy as np
import torch
import torch.optim as optim
from tqdm import trange


from neural_clf.controllers.clf_qp_net import (
    CLF_QP_Net,
    lyapunov_loss,
    controller_loss,
)
from models.quad9d import (
    control_affine_dynamics,
    u_nominal,
    n_controls,
    n_dims,
    g,
    StateIndex,
)


torch.set_default_dtype(torch.float64)

# Define operational domain through min/max tuples
domain = [
    (-30, 30),                # x
    (-30, 30),                # y
    (-30, 30),                # z
    (-1.5, 1.5),              # vx
    (-1.5, 1.5),              # vy
    (-1.5, 1.5),              # vz
    (-0.5 * g, 2 * g),        # f
    (-np.pi / 3, np.pi / 3),  # roll
    (-np.pi / 3, np.pi / 3),  # pitch
    (-np.pi / 3, np.pi / 3),  # yaw
]

# First, sample training data uniformly from the state space
N_train = 1000000
x_train = torch.Tensor(N_train, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain[i]
    x_train[:, i] = x_train[:, i] * (max_val - min_val) + min_val

# Also get some testing data, just to be principled
N_test = 10000
x_test = torch.Tensor(N_test, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain[i]
    x_test[:, i] = x_test[:, i] * (max_val - min_val) + min_val

# Create a tensor for the origin as well, which is our goal
x0 = torch.zeros(1, n_dims)
x0[0, StateIndex.F] = g

# Also define the safe and unsafe regions
safe_z = -0.1
unsafe_z = -0.5
safe_xyz_radius = 27
unsafe_xyz_radius = 29
safe_mask_test = torch.logical_and(x_test[:, StateIndex.PZ] >= safe_z,
                                   x_test[:, :StateIndex.PZ].norm(dim=-1) <= safe_xyz_radius)
unsafe_mask_test = torch.logical_or(x_test[:, StateIndex.PZ] <= unsafe_z,
                                    x_test[:, :StateIndex.PZ].norm(dim=-1) >= unsafe_xyz_radius)

# Define the scenarios
nominal_scenario = {}
scenarios = [
    {},
]

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 10.0
clf_lambda = 0.1
safe_level = 1.0
timestep = 0.01
n_hidden = 48
learning_rate = 0.001
epochs = 1000
batch_size = 64

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF condition, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(clf_net, epoch):
    penalty = relaxation_penalty * (2 ** (epoch // 2))
    clf_net.relaxation_penalty = penalty


# Instantiate the network
filename = "logs/quad9d_robust_clf_qp.pth.tar"
# checkpoint = torch.load(filename)
clf_net = CLF_QP_Net(n_dims, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                     control_affine_dynamics, u_nominal, scenarios, nominal_scenario)
# clf_net.load_state_dict(checkpoint['clf_net'])
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
        safe_mask = torch.logical_and(x[:, 1] >= safe_z,
                                      x[:, :3].norm(dim=-1) <= safe_xyz_radius)
        unsafe_mask = torch.logical_or(x[:, 1] <= unsafe_z,
                                       x[:, :3].norm(dim=-1) >= unsafe_xyz_radius)

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
        loss += controller_loss(x, clf_net, print_loss=False, use_nominal=False)

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
        loss += controller_loss(x_test, clf_net, print_loss=True, use_nominal=False)
        print(f"Epoch {epoch + 1}     test loss: {loss.item()}")

        # Save the model if it's the best yet
        if not test_losses or loss.item() < min(test_losses):
            print("saving new model")
            filename = 'logs/quad9d_robust_clf_qp.pth.tar'
            torch.save({'n_hidden': n_hidden,
                        'relaxation_penalty': clf_net.relaxation_penalty,
                        'safe_z': safe_z,
                        'unsafe_z': unsafe_z,
                        'safe_xyz_radius': safe_xyz_radius,
                        'unsafe_xyz_radius': unsafe_xyz_radius,
                        'safe_level': safe_level,
                        'clf_lambda': clf_lambda,
                        'clf_net': clf_net.state_dict()}, filename)
        test_losses.append(loss.item())
