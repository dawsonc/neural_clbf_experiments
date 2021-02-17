import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import gc


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
    (-4, 4),                  # x
    (-4, 4),                  # y
    (-4, 4),                  # z
    (-8, 8),              # vx
    (-8, 8),              # vy
    (-8, 8),              # vz
    (-0.5 * g, 2 * g),        # f
    (-np.pi, np.pi),  # roll
    (-np.pi, np.pi),  # pitch
    (-np.pi, np.pi),  # yaw
]
domain_near_origin = [
    (-0.5, 0.5),              # x
    (-0.5, 0.5),              # y
    (-0.5, 0.5),              # z
    (-0.5, 0.5),              # vx
    (-0.5, 0.5),              # vy
    (-0.5, 0.5),              # vz
    (0.5 * g, 1.5 * g),       # f
    (-np.pi / 3, np.pi / 3),  # roll
    (-np.pi / 3, np.pi / 3),  # pitch
    (-np.pi / 3, np.pi / 3),  # yaw
]

# First, sample training data uniformly from the state space
N_train = 10000000
x_train = torch.Tensor(N_train, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain[i]
    x_train[:, i] = x_train[:, i] * (max_val - min_val) + min_val
x_train_near_origin = torch.Tensor(N_train, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain_near_origin[i]
    x_train_near_origin[:, i] = x_train_near_origin[:, i] * (max_val - min_val) + min_val
x_train = torch.vstack((x_train, x_train_near_origin))

# Also get some testing data
N_test = 50000
x_test = torch.Tensor(N_test, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain[i]
    x_test[:, i] = x_test[:, i] * (max_val - min_val) + min_val
x_test_near_origin = torch.Tensor(N_train, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain_near_origin[i]
    x_test_near_origin[:, i] = x_test_near_origin[:, i] * (max_val - min_val) + min_val
x_test = torch.vstack((x_test, x_test_near_origin))

# Sample some goal states as well
N_goal = 1
goal_domain = [
    (0.0, 0.0),             # x
    (0.0, 0.0),             # y
    (0.0, 0.0),             # z
    (0.0, 0.0),              # vx
    (0.0, 0.0),              # vy
    (0.0, 0.0),              # vz
    (g, g),                  # f
    (0.0, 0.0),  # roll
    (0.0, 0.0),  # pitch
    (0.0, 0.0),  # yaw
]
x0 = torch.Tensor(N_goal, n_dims).uniform_(0.0, 1.0)
for i in range(n_dims):
    min_val, max_val = domain[i]
    x0[:, i] = x0[:, i] * (max_val - min_val) + min_val

# Also define the safe and unsafe regions
# Remember that z is positive pointing downwards
safe_z = 0.2
unsafe_z = 0.6
safe_xyz_radius = 1
unsafe_xyz_radius = 2
safe_mask_test = torch.logical_and(x_test[:, StateIndex.PZ] <= safe_z,
                                   x_test[:, :StateIndex.PZ + 1].norm(dim=-1) <= safe_xyz_radius)
unsafe_mask_test = torch.logical_or(x_test[:, StateIndex.PZ] >= unsafe_z,
                                    x_test[:, :StateIndex.PZ + 1].norm(dim=-1) >= unsafe_xyz_radius)

# Define the scenarios
nominal_scenario = {}
scenarios = [
    {},
]

# Define hyperparameters and define the learning rate and penalty schedule
relaxation_penalty = 10.0
clf_lambda = 1.0
safe_level = 1.0
timestep = 0.01
n_hidden = 48
learning_rate = 0.001
epochs = 1000
batch_size = 64
init_controller_loss_coeff = 1e-4


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr, 1e-5)


# We start by allowing the QP to relax the CLF condition, but we'll gradually increase the
# cost of doing so.
def adjust_relaxation_penalty(clf_net, epoch):
    penalty = relaxation_penalty * (2 ** (epoch // 2))
    clf_net.relaxation_penalty = penalty


# We penalize deviation from the nominal controller more heavily to start, then gradually relax
def adjust_controller_penalty(epoch):
    penalty = init_controller_loss_coeff * (0.8 ** (epoch // 1))
    return max(penalty, 1e-8)


# Instantiate the network
filename = "logs/quad9d_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
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
    # And reduce the reliance on the nominal controller loss
    controller_loss_coeff = adjust_controller_penalty(epoch)

    loss_acumulated = 0.0
    for i in trange(0, N_train, batch_size):
        # Get state from training data
        indices = permutation[i:i+batch_size]
        x = x_train[indices]

        # Segment into safe/unsafe
        safe_mask = torch.logical_and(x[:, StateIndex.PZ] <= safe_z,
                                      x[:, :StateIndex.PZ + 1].norm(dim=-1) <= safe_xyz_radius)
        unsafe_mask = torch.logical_or(x[:, StateIndex.PZ] >= unsafe_z,
                                       x[:, :StateIndex.PZ + 1].norm(dim=-1) >= unsafe_xyz_radius)

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
        loss += controller_loss(x, clf_net, print_loss=False, use_nominal=True,
                                loss_coeff=controller_loss_coeff)

        # Accumulate loss from this epoch and do backprop
        loss.backward()
        loss_acumulated += loss.detach()

        # Update the parameters
        optimizer.step()

    # Print progress on each epoch, then re-zero accumulated loss for the next epoch
    print(f'Epoch {epoch + 1} training loss: {loss_acumulated / (N_train / batch_size)}')
    loss_acumulated = 0.0

    # Make sure no unused memory is hanging around
    gc.collect()

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
        loss += controller_loss(x_test, clf_net, print_loss=True, use_nominal=True,
                                loss_coeff=controller_loss_coeff)
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
