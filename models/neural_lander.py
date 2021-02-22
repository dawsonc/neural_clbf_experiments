import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from models.utils import lqr

# Neural lander dynamics. Thanks to Dawei for the code!!
n_dims = 6
n_controls = 3

rho = 1.225
gravity = 9.81
drone_height = 0.09
mass = 1.47                  # mass

Sim_duration = 1000


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        if not x.is_cuda:
            self.cpu()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def read_weight(filename):
    model_weight = torch.load(filename, map_location=torch.device('cpu'))
    model = Network().double()
    model.load_state_dict(model_weight)
    model = model.float()
    # .cuda()
    return model


num_dim_x = 6
num_dim_control = 3

Fa_model = read_weight('data/Fa_net_12_3_full_Lip16.pth')


def Fa_func(z, vx, vy, vz):
    if next(Fa_model.parameters()).device != z.device:
        Fa_model.to(z.device)
    bs = z.shape[0]
    # use prediction from NN as ground truth
    state = torch.zeros([bs, 1, 12]).type(z.type())
    state[:, 0, 0] = z + drone_height
    state[:, 0, 1] = vx  # velocity
    state[:, 0, 2] = vy  # velocity
    state[:, 0, 3] = vz  # velocity
    state[:, 0, 7] = 1.0
    state[:, 0, 8:12] = 6508.0/8000

    Fa = Fa_model(state).squeeze(
        1) * torch.tensor([30., 15., 10.]).reshape(1, 3).type(z.type())
    return Fa


def Fa_func_np(x):
    z = torch.tensor(x[2]).float().view(1, -1)
    vx = torch.tensor(x[3]).float().view(1, -1)
    vy = torch.tensor(x[4]).float().view(1, -1)
    vz = torch.tensor(x[5]).float().view(1, -1)
    Fa = Fa_func(z, vx, vy, vz).cpu().detach().numpy()
    return Fa


def f_func(x, mass=mass):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, y, z, vx, vy, vz = [x[:, i, 0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = vx
    f[:, 1, 0] = vy
    f[:, 2, 0] = vz

    Fa = Fa_func(z, vx, vy, vz)
    f[:, 3, 0] = Fa[:, 0] / mass
    f[:, 4, 0] = Fa[:, 1] / mass
    f[:, 5, 0] = Fa[:, 2] / mass - gravity
    return f


def g_func(x, mass=mass):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 3, 0] = 1
    B[:, 4, 1] = 1
    B[:, 5, 2] = 1
    return B


def control_affine_dynamics(x, **kwargs):
    """
    Return the control-affine dynamics evaluated at the given state

    x = [[x, z, theta, vx, vz, theta_dot]_1, ...]
    """
    return f_func(x, **kwargs), g_func(x, **kwargs)


# Define linearized matrices for LQR control (assuming no residual force)
A = np.zeros(n_dims, n_dims)
A[:3, 3:] = np.eye(3)
B = np.zeros(n_dims, n_controls)
B[3:, :] = np.eye(n_controls)
# Define cost matrices as identity
Q = np.eye(n_dims)
R = np.eye(n_controls)
# Get feedback matrix
K_np = lqr(A, B, Q, R)


def u_nominal(x, **kwargs):
    """
    Return the nominal controller for the system at state x
    """
    # Compute nominal control from feedback + equilibrium control
    K = torch.tensor(K_np, dtype=x.dtype)
    u_nominal = -(K @ x.T).T
    u_eq = torch.zeros_like(u_nominal)

    return u_nominal + u_eq
