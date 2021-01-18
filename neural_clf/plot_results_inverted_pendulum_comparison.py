import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

from neural_clf.train_clf_cbf_net_inverted_pendulum import CLF_CBF_QP_Net
from neural_clf.train_lf_bf_net_inverted_pendulum import LF_BF_QP_Net
from models.inverted_pendulum import (
    f_func,
    g_func,
    n_controls,
    n_dims
)


# Set default matplotlib font size
matplotlib.rcParams.update({'font.size': 22})


# Load the models from file
filename = "logs/pendulum_model_best_clf_cbf.pth.tar"
checkpoint = torch.load(filename)
clf_cbf_net = CLF_CBF_QP_Net(n_dims,
                             checkpoint['n_hidden'],
                             n_controls,
                             checkpoint['clf_lambda'],
                             checkpoint['cbf_lambda'],
                             checkpoint['clf_relaxation_penalty'],
                             checkpoint['cbf_relaxation_penalty'],
                             checkpoint['G'],
                             checkpoint['h'])
clf_cbf_net.load_state_dict(checkpoint['clf_cbf_net'])

filename = "logs/pendulum_model_best_lf_bf.pth.tar"
checkpoint = torch.load(filename)
lf_bf_net = LF_BF_QP_Net(n_dims,
                         checkpoint['n_hidden'],
                         n_controls,
                         checkpoint['lf_lambda'],
                         checkpoint['bf_lambda'])
lf_bf_net.load_state_dict(checkpoint['lf_bf_net'])

# Simulate some results
with torch.no_grad():
    N_sim = 1
    theta = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    theta_dot = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    x_sim_start = torch.cat((theta, theta_dot), 1)

    t_sim = 10
    delta_t = 0.01
    num_timesteps = int(t_sim // delta_t)
    x_sim_clf = torch.zeros(num_timesteps, N_sim, 2)
    u_sim_clf = torch.zeros(num_timesteps, N_sim, 1)
    V_sim_clf = torch.zeros(num_timesteps, N_sim, 1)
    H_sim_clf = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_clf[0, :, :] = x_sim_start

    x_sim_lf = torch.zeros(num_timesteps, N_sim, 2)
    u_sim_lf = torch.zeros(num_timesteps, N_sim, 1)
    V_sim_lf = torch.zeros(num_timesteps, N_sim, 1)
    H_sim_lf = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_lf[0, :, :] = x_sim_start

    print("Simulating controller...")
    for tstep in tqdm(range(1, num_timesteps)):
        # Simulate for the control lyapunov version
        # Get the current state
        x_current = x_sim_clf[tstep - 1, :, :]
        # Get the control input at the current state
        u, r_V, r_H, V, Vdot, H, Hdot = clf_cbf_net(x_current)
        # Get the dynamics
        f_val = f_func(x_current)
        g_val = g_func(x_current)
        # Take one step to the future
        xdot = f_val.unsqueeze(-1) + torch.bmm(g_val, u.unsqueeze(-1))
        x_sim_clf[tstep, :, :] = x_current + delta_t * xdot.squeeze()
        u_sim_clf[tstep, :, :] = u
        V_sim_clf[tstep, :, 0] = V
        H_sim_clf[tstep, :, :] = H

        # Simulate for the closed-loop lyapunov version
        # Get the current state
        x_current = x_sim_lf[tstep - 1, :, :]
        # Get the control input at the current state
        u, V, Vdot, H, Hdot = lf_bf_net(x_current)
        # Get the dynamics
        f_val = f_func(x_current)
        g_val = g_func(x_current)
        # Take one step to the future
        xdot = f_val.unsqueeze(-1) + torch.bmm(g_val, u.unsqueeze(-1))
        x_sim_lf[tstep, :, :] = x_current + delta_t * xdot.squeeze()
        u_sim_lf[tstep, :, :] = u
        V_sim_lf[tstep, :, 0] = V
        H_sim_lf[tstep, :, :] = H

    N_traces_to_plot = 5
    t = np.linspace(0, t_sim, num_timesteps)
    ax = plt.subplot(4, 1, 1)
    ax.plot(t, x_sim_clf[:, :N_traces_to_plot, 0])
    ax.plot(t, x_sim_lf[:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\theta$")
    ax.legend(["CLF/CBF", "LF/BF"])
    # ax = plt.subplot(4, 1, 2)
    # ax.plot(t, x_sim[:, :N_traces_to_plot, 1])
    # ax.set_xlabel("$t$")
    # ax.set_ylabel("$\\dot{\\theta}$")
    ax = plt.subplot(4, 1, 2)
    ax.plot(t, u_sim_clf[:, :N_traces_to_plot, 0])
    ax.plot(t, u_sim_lf[:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u$")
    ax.legend(["CLF/CBF", "LF/BF"])
    ax = plt.subplot(4, 1, 3)
    ax.plot(t[1:], V_sim_clf[1:, :N_traces_to_plot, 0])
    ax.plot(t[1:], V_sim_lf[1:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$V$")
    ax.legend(["CLF/CBF", "LF/BF"])
    ax = plt.subplot(4, 1, 4)
    ax.plot(t[1:], H_sim_clf[1:, :N_traces_to_plot, 0])
    ax.plot(t[1:], H_sim_lf[1:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$H$")
    ax.legend(["CLF/CBF", "LF/BF"])
    plt.show()
