import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

from neural_clf.train_clf_cbf_net_inverted_pendulum import CLF_CBF_QP_Net
from models.inverted_pendulum import (
    f_func,
    g_func,
    n_controls,
    n_dims
)


# Set default matplotlib font size
matplotlib.rcParams.update({'font.size': 22})


# Load the model from file
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

# Make a grid and plot the relaxation
with torch.no_grad():
    n_theta = 30
    n_theta_dot = 30
    theta = torch.linspace(-np.pi, np.pi, n_theta)
    theta_dot = torch.linspace(-np.pi, np.pi, n_theta_dot)
    grid_theta, grid_theta_dot = torch.meshgrid(theta, theta_dot)
    residuals_V = torch.zeros(n_theta, n_theta_dot)
    residuals_H = torch.zeros(n_theta, n_theta_dot)
    V_values = torch.zeros(n_theta, n_theta_dot)
    V_dot_values = torch.zeros(n_theta, n_theta_dot)
    H_values = torch.zeros(n_theta, n_theta_dot)
    H_dot_values = torch.zeros(n_theta, n_theta_dot)
    print("Plotting V and H on grid...")
    for i in tqdm(range(n_theta)):
        for j in range(n_theta_dot):
            # Get the residual from running the model
            x = torch.tensor([[theta[i], theta_dot[j]]])
            _, r_V, r_H, V, V_dot, H, H_dot = clf_cbf_net(x)
            residuals_V[i, j] = r_V
            residuals_H[i, j] = r_H
            V_values[i, j] = V
            V_dot_values[i, j] = V_dot
            H_values[i, j] = H
            H_dot_values[i, j] = H_dot

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, residuals_V.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('V Residual')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, residuals_H.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('H Residual')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, V_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$V$')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, V_dot_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$\\dot{V}$')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, H_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$H$')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, H_dot_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$\\dot{H}$')

    plt.show()

# Simulate some results
with torch.no_grad():
    N_sim = 10
    theta = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    theta_dot = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    x_sim_start = torch.cat((theta, theta_dot), 1)

    t_sim = 5
    delta_t = 0.01
    num_timesteps = int(t_sim // delta_t)
    x_sim = torch.zeros(num_timesteps, N_sim, 2)
    u_sim = torch.zeros(num_timesteps, N_sim, 1)
    V_sim = torch.zeros(num_timesteps, N_sim, 1)
    Vd_sim = torch.zeros(num_timesteps, N_sim, 1)
    H_sim = torch.zeros(num_timesteps, N_sim, 1)
    Hd_sim = torch.zeros(num_timesteps, N_sim, 1)
    x_sim[0, :, :] = x_sim_start

    print("Simulating controller...")
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state
        u, r_V, r_H, V, Vdot, H, Hdot = clf_cbf_net(x_current)
        # Get the dynamics
        f_val = f_func(x_current)
        g_val = g_func(x_current)
        # Take one step to the future
        xdot = f_val.unsqueeze(-1) + torch.bmm(g_val, u.unsqueeze(-1))
        x_sim[tstep, :, :] = x_current + delta_t * xdot.squeeze()
        u_sim[tstep, :, :] = u
        V_sim[tstep, :, 0] = V
        Vd_sim[tstep, :, 0] = Vdot
        H_sim[tstep, :, :] = H
        Hd_sim[tstep, :, 0] = Hdot

    N_traces_to_plot = 5
    t = np.linspace(0, t_sim, num_timesteps)
    ax = plt.subplot(4, 1, 1)
    ax.plot(t, x_sim[:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\theta$")
    # ax = plt.subplot(4, 1, 2)
    # ax.plot(t, x_sim[:, :N_traces_to_plot, 1])
    # ax.set_xlabel("$t$")
    # ax.set_ylabel("$\\dot{\\theta}$")
    ax = plt.subplot(4, 1, 2)
    ax.plot(t, u_sim[:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u$")
    ax = plt.subplot(4, 1, 3)
    ax.plot(t[1:], V_sim[1:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$V$")
    ax = plt.subplot(4, 1, 4)
    ax.plot(t[1:], H_sim[1:, :N_traces_to_plot, 0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$H$")
    plt.show()
