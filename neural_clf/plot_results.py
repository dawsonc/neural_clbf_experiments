import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm

from neural_clf.train_clf_net_inverted_pendulum import CLF_QP_Net
from models.inverted_pendulum import (
    f_func,
    g_func,
    n_controls,
    n_dims
)


# Load the model from file
filename = "logs/pendulum_model_best.pth.tar"
checkpoint = torch.load(filename)
clf_net = CLF_QP_Net(n_dims,
                     checkpoint['n_hidden'],
                     n_controls,
                     checkpoint['clf_lambda'],
                     checkpoint['relaxation_penalty'],
                     checkpoint['G'],
                     checkpoint['h'])
clf_net.load_state_dict(checkpoint['clf_net'])

# Make a grid and plot the relaxation
with torch.no_grad():
    n_theta = 30
    n_theta_dot = 30
    theta = torch.linspace(-np.pi, np.pi, n_theta)
    theta_dot = torch.linspace(-np.pi, np.pi, n_theta_dot)
    grid_theta, grid_theta_dot = torch.meshgrid(theta, theta_dot)
    residuals = torch.zeros(n_theta, n_theta_dot)
    V_values = torch.zeros(n_theta, n_theta_dot)
    V_dot_values = torch.zeros(n_theta, n_theta_dot)
    print("Plotting V on grid...")
    for i in tqdm(range(n_theta)):
        for j in range(n_theta_dot):
            # Get the residual from running the model
            x = torch.tensor([[theta[i], theta_dot[j]]])
            _, r, V, V_dot = clf_net(x)
            residuals[i, j] = r
            V_values[i, j] = V
            V_dot_values[i, j] = V_dot
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, residuals.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('Residual')
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, V_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$V$')
    #
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_theta, grid_theta_dot, V_dot_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$\\theta$')
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_zlabel('$\\dot{V}$')
    #
    plt.show()

# Simulate some results
with torch.no_grad():
    N_sim = 1
    theta = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    theta_dot = torch.Tensor(N_sim, 1).uniform_(-0.5*np.pi, 0.5*np.pi)
    x_sim_start = torch.cat((theta, theta_dot), 1)

    t_sim = 5
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)
    x_sim = torch.zeros(num_timesteps, N_sim, 2)
    u_sim = torch.zeros(num_timesteps, N_sim, 1)
    V_sim = torch.zeros(num_timesteps, N_sim, 1)
    Vd_sim = torch.zeros(num_timesteps, N_sim, 1)
    x_sim[0, :, :] = x_sim_start

    print("Simulating controller...")
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state
        u, r, V, Vdot = clf_net(x_current)
        # Get the dynamics
        f_val = f_func(x_current)
        g_val = g_func(x_current)
        # Take one step to the future
        xdot = f_val.unsqueeze(-1) + torch.bmm(g_val, u.unsqueeze(-1))
        x_sim[tstep, :, :] = x_current + delta_t * xdot.squeeze()
        u_sim[tstep, :, 0] = u
        V_sim[tstep, :, 0] = V
        Vd_sim[tstep, :, 0] = Vdot

    N_traces_to_plot = 5
    t = np.linspace(0, t_sim, num_timesteps)
    ax = plt.subplot(5, 1, 1)
    ax.plot(t, x_sim[:, :N_traces_to_plot, 0])
    ax.set_xlabel("timestep")
    ax.set_ylabel("$\\theta$")
    ax = plt.subplot(5, 1, 2)
    ax.plot(t, x_sim[:, :N_traces_to_plot, 1])
    ax.set_xlabel("timestep")
    ax.set_ylabel("$\\dot{\\theta}$")
    ax = plt.subplot(5, 1, 3)
    ax.plot(t, u_sim[:, :N_traces_to_plot, 0])
    ax.set_xlabel("timestep")
    ax.set_ylabel("$u$")
    ax = plt.subplot(5, 1, 4)
    ax.plot(t, V_sim[:, :N_traces_to_plot, 0], 'x')
    ax.set_xlabel("timestep")
    ax.set_ylabel("$V$")
    ax = plt.subplot(5, 1, 5)
    ax.plot(t, Vd_sim[:, :N_traces_to_plot, 0], 'x')
    ax.set_xlabel("timestep")
    ax.set_ylabel("$Vdot$")
    plt.show()
