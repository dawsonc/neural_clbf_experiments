import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm

from neural_clf.train_clf_net_pvtol import CLF_QP_Net
from models.pvtol import (
    f_func,
    g_func,
    n_controls,
    n_dims,
    low_m,
    high_m,
    low_I,
    high_I,
)


# Load the model from file
filename = "logs/pvtol_model_best.pth.tar"
checkpoint = torch.load(filename)
clf_net = CLF_QP_Net(n_dims,
                     checkpoint['n_hidden'],
                     n_controls,
                     checkpoint['clf_lambda'],
                     checkpoint['relaxation_penalty'],
                     allow_relax=False)
clf_net.load_state_dict(checkpoint['clf_net'])

with torch.no_grad():
    n_grid = 30
    x = torch.linspace(-1, 1, n_grid)
    y = torch.linspace(-1, 1, n_grid)
    grid_x, grid_y = torch.meshgrid(x, y)
    residuals = torch.zeros(n_grid, n_grid)
    V_values = torch.zeros(n_grid, n_grid)
    V_dot_values = torch.zeros(n_grid, n_grid)
    print("Plotting V on grid...")
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            # Get the residual from running the model
            q = torch.zeros(1, n_dims)
            q[0, 0] = x[i]
            q[0, 1] = y[j]
            _, r, V, V_dot = clf_net(q)
            residuals[i, j] = r
            V_values[i, j] = V
            V_dot_values[i, j] = V_dot

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(grid_x, grid_y, residuals.numpy(),
    #                 rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    # ax.set_zlabel('Residual')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_x, grid_y, V_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$V$')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_x, grid_y, V_dot_values.numpy(),
                    rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$\\dot{V}$')

    plt.show()

# Simulate some results
with torch.no_grad():
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = 0.2
    x_sim_start[:, 1] = 0.2

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(low_m, high_m)
    inertias = torch.Tensor(N_sim, 1).uniform_(low_I, high_I)

    t_sim = 5
    delta_t = 0.01
    num_timesteps = int(t_sim // delta_t)
    x_sim = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim = torch.zeros(num_timesteps, N_sim, 1)
    r_sim = torch.zeros(num_timesteps, N_sim, 1)
    x_sim[0, :, :] = x_sim_start

    print("Simulating controller...")
    V_last = np.inf
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_sim[tstep - 1, :, :]
        # Get the control input at the current state
        u, r, V, Vdot = clf_net(x_current)
        # Get the dynamics
        for i in range(N_sim):
            f_val = f_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            g_val = g_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            # Take one step to the future
            xdot = f_val + g_val @ u[i, :]
            x_sim[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            # if Vdot > 0:
            #     import pdb; pdb.set_trace()
            # V_last = V[i]

        u_sim[tstep, :, :] = u
        V_sim[tstep, :, 0] = V
        Vdot_sim[tstep, :, 0] = Vdot
        r_sim[tstep, :, 0] = r

    t = np.linspace(0, t_sim, num_timesteps)
    ax = plt.subplot(4, 1, 1)
    ax.plot(t, x_sim[:, 0, :3].norm(dim=-1))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$||q||$")
    ax = plt.subplot(4, 1, 2)
    ax.plot(t[1:], V_sim[1:, :, 0], 'o')
    ax.set_xlabel("$t$")
    ax.set_ylabel("$V$")
    ax = plt.subplot(4, 1, 3)
    ax.plot(t[1:], Vdot_sim[1:, :, 0], 'o')
    ax.set_xlabel("$t$")
    ax.set_ylabel("$Vdot$")
    ax = plt.subplot(4, 1, 4)
    ax.plot(t[1:], u_sim[1:, 0, :])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u$")
    plt.show()
