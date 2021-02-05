import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clf.controllers.clf_cbf_qp_net import CLF_CBF_QP_Net
from models.pvtol import (
    f_func,
    g_func,
    u_nominal,
    n_controls,
    n_dims,
    low_m,
    high_m,
    low_I,
    high_I,
)


# Beautify plots
sns.set_theme(context="talk", style="white")

# Load the model from file
filename = "logs/pvtol_robust_clf_cbf_qp.pth.tar"
checkpoint = torch.load(filename)
scenarios = [
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I},
]
nominal_scenario = scenarios[0]
clf_cbf_net = CLF_CBF_QP_Net(n_dims,
                             checkpoint['n_hidden'],
                             n_controls,
                             checkpoint['clf_lambda'],
                             checkpoint['cbf_lambda'],
                             checkpoint['clf_relaxation_penalty'],
                             checkpoint['cbf_relaxation_penalty'],
                             f_func,
                             g_func,
                             u_nominal,
                             scenarios,
                             nominal_scenario)
clf_cbf_net.load_state_dict(checkpoint['clf_cbf_net'])

with torch.no_grad():
    n_grid = 20
    x = torch.linspace(-5, 5, n_grid)
    z = torch.linspace(-2, 5, n_grid)
    grid_x, grid_z = torch.meshgrid(x, z)
    residuals = torch.zeros(n_grid, n_grid)
    V_values = torch.zeros(n_grid, n_grid)
    V_dot_values = torch.zeros(n_grid, n_grid)
    H_values = torch.zeros(n_grid, n_grid)
    H_dot_values = torch.zeros(n_grid, n_grid)
    print("Plotting V and H on grid...")
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            # Get the residual from running the model
            q = torch.zeros(1, n_dims)
            q[0, 0] = x[i]
            q[0, 1] = z[j]
            _, r, V, V_dot, H, H_dot = clf_cbf_net(q)
            # Remember: rows are z and columns are x
            residuals[j, i] = r
            V_values[j, i] = V
            V_dot_values[j, i] = V_dot
            H_values[j, i] = H
            H_dot_values[j, i] = H_dot

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(17, 17)
    contours = axs[0, 0].contourf(x, z, V_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0, 0], orientation="horizontal")
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel('$z$')
    axs[0, 0].set_title('$V$')

    contours = axs[0, 1].contourf(x, z, V_dot_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0, 1], orientation="horizontal")
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel('$z$')
    axs[0, 1].set_title('$dV/dt$')

    contours = axs[1, 0].contourf(x, z, H_values, cmap="magma", levels=[-1, -0.01, 0.0, 0.01, 1])
    axs[1, 0].plot([x.min(), x.max()], [checkpoint["safe_z"], checkpoint["safe_z"]],
                   color="g", label="Safe")
    axs[1, 0].plot([x.min(), x.max()], [checkpoint["unsafe_z"], checkpoint["unsafe_z"]],
                   color="r", label="Unsafe")
    plt.colorbar(contours, ax=axs[1, 0], orientation="horizontal")
    axs[1, 0].set_xlabel('$x$')
    axs[1, 0].set_ylabel('$z$')
    axs[1, 0].set_title('$H$')

    contours = axs[1, 1].contourf(x, z, H_dot_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[1, 1], orientation="horizontal")
    axs[1, 1].set_xlabel('$x$')
    axs[1, 1].set_ylabel('$z$')
    axs[1, 1].set_title('$dH/dt$')

    # plt.savefig("logs/plots/pvtol/clf_cbf_contour.png")
    plt.show()
