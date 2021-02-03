import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
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
filename = "logs/pvtol_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
scenarios = [
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I},
]
nominal_scenario = scenarios[0]
clf_net = CLF_QP_Net(n_dims,
                     checkpoint['n_hidden'],
                     n_controls,
                     checkpoint['clf_lambda'],
                     10.0,  # checkpoint['relaxation_penalty'],
                     f_func,
                     g_func,
                     u_nominal,
                     scenarios,
                     nominal_scenario,
                     allow_relax=False)
clf_net.load_state_dict(checkpoint['clf_net'])

with torch.no_grad():
    n_grid = 50
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
            residuals[j, i] = r
            V_values[j, i] = V
            V_dot_values[j, i] = V_dot

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(17, 8)
    contours = axs[0].contourf(x, y, V_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0], orientation="horizontal")
    # axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$y$')
    axs[0].set_title('$V$')

    contours = axs[1].contourf(x, y, V_dot_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[1], orientation="horizontal")
    # axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$y$')
    axs[1].set_title('$dV/dt$')

    plt.savefig("logs/plots/pvtol/lyap_contour.png")
    plt.show()
