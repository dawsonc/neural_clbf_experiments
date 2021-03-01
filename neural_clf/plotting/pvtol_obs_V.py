import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import seaborn as sns

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
from models.pvtol import (
    control_affine_dynamics,
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
obs_color = sns.color_palette("pastel")[3]

# Load the model from file
filename = "logs/pvtol_obs_clf.pth.tar"
checkpoint = torch.load(filename)
scenarios = [
    {"m": low_m, "inertia": low_I},
    # {"m": low_m, "inertia": low_I},
    # {"m": low_m, "inertia": low_I},
    # {"m": low_m, "inertia": low_I},
]
nominal_scenario = scenarios[0]
clf_net = CLF_QP_Net(n_dims,
                     checkpoint['n_hidden'],
                     n_controls,
                     0.1,  # checkpoint['clf_lambda'],
                     1000.0,  # checkpoint['relaxation_penalty'],
                     control_affine_dynamics,
                     u_nominal,
                     scenarios,
                     nominal_scenario)
clf_net.load_state_dict(checkpoint['clf_net'])
# clf_net.use_QP = False


# Also define the safe and unsafe regions
def unsafe_mask_fn(grid_x, grid_z):
    """Return the mask of x indicating safe regions"""
    unsafe_mask = torch.zeros_like(grid_x, dtype=torch.bool)

    # We have a floor at z=-0.1 that we need to avoid
    unsafe_z = -0.3
    floor_mask = grid_z <= unsafe_z
    unsafe_mask.logical_or_(floor_mask)

    # We also have a block obstacle to the left at ground level
    obs1_min_x, obs1_max_x = (-1.0, -0.5)
    obs1_min_z, obs1_max_z = (-0.4, 0.5)
    obs1_mask_x = torch.logical_and(grid_x >= obs1_min_x, grid_x <= obs1_max_x)
    obs1_mask_z = torch.logical_and(grid_z >= obs1_min_z, grid_z <= obs1_max_z)
    obs1_mask = torch.logical_and(obs1_mask_x, obs1_mask_z)
    unsafe_mask.logical_or_(obs1_mask)

    # We also have a block obstacle to the right in the air
    obs2_min_x, obs2_max_x = (0.05, 1.0)
    obs2_min_z, obs2_max_z = (0.8, 1.35)
    obs2_mask_x = torch.logical_and(grid_x >= obs2_min_x, grid_x <= obs2_max_x)
    obs2_mask_z = torch.logical_and(grid_z >= obs2_min_z, grid_z <= obs2_max_z)
    obs2_mask = torch.logical_and(obs2_mask_x, obs2_mask_z)
    unsafe_mask.logical_or_(obs2_mask)

    return unsafe_mask


with torch.no_grad():
    n_grid = 1000
    x = torch.linspace(-2, 1, n_grid)
    z = torch.linspace(-0.5, 1.5, n_grid)
    grid_x, grid_z = torch.meshgrid(x, z)
    residuals = torch.zeros(n_grid, n_grid)
    V_values = torch.zeros(n_grid, n_grid)
    V_dot_values = torch.zeros(n_grid, n_grid)
    print("Plotting V on grid...")
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            # Get the residual from running the model
            q = torch.zeros(1, n_dims)
            # q = torch.tensor([[0.0129, -0.1149, -0.0083,  0.0534, -2.0552,  0.0201]])
            q[0, 0] = x[i]
            q[0, 1] = z[j]
            _, r, V, V_dot = clf_net(q)
            residuals[j, i] = r
            V_values[j, i] = V
            V_dot_values[j, i] = V_dot

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(17, 8)
    contours = axs[0].contourf(x, z, V_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0], orientation="horizontal")
    contours = axs[0].contour(x, z, V_values, colors=["blue"], levels=[checkpoint["safe_level"]])
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$z$')
    axs[0].set_title('$V$')
    # axs[0].legend(fontsize=25)
    axs[0].set_xlim([-2.0, 1.0])
    axs[0].set_ylim([-0.5, 1.5])

    # for item in ([axs[0].title, axs[0].xaxis.label, axs[0].yaxis.label] +
    #              axs[0].get_xticklabels() + axs[0].get_yticklabels()):
    #     item.set_fontsize(25)
    # Add patches for unsafe region
    obs1 = patches.Rectangle((-1.0, -0.4), 0.5, 0.9, linewidth=1,
                             edgecolor='r', facecolor=obs_color, label="Unsafe Region")
    obs2 = patches.Rectangle((0.0, 0.8), 1.0, 0.6, linewidth=1,
                             edgecolor='r', facecolor=obs_color)
    ground = patches.Rectangle((-4.0, -4.0), 8.0, 3.7, linewidth=1,
                               edgecolor='r', facecolor=obs_color)
    axs[0].add_patch(obs1)
    axs[0].add_patch(obs2)
    axs[0].add_patch(ground)
    axs[0].plot([], [], c="blue", label="V(x) = c")
    axs[0].legend()

    unsafe_mask = unsafe_mask_fn(grid_x, grid_z).T
    V_dot_values[unsafe_mask] = 0.0

    contours = axs[1].contourf(x, z, V_dot_values, cmap="Greys", levels=20)

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    cbar = plt.colorbar(contours, ax=axs[1], orientation="horizontal", format=ticker.FuncFormatter(fmt))
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$z$')
    axs[1].set_title('$[dV/dt]_+$')
    # axs[1].legend(fontsize=25)
    axs[1].set_xlim([-2.0, 1.0])
    axs[1].set_ylim([-0.5, 1.5])

    # for item in ([axs[1].title, axs[1].xaxis.label, axs[1].yaxis.label] +
    #              axs[1].get_xticklabels() + axs[1].get_yticklabels()):
    #     item.set_fontsize(25)
    # Add patches for unsafe region
    obs1 = patches.Rectangle((-1.0, -0.4), 0.5, 0.9, linewidth=1,
                             edgecolor='r', facecolor=obs_color, label="Unsafe Region")
    obs2 = patches.Rectangle((0.0, 0.8), 1.0, 0.6, linewidth=1,
                             edgecolor='r', facecolor=obs_color)
    ground = patches.Rectangle((-4.0, -4.0), 8.0, 3.7, linewidth=1,
                               edgecolor='r', facecolor=obs_color)
    axs[1].add_patch(obs1)
    axs[1].add_patch(obs2)
    axs[1].add_patch(ground)

    fig.tight_layout()
    plt.show()
