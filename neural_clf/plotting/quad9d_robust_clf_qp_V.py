import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker
import seaborn as sns

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
from models.quad9d import (
    control_affine_dynamics,
    u_nominal,
    n_controls,
    n_dims,
    StateIndex,
    m_low,
    m_high,
)


# Beautify plots
sns.set_theme(context="talk", style="white")

# Load the model from file
filename = "logs/quad9d_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"m": m_low}
scenarios = [
    {"m": m_low},
    {"m": m_high},
]
clf_net = CLF_QP_Net(n_dims,
                     checkpoint['n_hidden'],
                     n_controls,
                     1.0,  # checkpoint['clf_lambda'],
                     float('inf'),  # checkpoint['relaxation_penalty'],
                     control_affine_dynamics,
                     u_nominal,
                     scenarios,
                     nominal_scenario)
clf_net.load_state_dict(checkpoint['clf_net'])
# clf_net.use_QP = False

with torch.no_grad():
    n_grid = 1000
    x = torch.linspace(-4, 4, n_grid)
    z = torch.linspace(-4, 1, n_grid)
    grid_x, grid_z = torch.meshgrid(x, z)
    residuals = torch.zeros(n_grid, n_grid)
    V_values = torch.zeros(n_grid, n_grid)
    V_dot_values = torch.zeros(n_grid, n_grid)
    print("Plotting V on grid...")
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            # Get the residual from running the model
            q = torch.zeros(1, n_dims)
            q[0, StateIndex.PX] = x[i]
            q[0, StateIndex.PZ] = z[j]
            _, r, V, V_dot = clf_net(q)
            residuals[j, i] = r
            V_values[j, i] = V
            V_dot_values[j, i] = V_dot

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(17, 8)
    contours = axs[0].contourf(x, -z, V_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0], orientation="horizontal")
    contours = axs[0].contour(x, -z, V_values, colors=["blue"], levels=[checkpoint["safe_level"]])

    # Draw safe set
    theta = np.linspace(0, 2*np.pi, 500)
    xy_safe = np.zeros((500, 2))
    xy_safe[:, 0] = checkpoint["safe_radius"] * np.cos(theta)
    xy_safe[:, 1] = np.maximum(checkpoint["safe_radius"] * np.sin(theta), -checkpoint["safe_z"])
    safe_patch = Polygon(xy_safe, color="g", fill=False)
    axs[0].add_patch(safe_patch)
    xy_unsafe = np.zeros((500, 2))
    xy_unsafe[:, 0] = checkpoint["unsafe_radius"] * np.cos(theta)
    xy_unsafe[:, 1] = np.maximum(checkpoint["unsafe_radius"] * np.sin(theta), -checkpoint["unsafe_z"])
    unsafe_patch = Polygon(xy_unsafe, color="r", fill=False)
    axs[0].add_patch(unsafe_patch)

    axs[0].plot([], [], c="g", label="Safe")
    axs[0].plot([], [], c="r", label="Unsafe")
    axs[0].plot([], [], c="blue", label="V(x) = c")
    # axs[0].plot([x.min(), x.max()], [-checkpoint["safe_z"], -checkpoint["safe_z"]],
    #             c="g", label="Safe")
    # axs[0].plot([x.min(), x.max()], [-checkpoint["unsafe_z"], -checkpoint["unsafe_z"]],
    #             c="r", label="Unsafe")
    # safe_circle = plt.Circle((0.0, 0.0), checkpoint["safe_radius"], color='g', fill=False)
    unsafe_circle = plt.Circle((0.0, 0.0), checkpoint["unsafe_radius"], color='r', fill=False)
    # axs[0].add_patch(safe_circle)
    # axs[0].add_patch(unsafe_circle)
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$z$')
    axs[0].set_title('$V$')
    axs[0].legend(loc="upper right")
    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-1, 4])

    contours = axs[1].contourf(x, -z, V_dot_values, cmap="Greys", levels=20)

    xy_safe = np.zeros((500, 2))
    xy_safe[:, 0] = checkpoint["safe_radius"] * np.cos(theta)
    xy_safe[:, 1] = np.maximum(checkpoint["safe_radius"] * np.sin(theta), -checkpoint["safe_z"])
    safe_patch = Polygon(xy_safe, color="g", fill=False)
    axs[1].add_patch(safe_patch)
    xy_unsafe = np.zeros((500, 2))
    xy_unsafe[:, 0] = checkpoint["unsafe_radius"] * np.cos(theta)
    xy_unsafe[:, 1] = np.maximum(checkpoint["unsafe_radius"] * np.sin(theta), -checkpoint["unsafe_z"])
    unsafe_patch = Polygon(xy_unsafe, color="r", fill=False)
    axs[1].add_patch(unsafe_patch)

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    cbar = plt.colorbar(contours, ax=axs[1], orientation="horizontal", format=ticker.FuncFormatter(fmt))
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
    axs[1].set_xlabel('$x$')
    axs[1].set_ylabel('$z$')
    axs[1].set_title('max($dV/dt$, 0)')
    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-1, 4])

    # plt.savefig("logs/plots/pvtol/lyap_contour.png")
    plt.show()
