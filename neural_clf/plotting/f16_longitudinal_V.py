import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clf.controllers.lf_net_f16_longitudinal import LF_Net
from models.f16_longitudinal import (
    dynamics,
    u_nominal,
    n_controls,
    n_dims,
)


# Beautify plots
sns.set_theme(context="talk", style="white")

# Load the model from file
filename = "logs/f16_lf_longitudinal.pth.tar"
checkpoint = torch.load(filename)
clf_net = LF_Net(n_dims,
                 checkpoint['n_hidden'],
                 n_controls,
                 checkpoint['clf_lambda'],
                 checkpoint['relaxation_penalty'],
                 dynamics,
                 u_nominal)
clf_net.load_state_dict(checkpoint['lf_net'])

with torch.no_grad():
    n_grid = 50
    alt = torch.linspace(200, 800, n_grid)
    vt = torch.linspace(400, 600, n_grid)
    V_values = torch.zeros(n_grid, n_grid)
    V_dot_values = torch.zeros(n_grid, n_grid)
    Nz_values = torch.zeros(n_grid, n_grid)
    print("Plotting V on grid...")
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            # Get the residual from running the model
            q = torch.zeros(1, n_dims)
            q[0, 0] += vt[i]
            q[0, 4] += alt[j]
            u, V, V_dot = clf_net(q)
            Nz_nominal, _, _, throttle_nominal = u_nominal(q, alt_setpoint=0)
            Nz_nominal = torch.zeros_like(Nz_nominal) - 1000
            u = torch.hstack((Nz_nominal.unsqueeze(-1), throttle_nominal.unsqueeze(-1)))

            V_values[j, i] = V
            V_dot_values[j, i] = V_dot
            _, Nz = dynamics(q, u, return_Nz=True)
            Nz_values[j, i] = Nz

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(17, 8)
    contours = axs[0].contourf(alt, vt, V_values, cmap="magma", levels=20)
    plt.colorbar(contours, ax=axs[0], orientation="horizontal")
    contours = axs[0].contour(alt, vt, V_values, colors=["blue"], levels=[checkpoint["safe_level"]])
    contours = axs[0].contour(alt, vt, Nz_values, colors=["red", "orange"], levels=[-2, 9])
    axs[0].set_xlabel('$vt$')
    axs[0].set_ylabel('$alt$')
    axs[0].set_title('$V$')
    axs[0].legend()

    contours = axs[1].contourf(vt, alt, V_dot_values, cmap="magma", levels=[-1, 0.0, 1.0])
    plt.colorbar(contours, ax=axs[1], orientation="horizontal")
    axs[1].set_xlabel('$vt$')
    axs[1].set_ylabel('$alt$')
    axs[1].set_title('$dV/dt$')

    # plt.savefig("logs/plots/pvtol/lyap_contour.png")
    plt.show()
