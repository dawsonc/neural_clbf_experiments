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
filename = "logs/pvtol_model_best.pth.tar"
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

# Simulate some results
with torch.no_grad():
    N_sim = 10
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = 2
    x_sim_start[:, 1] = 2
    x_sim_start[:, 3] = 0.6
    x_sim_start[:, 4] = 0.6
    x_sim_start[:, 5] = 5

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

        u_sim[tstep, :, :] = u
        V_sim[tstep, :, 0] = V.squeeze()
        Vdot_sim[tstep, :, 0] = Vdot.squeeze()
        r_sim[tstep, :, 0] = r.squeeze()

    fig, ax1 = plt.subplots()
    t = np.linspace(0, t_sim, num_timesteps)
    ax1.plot(t, x_sim[:, :, :3].norm(dim=-1))
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$||q||$")
    ax1.set_xlim([0, 5])

    ax2 = ax1.twinx()
    ax2.plot(t, V_sim[:, :, 0])
    ax2.set_ylabel('V', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()
