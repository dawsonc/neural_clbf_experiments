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

#################################################
#
# In this file, we'll simulate the PVTOL system
# with a number of different controllers and
# compare the performance of the controllers
#
#################################################

# First simulate the robust CLF QP

# Load the robust model from file
filename = "logs/pvtol_robust_clf_qp.pth.tar"
filename = "logs/pvtol_model_best.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"m": low_m, "inertia": low_I}
scenarios = [
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": high_I},
    {"m": high_m, "inertia": low_I},
    {"m": high_m, "inertia": high_I},
]
robust_clf_net = CLF_QP_Net(n_dims,
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
robust_clf_net.load_state_dict(checkpoint['clf_net'])
nonrobust_clf_net = CLF_QP_Net(n_dims,
                               checkpoint['n_hidden'],
                               n_controls,
                               checkpoint['clf_lambda'],
                               10.0,  # checkpoint['relaxation_penalty'],
                               f_func,
                               g_func,
                               u_nominal,
                               [nominal_scenario],
                               nominal_scenario,
                               allow_relax=False)
nonrobust_clf_net.load_state_dict(checkpoint['clf_net'])

# Simulate some results
with torch.no_grad():
    N_sim = 5
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = 2.0
    x_sim_start[:, 1] = 2.0
    x_sim_start[:, 3] = 0.6
    x_sim_start[:, 4] = 0.6
    x_sim_start[:, 5] = 5.0

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(low_m, high_m)
    inertias = torch.Tensor(N_sim, 1).uniform_(low_I, high_I)

    t_sim = 5
    delta_t = 0.005
    num_timesteps = int(t_sim // delta_t)

    print("Simulating robust CLF QP controller...")
    x_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_rclfqp[0, :, :] = x_sim_start
    t_final_rclfqp = 0
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_sim_rclfqp[tstep - 1, :, :]
        # Get the control input at the current state
        try:
            u, r, V, Vdot = robust_clf_net(x_current)
        except Exception:
            print("Controller failed")
            break
        # u = u_nominal(x_current, **nominal_scenario)
        # Get the dynamics
        for i in range(N_sim):
            f_val = f_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            g_val = g_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            # Take one step to the future
            xdot = f_val + g_val @ u[i, :]
            x_sim_rclfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

        t_final_rclfqp = tstep

    # print("Simulating non-robust controller...")
    # x_sim_clfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    # x_sim_clfqp[0, :, :] = x_sim_start
    # t_final_clfqp = 0
    # for tstep in tqdm(range(1, num_timesteps)):
    #     # Get the current state
    #     x_current = x_sim_clfqp[tstep - 1, :, :]
    #     # Get the control input at the current state
    #     try:
    #         u, r, V, Vdot = nonrobust_clf_net(x_current)
    #     except Exception:
    #         print("Controller failed")
    #         break
    #     # u = u_nominal(x_current, **nominal_scenario)
    #     # Get the dynamics
    #     for i in range(N_sim):
    #         f_val = f_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
    #         g_val = g_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
    #         # Take one step to the future
    #         xdot = f_val + g_val @ u[i, :]
    #         x_sim_clfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

    #     t_final_clfqp = tstep

    print("Simulating LQR controller...")
    x_sim_lqr = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_lqr[0, :, :] = x_sim_start
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_sim_lqr[tstep - 1, :, :]
        # Get the control input at the current state
        u = u_nominal(x_current, **nominal_scenario)
        # Get the dynamics
        for i in range(N_sim):
            f_val = f_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            g_val = g_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            # Take one step to the future
            xdot = f_val + g_val @ u[i, :]
            x_sim_lqr[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

    fig, ax1 = plt.subplots()
    t = np.linspace(0, t_sim, num_timesteps)
    ax1.plot([], c=sns.color_palette("pastel")[0], label="Robust CLF QP")
    ax1.plot([], c=sns.color_palette("pastel")[1], label="CLF QP")
    ax1.plot([], c=sns.color_palette("pastel")[2], label="LQR")
    ax1.plot(t[:t_final_rclfqp], x_sim_rclfqp[:t_final_rclfqp, :, :].norm(dim=-1),
             c=sns.color_palette("pastel")[0])
    # ax1.plot(t[:t_final_clfqp], x_sim_clfqp[:t_final_clfqp, :, :].norm(dim=-1),
    #          c=sns.color_palette("pastel")[1])
    ax1.plot(t, x_sim_lqr.norm(dim=-1), c=sns.color_palette("pastel")[2])
    # ax1.plot(t, x_sim_rclfqp[:, :, 1], c=sns.color_palette("pastel")[0])
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$||q||$")
    ax1.legend()
    ax1.set_xlim([0, t_sim])

    fig.tight_layout()
    plt.show()
