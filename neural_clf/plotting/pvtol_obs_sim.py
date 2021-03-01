import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
from neural_clf.controllers.mpc import PVTOLObsMPC
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
mpc_color = sns.color_palette("pastel")[0]
rclfqp_color = sns.color_palette("pastel")[1]

#################################################
#
# In this file, we'll simulate the PVTOL system
# with a number of different controllers and
# compare the performance of the controllers
#
#################################################
torch.set_default_dtype(torch.float64)

# First simulate the robust CLF QP

# Load the robust model from file
filename = "logs/pvtol_obs_clf.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"m": low_m, "inertia": low_I}
scenarios = [
    {"m": low_m, "inertia": low_I},
    {"m": low_m, "inertia": low_I * 1.05},
    {"m": low_m * 1.05, "inertia": low_I},
    {"m": low_m * 1.05, "inertia": low_I * 1.05},
    # {"m": low_m, "inertia": high_I},
    # {"m": high_m, "inertia": low_I},
    # {"m": high_m, "inertia": high_I},
]
robust_clf_net = CLF_QP_Net(n_dims,
                            checkpoint['n_hidden'],
                            n_controls,
                            7.0,  # checkpoint['clf_lambda'],
                            1100.0,  # checkpoint['relaxation_penalty'],
                            control_affine_dynamics,
                            u_nominal,
                            scenarios,
                            nominal_scenario,
                            use_casadi=True)
robust_clf_net.load_state_dict(checkpoint['clf_net'])
# robust_clf_net.use_QP = False

# Simulate some results
with torch.no_grad():
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = -1.5
    x_sim_start[:, 1] = 0.1

    # Get a random distribution of masses and inertias
    # ms = torch.linspace(low_m, low_m * 1.05, N_sim)
    # inertias = torch.linspace(low_I, low_I * 1.05, N_sim)
    ms = torch.linspace(1.05 * low_m, 1.05 * low_m, N_sim)
    inertias = torch.linspace(1.0 * low_I, 1.0 * low_I, N_sim)
    # title_string = "$m=1.00$, $I=0.0100$"
    title_string = ""

    t_sim = 5.0
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    print("Simulating robust CLF QP controller...")
    x_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    r_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclfqp[0, :, :] = x_sim_start
    t_final_rclfqp = 0
    rclbf_runtime = 0.0
    rclbf_calls = 0.0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_rclfqp[tstep - 1, :, :]
            # Get the control input at the current state
            ts = time.time()
            u, r, V, Vdot = robust_clf_net(x_current)
            tf = time.time()
            rclbf_runtime += tf - ts
            rclbf_calls += 1

            u_sim_rclfqp[tstep, :, :] = u
            V_sim_rclfqp[tstep, :, 0] = V
            Vdot_sim_rclfqp[tstep, :, 0] = Vdot.squeeze()
            # r_sim_rclfqp[tstep, :, 0] = r
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                x_sim_rclfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_rclfqp = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print("Simulating LQR controller...")
    x_sim_mpc = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_mpc[0, :, :] = x_sim_start
    u_sim_mpc = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_mpc = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_mpc = torch.zeros(num_timesteps, N_sim, 1)
    obs_pos = np.array([[-0.75, 0.25], [0.0, 1.1], [0.6, 1.1], [1.0, 1.1]])
    obs_r = np.array([0.36, 0.3, 0.3, 0.3])
    mpc_ctrl_period = 1/24.0
    mpc_update_frequency = int(mpc_ctrl_period / delta_t)
    t_final_mpc = 0.0
    mpc_runtime = 0.0
    mpc_calls = 0.0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_mpc[tstep - 1, :, :]

            # and measure the Lyapunov function value here
            V, grad_V = robust_clf_net.compute_lyapunov(x_current)

            u_sim_mpc[tstep, :, :] = u
            V_sim_mpc[tstep, :, 0] = V
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Get the control input at the current state if we're at the appropriate timing
                if tstep == 1 or tstep % mpc_update_frequency == 0:
                    ts = time.time()
                    u = torch.tensor(PVTOLObsMPC(x_current[i, :].numpy(), obs_pos, obs_r))
                    tf = time.time()
                    mpc_runtime += tf - ts
                    mpc_calls += 1
                    u_sim_mpc[tstep, i, :] = u
                else:
                    u = u_sim_mpc[tstep - 1, i, :]
                    u_sim_mpc[tstep, i, :] = u

                # Take one step to the future
                xdot = f_val + g_val @ u
                Vdot_sim_mpc[tstep, :, 0] = (grad_V @ xdot.T).squeeze()
                x_sim_mpc[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
            t_final_mpc = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print(f"rCLBF qp total runtime = {rclbf_runtime} s ({rclbf_runtime / rclbf_calls} s per iteration)")
    print(f"MPC total runtime = {mpc_runtime} s ({mpc_runtime / mpc_calls} s per iteration)")

    # fig, axs = plt.subplots(2, 2)
    fig, axs = plt.subplots(1, 1)
    t = np.linspace(0, t_sim, num_timesteps)
    # ax1 = axs[0, 0]
    ax1 = axs
    ax1.plot([], c=rclfqp_color, label="rCLF-QP", linewidth=6)
    ax1.plot([], c=mpc_color, label="MPC", linewidth=6)
    ax1.plot(x_sim_rclfqp[:t_final_rclfqp, :, 0], x_sim_rclfqp[:t_final_rclfqp, :, 1],
             c=rclfqp_color, linewidth=6)
    ax1.plot(x_sim_mpc[:t_final_mpc, :, 0], x_sim_mpc[:t_final_mpc, :, 1], c=mpc_color, linewidth=6)
    ax1.scatter([], [], label="Goal", s=1000, facecolors='none', edgecolors='k',
                linestyle='--')
    ax1.scatter([0.0], [0.0], s=10000, facecolors='none', edgecolors='k',
                linestyle='--')

    # Add patches for unsafe region
    obs1 = patches.Rectangle((-1.0, -0.4), 0.5, 0.9, linewidth=1,
                             edgecolor='r', facecolor=obs_color, label="Unsafe Region")
    obs2 = patches.Rectangle((0.0, 0.8), 1.0, 0.6, linewidth=1,
                             edgecolor='r', facecolor=obs_color)
    ground = patches.Rectangle((-4.0, -4.0), 8.0, 3.7, linewidth=1,
                               edgecolor='r', facecolor=obs_color)
    ax1.add_patch(obs1)
    ax1.add_patch(obs2)
    ax1.add_patch(ground)

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$z$")
    ax1.set_title(title_string)
    ax1.legend(fontsize=25, loc="upper left")
    ax1.set_xlim([-2.0, 1.0])
    ax1.set_ylim([-0.5, 1.5])

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(25)

    fig.tight_layout()
    plt.show()
