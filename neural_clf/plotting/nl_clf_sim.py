import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
from neural_clf.controllers.mpc import NlHoverMPC
from models.neural_lander import (
    control_affine_dynamics,
    u_nominal,
    n_controls,
    n_dims,
    StateIndex,
    mass,
)


# Beautify plots
sns.set_theme(context="talk", style="white")
obs_color = sns.color_palette("pastel")[3]
mpc_color = sns.color_palette("pastel")[0]
rclbfqp_color = sns.color_palette("pastel")[1]
nclbf_color = sns.color_palette("pastel")[2]
# quad_color = sns.color_palette("pastel")[2]

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
filename = "logs/nl_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"mass": mass}
scenarios = [
    {"mass": mass},
    {"mass": 2.0},
]

robust_clf_net = CLF_QP_Net(n_dims,
                            checkpoint['n_hidden'],
                            n_controls,
                            0.1,  # checkpoint['clf_lambda'],
                            10.0,  # checkpoint['relaxation_penalty'],
                            control_affine_dynamics,
                            u_nominal,
                            scenarios,
                            nominal_scenario)
robust_clf_net.load_state_dict(checkpoint['clf_net'])
# robust_clf_net.use_QP = False

# # Also load the non-robust model from file
# filename = "logs/pvtol_robust_clf_qp_single_scenario.pth.tar"
# checkpoint = torch.load(filename)
# nominal_scenario = {"m": low_m, "inertia": low_I}
# scenarios = [
#     {"m": low_m, "inertia": low_I},
# ]
# nonrobust_clf_net = CLF_QP_Net(n_dims,
#                                checkpoint['n_hidden'],
#                                n_controls,
#                                checkpoint['clf_lambda'],
#                                checkpoint['relaxation_penalty'],
#                                control_affine_dynamics,
#                                u_nominal,
#                                scenarios,
#                                nominal_scenario)
# nonrobust_clf_net.load_state_dict(checkpoint['clf_net'])

# Simulate some results
with torch.no_grad():
    N_sim = 2
    x_sim_start = torch.zeros(N_sim, n_dims) + 0.5
    x_sim_start[:, StateIndex.VZ] = -1

    t_sim = 2
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    # Get a random distribution of masses and inertias
    # ms = torch.Tensor(N_sim, 1).uniform_(mass, mass)
    ms = torch.linspace(mass, 2.0, N_sim)

    print("Simulating robust CLF QP controller...")
    x_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclbfqp[0, :, :] = x_sim_start
    t_final_rclbfqp = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_rclbfqp[tstep - 1, :, :]
            # Get the control input at the current state
            u, r, V, Vdot = robust_clf_net(x_current)

            u_sim_rclbfqp[tstep, :, :] = u
            V_sim_rclbfqp[tstep, :, 0] = V
            Vdot_sim_rclbfqp[tstep, :, 0] = Vdot.squeeze()
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0), mass=ms[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                x_sim_rclbfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_rclbfqp = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print("Simulating robust CLF pi proof controller...")
    x_sim_nclbf = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_nclbf = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_nclbf = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_nclbf = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_nclbf[0, :, :] = x_sim_start
    t_final_nclbf = 0
    robust_clf_net.use_QP = False
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_nclbf[tstep - 1, :, :]
            # Get the control input at the current state
            u, r, V, Vdot = robust_clf_net(x_current)

            u_sim_nclbf[tstep, :, :] = u
            V_sim_nclbf[tstep, :, 0] = V
            Vdot_sim_nclbf[tstep, :, 0] = Vdot.squeeze()
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0), mass=ms[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                x_sim_nclbf[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_nclbf = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print("Simulating mpc controller...")
    x_sim_mpc = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_mpc[0, :, :] = x_sim_start
    u_sim_mpc = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_mpc = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_mpc = torch.zeros(num_timesteps, N_sim, 1)
    mpc_ctrl_period = 1/24.0
    mpc_update_frequency = int(mpc_ctrl_period / delta_t)
    mpc_runtime = 0.0
    mpc_calls = 0.0
    t_final_mpc = 0
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
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0), mass=ms[i])

                # Get the control input at the current state if we're at the appropriate timing
                if tstep == 1 or tstep % mpc_update_frequency == 0:
                    u = torch.tensor(NlHoverMPC(x_current[i, :].numpy()))
                    mpc_calls += 1
                    u_sim_mpc[tstep, i, :] = u
                else:
                    u = u_sim_mpc[tstep - 1, i, :]
                    u_sim_mpc[tstep, i, :] = u

                # Take one step to the future
                xdot = f_val + g_val @ u
                Vdot_sim_mpc[tstep, :, 0] = (grad_V @ xdot.T).squeeze()
                x_sim_mpc[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    # fig, axs = plt.subplots(2, 2)
    fig, axs = plt.subplots(1, 1)
    t = np.linspace(0, t_sim, num_timesteps)
    # ax1 = axs[0, 0]
    ax1 = axs
    ax1.plot([], c=rclbfqp_color, label="rCLBF-QP")
    ax1.plot([], c=nclbf_color, label="rCLBF $\\pi_{proof}$")
    ax1.plot([], c=sns.color_palette("pastel")[0], label="MPC")
    # ax1.plot([], c="g", label="Safe")
    # ax1.plot([], c="r", label="Unsafe")
    ax1.fill_between(
        t,
        x_sim_nclbf[:, 0, StateIndex.PZ],
        x_sim_nclbf[:, -1, StateIndex.PZ],
        color=nclbf_color,
        alpha=0.9)
    ax1.fill_between(
        t,
        x_sim_mpc[:, 0, StateIndex.PZ],
        x_sim_mpc[:, -1, StateIndex.PZ],
        color=mpc_color,
        alpha=0.9)
    ax1.fill_between(
        t[:t_final_rclbfqp],
        x_sim_rclbfqp[:t_final_rclbfqp, 0, StateIndex.PZ],
        x_sim_rclbfqp[:t_final_rclbfqp, -1, StateIndex.PZ],
        color=rclbfqp_color,
        alpha=0.9)
    ax1.plot(t, t * 0.0 + checkpoint["safe_z"], c="g")
    ax1.text(8, 0.1 + checkpoint["safe_z"], "Safe", fontsize=20)
    ax1.plot(t, t * 0.0 + checkpoint["unsafe_z"], c="r")
    ax1.text(8, -0.5 + checkpoint["unsafe_z"], "Unsafe", fontsize=20)

    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$z$")
    ax1.legend(loc="upper left")
    ax1.set_xlim([0, t_sim])
    ax1.set_ylim([-1, 2])

    # fig, axs = plt.subplots(2, 2)
    # t = np.linspace(0, t_sim, num_timesteps)
    # ax1 = axs[0, 0]
    # ax1.plot([], c=sns.color_palette("pastel")[1], label="rCLF")
    # ax1.plot([], c=sns.color_palette("pastel")[0], label="mpc")
    # ax1.plot(t[:t_final_rclbfqp], x_sim_rclbfqp[:t_final_rclbfqp, :, StateIndex.PZ],
    #          c=sns.color_palette("pastel")[1])
    # ax1.plot(t, x_sim_mpc[:, :, StateIndex.PZ], c=sns.color_palette("pastel")[0])
    # ax1.plot(t, t * 0.0 + checkpoint["safe_z"], c="g")
    # ax1.plot(t, t * 0.0 + checkpoint["unsafe_z"], c="r")

    # ax1.set_xlabel("$t$")
    # ax1.set_ylabel("$z$")
    # ax1.legend()
    # ax1.set_xlim([0, t_sim])

    # ax3 = axs[1, 1]
    # ax3.plot([], c=sns.color_palette("pastel")[0], label="mpc V")
    # ax3.plot([], c=sns.color_palette("pastel")[1], label="rCLF V")
    # ax3.plot(t[1:], V_sim_mpc[1:, :, 0],
    #          c=sns.color_palette("pastel")[0])
    # ax3.plot(t[1:t_final_rclbfqp], V_sim_rclbfqp[1:t_final_rclbfqp, :, 0],
    #          c=sns.color_palette("pastel")[1])
    # ax3.plot(t, t * 0.0, c="k")
    # ax3.legend()

    # ax2 = axs[0, 1]
    # ax2.plot([], c=sns.color_palette("pastel")[0], label="mpc dV/dt")
    # ax2.plot([], c=sns.color_palette("pastel")[1], label="rCLF dV/dt")
    # ax2.plot(t[1:t_final_rclbfqp], Vdot_sim_rclbfqp[1:t_final_rclbfqp, :, 0],
    #          c=sns.color_palette("pastel")[1])
    # ax2.plot(t[1:], Vdot_sim_mpc[1:, :, 0],
    #          c=sns.color_palette("pastel")[0])
    # ax2.plot(t, t * 0.0, c="k")
    # ax2.legend()

    # ax4 = axs[1, 0]
    # ax4.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="mpc $a_z$")
    # ax4.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLF $a_z$")
    # ax4.plot()
    # ax4.plot(t[1:t_final_rclbfqp], u_sim_rclbfqp[1:t_final_rclbfqp, :, 2],
    #          c=sns.color_palette("pastel")[1], linestyle="-")
    # ax4.plot(t[1:], u_sim_mpc[1:, :, 2],
    #          c=sns.color_palette("pastel")[0], linestyle="-")
    # ax4.legend()

    fig.tight_layout()
    plt.show()
