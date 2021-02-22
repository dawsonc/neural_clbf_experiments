import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
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

#################################################
#
# In this file, we'll simulate the PVTOL system
# with a number of different controllers and
# compare the performance of the controllers
#
#################################################

# First simulate the robust CLF QP

# Load the robust model from file
filename = "logs/nl_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"mass": mass}
scenarios = [
    {"mass": mass},
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
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims) + 0.5
    x_sim_start[:, StateIndex.VZ] = -0.5

    t_sim = 2
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(mass, mass)

    print("Simulating robust CLF QP controller...")
    x_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclfqp[0, :, :] = x_sim_start
    t_final_rclfqp = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_rclfqp[tstep - 1, :, :]
            # Get the control input at the current state
            u, r, V, Vdot = robust_clf_net(x_current)

            u_sim_rclfqp[tstep, :, :] = u
            V_sim_rclfqp[tstep, :, 0] = V
            Vdot_sim_rclfqp[tstep, :, 0] = Vdot.squeeze()
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0), mass=ms[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                x_sim_rclfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_rclfqp = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print("Simulating LQR controller...")
    x_sim_lqr = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_lqr[0, :, :] = x_sim_start
    u_sim_lqr = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_lqr = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_lqr = torch.zeros(num_timesteps, N_sim, 1)
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_lqr[tstep - 1, :, :]
            # Get the control input at the current state
            u = u_nominal(x_current, **nominal_scenario)
            # and measure the Lyapunov function value here
            V, grad_V = robust_clf_net.compute_lyapunov(x_current)

            u_sim_lqr[tstep, :, :] = u
            V_sim_lqr[tstep, :, 0] = V
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0))
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                Vdot_sim_lqr[tstep, :, 0] = (grad_V @ xdot.T).squeeze()
                x_sim_lqr[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    ax1.plot([], c=sns.color_palette("pastel")[1], label="rCLF")
    ax1.plot([], c=sns.color_palette("pastel")[0], label="LQR")
    ax1.plot(t[:t_final_rclfqp], x_sim_rclfqp[:t_final_rclfqp, :, StateIndex.PZ],
             c=sns.color_palette("pastel")[1])
    ax1.plot(t, x_sim_lqr[:, :, StateIndex.PZ], c=sns.color_palette("pastel")[0])
    ax1.plot(t, t * 0.0 + checkpoint["safe_z"], c="g")
    ax1.plot(t, t * 0.0 + checkpoint["unsafe_z"], c="r")

    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$z$")
    ax1.legend()
    ax1.set_xlim([0, t_sim])

    ax3 = axs[1, 1]
    ax3.plot([], c=sns.color_palette("pastel")[0], label="LQR V")
    ax3.plot([], c=sns.color_palette("pastel")[1], label="rCLF V")
    ax3.plot(t[1:], V_sim_lqr[1:, :, 0],
             c=sns.color_palette("pastel")[0])
    ax3.plot(t[1:t_final_rclfqp], V_sim_rclfqp[1:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax3.plot(t, t * 0.0, c="k")
    ax3.legend()

    ax2 = axs[0, 1]
    ax2.plot([], c=sns.color_palette("pastel")[0], label="LQR dV/dt")
    ax2.plot([], c=sns.color_palette("pastel")[1], label="rCLF dV/dt")
    ax2.plot(t[1:t_final_rclfqp], Vdot_sim_rclfqp[1:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax2.plot(t[1:], Vdot_sim_lqr[1:, :, 0],
             c=sns.color_palette("pastel")[0])
    ax2.plot(t, t * 0.0, c="k")
    ax2.legend()

    ax4 = axs[1, 0]
    ax4.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="LQR $a_z$")
    ax4.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLF $a_z$")
    ax4.plot()
    ax4.plot(t[1:t_final_rclfqp], u_sim_rclfqp[1:t_final_rclfqp, :, 2],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax4.plot(t[1:], u_sim_lqr[1:, :, 2],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax4.legend()

    fig.tight_layout()
    plt.show()