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

#################################################
#
# In this file, we'll simulate the PVTOL system
# with a number of different controllers and
# compare the performance of the controllers
#
#################################################

# First simulate the robust CLF QP

# Load the robust model from file
filename = "logs/pvtol_robust_clf_cbf_qp.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"m": high_m, "inertia": high_I}
scenarios = [
    # {"m": low_m, "inertia": low_I},
    # {"m": low_m, "inertia": high_I},
    # {"m": high_m, "inertia": low_I},
    {"m": high_m, "inertia": high_I},
]
robust_clf_cbf_net = CLF_CBF_QP_Net(n_dims,
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
                                    nominal_scenario,
                                    allow_cbf_relax=False)
robust_clf_cbf_net.load_state_dict(checkpoint['clf_cbf_net'])
# nonrobust_clf_cbf_net = CLF_CBF_QP_Net(n_dims,
#                                        checkpoint['n_hidden'],
#                                        n_controls,
#                                        checkpoint['clf_lambda'],
#                                        checkpoint['cbf_lambda'],
#                                        checkpoint['clf_relaxation_penalty'],
#                                        checkpoint['cbf_relaxation_penalty'],
#                                        f_func,
#                                        g_func,
#                                        u_nominal,
#                                        [nominal_scenario],
#                                        nominal_scenario)
# nonrobust_clf_cbf_net.load_state_dict(checkpoint['clf_cbf_net'])

# Simulate some results
with torch.no_grad():
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = 0.0
    x_sim_start[:, 1] = 0.0
    x_sim_start[:, 3] = 0.0
    x_sim_start[:, 4] = -2.0
    x_sim_start[:, 5] = 0.0

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(high_m, high_m)
    inertias = torch.Tensor(N_sim, 1).uniform_(high_I, high_I)

    t_sim = 1
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    print("Simulating robust CLF/CBF QP controller...")
    x_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    H_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Hdot_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Hdot_est_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    r_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclfqp[0, :, :] = x_sim_start
    t_final_rclfqp = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_rclfqp[tstep - 1, :, :]
            # Get the control input at the current state
            u, r, V, Vdot, H, Hdot = robust_clf_cbf_net(x_current)

            u_sim_rclfqp[tstep, :, :] = u.squeeze()
            V_sim_rclfqp[tstep, :, 0] = V.squeeze()
            Vdot_sim_rclfqp[tstep, :, 0] = Vdot.squeeze()
            H_sim_rclfqp[tstep, :, 0] = H.squeeze()
            Hdot_sim_rclfqp[tstep, :, 0] = Hdot.squeeze()
            Hdot_est_rclfqp[tstep - 1, :, 0] = H_sim_rclfqp[tstep, :, 0] - H_sim_rclfqp[tstep - 1, :, 0]
            Hdot_est_rclfqp[tstep - 1, :, 0] /= delta_t
            r_sim_rclfqp[tstep, :, 0] = r.squeeze()
            # Get the dynamics
            for i in range(N_sim):
                f_val = f_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
                g_val = g_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                x_sim_rclfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_rclfqp = tstep

            if x_sim_rclfqp[tstep, 0, 1] <= checkpoint["unsafe_z"]:
                break
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    # print("Simulating non-robust controller...")
    # x_sim_clfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    # x_sim_clfqp[0, :, :] = x_sim_start
    # t_final_clfqp = 0
    # for tstep in tqdm(range(1, num_timesteps)):
    #     # Get the current state
    #     x_current = x_sim_clfqp[tstep - 1, :, :]
    #     # Get the control input at the current state
    #     try:
    #         u, r, V, Vdot = nonrobust_clf_cbf_net(x_current)
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
    u_sim_lqr = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_lqr = torch.zeros(num_timesteps, N_sim, 1)
    H_sim_lqr = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_lqr[0, :, :] = x_sim_start
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_sim_lqr[tstep - 1, :, :]
        # Get the control input at the current state
        u = u_nominal(x_current, **nominal_scenario)
        u_sim_lqr[tstep, :, :] = u.squeeze()
        # Get the barrier function at the current state
        _, _, V, _, H, _ = robust_clf_cbf_net(x_current)
        H_sim_lqr[tstep, :, 0] = H.squeeze()
        V_sim_lqr[tstep, :, 0] = V.squeeze()
        # Get the dynamics
        for i in range(N_sim):
            f_val = f_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            g_val = g_func(x_current[i, :].unsqueeze(0), m=ms[i], inertia=inertias[i])
            # Take one step to the future
            xdot = f_val + g_val @ u[i, :]
            x_sim_lqr[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    ax1.plot([], c=sns.color_palette("pastel")[1], label="Robust CLF QP")
    ax1.plot([], c=sns.color_palette("pastel")[0], label="LQR")
    ax1.plot(t[:t_final_rclfqp], x_sim_rclfqp[:t_final_rclfqp, :, 1],
             c=sns.color_palette("pastel")[1])
    ax1.plot(t, x_sim_lqr[:, :, 1], c=sns.color_palette("pastel")[0])
    ax1.plot(t, t * 0.0 + checkpoint["safe_z"], c="g")
    ax1.plot(t, t * 0.0 + checkpoint["unsafe_z"], c="r")

    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$y$")
    ax1.legend()
    ax1.set_xlim([0, t_sim])

    ax2 = axs[0, 1]
    ax2.plot([], c=sns.color_palette("pastel")[0], label="LQR H")
    ax2.plot([], c=sns.color_palette("pastel")[1], label="CLF H")
    ax2.plot(t, H_sim_lqr[:, :, 0],
             c=sns.color_palette("pastel")[0])
    ax2.plot(t[:t_final_rclfqp], H_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax2.plot(t, t * 0.0, c="k")
    ax2.legend()

    ax3 = axs[1, 1]
    ax3.plot([], c=sns.color_palette("pastel")[0], label="LQR V")
    ax3.plot([], c=sns.color_palette("pastel")[1], label="CLF V")
    ax3.plot(t, V_sim_lqr[:, :, 0],
             c=sns.color_palette("pastel")[0])
    ax3.plot(t[:t_final_rclfqp], V_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax3.plot(t, t * 0.0, c="k")
    ax3.legend()

    ax4 = axs[1, 0]
    ax4.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="LQR u1")
    ax4.plot([], c=sns.color_palette("pastel")[0], linestyle=":", label="LQR u1")
    ax4.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="CLF u1")
    ax4.plot([], c=sns.color_palette("pastel")[1], linestyle=":", label="CLF u1")
    ax4.plot()
    ax4.plot(t[:t_final_rclfqp], u_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax4.plot(t[:t_final_rclfqp], u_sim_rclfqp[:t_final_rclfqp, :, 1],
             c=sns.color_palette("pastel")[1], linestyle=":")
    ax4.plot(t, u_sim_lqr[:, :, 0],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax4.plot(t, u_sim_lqr[:, :, 1],
             c=sns.color_palette("pastel")[0], linestyle=":")
    ax4.legend()

    fig.tight_layout()
    plt.show()
