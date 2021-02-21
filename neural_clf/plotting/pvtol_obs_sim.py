import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from neural_clf.controllers.clf_uK_qp_net import CLF_K_QP_Net
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
lqr_color = sns.color_palette("pastel")[0]
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
    # {"m": low_m, "inertia": high_I},
    # {"m": high_m, "inertia": low_I},
    # {"m": high_m, "inertia": high_I},
]
robust_clf_net = CLF_K_QP_Net(n_dims,
                              checkpoint['n_hidden'],
                              n_controls,
                              checkpoint['clf_lambda'],
                              checkpoint['relaxation_penalty'],
                              control_affine_dynamics,
                              u_nominal,
                              scenarios,
                              nominal_scenario,
                              checkpoint['x_goal'],
                              checkpoint['u_eq'])
robust_clf_net.load_state_dict(checkpoint['clf_net'])
robust_clf_net.use_QP = False

# Simulate some results
with torch.no_grad():
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = 1.1
    x_sim_start[:, 1] = 1.36

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(low_m, low_m)
    inertias = torch.Tensor(N_sim, 1).uniform_(low_I, low_I)

    t_sim = 2
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

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

            u_sim_rclfqp[tstep, :, :] = u.squeeze()
            V_sim_rclfqp[tstep, :, 0] = V
            Vdot_sim_rclfqp[tstep, :, 0] = Vdot.squeeze()
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :, 0]
                x_sim_rclfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_rclfqp = tstep
    except (Exception, KeyboardInterrupt):
        raise
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
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                Vdot_sim_lqr[tstep, :, 0] = (grad_V @ xdot.T).squeeze()
                x_sim_lqr[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    ax1.plot([], c=rclfqp_color, label="rCLF")
    ax1.plot([], c=lqr_color, label="LQR")
    ax1.plot(x_sim_rclfqp[:t_final_rclfqp, :, 0], x_sim_rclfqp[:t_final_rclfqp, :, 1],
             c=rclfqp_color)
    ax1.plot(x_sim_lqr[:, :, 0], x_sim_lqr[:, :, 1], c=lqr_color)
    ax1.plot(0.0, 0.0, 'ko', label="Goal")

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
    ax1.legend()
    ax1.set_xlim([-3.0, 2.0])
    ax1.set_ylim([-1.0, 3.0])

    ax2 = axs[0, 1]
    ax2.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="LQR $u1$")
    ax2.plot([], c=sns.color_palette("pastel")[0], linestyle=":", label="LQR $u2$")
    ax2.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLF $u1$")
    ax2.plot([], c=sns.color_palette("pastel")[1], linestyle=":", label="rCLF $u2$")
    ax2.plot()
    ax2.plot(t[1:t_final_rclfqp], u_sim_rclfqp[1:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax2.plot(t[1:t_final_rclfqp], u_sim_rclfqp[1:t_final_rclfqp, :, 1],
             c=sns.color_palette("pastel")[1], linestyle=":")
    ax2.plot(t[1:], u_sim_lqr[1:, :, 0],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax2.plot(t[1:], u_sim_lqr[1:, :, 1],
             c=sns.color_palette("pastel")[0], linestyle=":")
    ax2.legend()

    ax3 = axs[1, 0]
    ax3.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="LQR V")
    ax3.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLF V")
    ax3.plot()
    ax3.plot(t[1:t_final_rclfqp], V_sim_rclfqp[1:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax3.plot(t[1:], V_sim_lqr[1:, :, 0],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax3.legend()

    fig.tight_layout()
    plt.show()
