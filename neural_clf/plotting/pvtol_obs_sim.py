import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
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
rclbfqp_color = sns.color_palette("pastel")[1]
quad_color = sns.color_palette("pastel")[2]

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
    {"m": low_m, "inertia": high_I},
    {"m": high_m, "inertia": low_I},
    {"m": high_m, "inertia": high_I},
]
robust_clf_net = CLF_QP_Net(n_dims,
                            checkpoint['n_hidden'],
                            n_controls,
                            10.0,  # checkpoint['clf_lambda'],
                            1000.0,  # checkpoint['relaxation_penalty'],
                            control_affine_dynamics,
                            u_nominal,
                            scenarios,
                            nominal_scenario)
robust_clf_net.load_state_dict(checkpoint['clf_net'])
# robust_clf_net.use_QP = False

# Simulate some results
with torch.no_grad():
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = -1.5
    x_sim_start[:, 1] = 0.1

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(low_m, high_m)
    inertias = torch.Tensor(N_sim, 1).uniform_(low_I, high_I)

    t_sim = 2
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    print("Simulating robust CLF QP controller...")
    x_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    r_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclbfqp[0, :, :] = x_sim_start
    t_final_rclbfqp = 0
    rclbf_runtime = 0.0
    rclbf_calls = 0.0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_rclbfqp[tstep - 1, :, :]
            # Get the control input at the current state
            ts = time.time()
            u, r, V, Vdot = robust_clf_net(x_current)
            tf = time.time()
            rclbf_runtime += tf - ts
            rclbf_calls += 1

            u_sim_rclbfqp[tstep, :, :] = u
            V_sim_rclbfqp[tstep, :, 0] = V
            Vdot_sim_rclbfqp[tstep, :, 0] = Vdot.squeeze()
            r_sim_rclbfqp[tstep, :, 0] = r.squeeze()
            # Get the dynamics
            for i in range(N_sim):
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[i, :]
                x_sim_rclbfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()

            t_final_rclbfqp = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print("Simulating mpc controller...")
    x_sim_mpc = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_mpc[0, :, :] = x_sim_start
    u_sim_mpc = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_mpc = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_mpc = torch.zeros(num_timesteps, N_sim, 1)
    mpc_ctrl_period = 1/20.0
    mpc_update_frequency = int(mpc_ctrl_period / delta_t)
    obs_pos = np.array([[-0.75, 0.25], [0.0, 1.1], [0.6, 1.1], [1.0, 1.1]])
    obs_r = np.array([0.354, 0.3, 0.3, 0.3])
    mpc_runtime = 0.0
    mpc_calls = 0.0
    t_final_mpc = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_mpc[tstep - 1, :, :]

            # and measure the Lyapunov function value here
            V, grad_V = robust_clf_net.compute_lyapunov(x_current)

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

    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    ax1.plot([], c=rclbfqp_color, label="rCLBF")
    ax1.plot([], c=mpc_color, label="MPC")
    ax1.plot(x_sim_rclbfqp[:t_final_rclbfqp, :, 0], x_sim_rclbfqp[:t_final_rclbfqp, :, 1],
             c=rclbfqp_color)
    ax1.plot(x_sim_mpc[:t_final_mpc, :, 0], x_sim_mpc[:t_final_mpc, :, 1], c=mpc_color)
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
    ax2.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="MPC $u1$")
    ax2.plot([], c=sns.color_palette("pastel")[0], linestyle=":", label="MPC $u2$")
    ax2.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLBF $u1$")
    ax2.plot([], c=sns.color_palette("pastel")[1], linestyle=":", label="rCLBF $u2$")
    ax2.plot()
    ax2.plot(t[1:t_final_rclbfqp], u_sim_rclbfqp[1:t_final_rclbfqp, :, 0],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax2.plot(t[1:t_final_rclbfqp], u_sim_rclbfqp[1:t_final_rclbfqp, :, 1],
             c=sns.color_palette("pastel")[1], linestyle=":")
    ax2.plot(t[1:t_final_mpc], u_sim_mpc[1:t_final_mpc, :, 0],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax2.plot(t[1:t_final_mpc], u_sim_mpc[1:t_final_mpc, :, 1],
             c=sns.color_palette("pastel")[0], linestyle=":")
    ax2.legend()

    ax3 = axs[1, 0]
    ax3.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="MPC V")
    ax3.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLBF V")
    ax3.plot()
    ax3.plot(t[1:t_final_rclbfqp], V_sim_rclbfqp[1:t_final_rclbfqp, :, 0],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax3.plot(t[1:t_final_mpc], V_sim_mpc[1:t_final_mpc, :, 0],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax3.legend()

    ax4 = axs[1, 1]
    ax4.plot([], c=sns.color_palette("pastel")[0], linestyle="-", label="MPC dV/dt")
    ax4.plot([], c=sns.color_palette("pastel")[1], linestyle="-", label="rCLBF dV/dt")
    ax4.plot()
    ax4.plot(t[1:t_final_rclbfqp], Vdot_sim_rclbfqp[1:t_final_rclbfqp, :, 0],
             c=sns.color_palette("pastel")[1], linestyle="-")
    ax4.plot(t[1:t_final_rclbfqp], r_sim_rclbfqp[1:t_final_rclbfqp, :, 0],
             c=sns.color_palette("pastel")[2], linestyle="-")
    ax4.plot(t[1:t_final_mpc], Vdot_sim_mpc[1:t_final_mpc, :, 0],
             c=sns.color_palette("pastel")[0], linestyle="-")
    ax4.legend()

    fig.tight_layout()
    plt.show()

    # # Animate the neural controller
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # Add patches for unsafe region
    # obs1 = patches.Rectangle((-1.0, -0.4), 0.5, 0.9, linewidth=1,
    #                          edgecolor='r', facecolor=obs_color, label="Unsafe Region")
    # obs2 = patches.Rectangle((0.0, 0.8), 1.0, 0.6, linewidth=1,
    #                          edgecolor='r', facecolor=obs_color)
    # ground = patches.Rectangle((-4.0, -4.0), 8.0, 3.7, linewidth=1,
    #                            edgecolor='r', facecolor=obs_color)
    # ax.add_patch(obs1)
    # ax.add_patch(obs2)
    # ax.add_patch(ground)

    # ax.set_xlabel("$x$")
    # ax.set_ylabel("$z$")
    # ax.set_xlim([-3.0, 2.0])
    # ax.set_ylim([-1.0, 3.0])

    # quad_clf = patches.Rectangle((0.0, 0.0), 0.25, 0.1, linewidth=1,
    #                              facecolor=rclbfqp_color, edgecolor=rclbfqp_color, label="rCLBF")
    # ax.add_patch(quad_clf)
    # quad_mpc = patches.Rectangle((0.0, 0.0), 0.25, 0.1, linewidth=1,
    #                              facecolor=mpc_color, edgecolor=mpc_color, label="MPC")
    # ax.add_patch(quad_mpc)
    # ax.legend()

    # def animate(i):
    #     # i is the frame. At 30 fps, t = i/30
    #     t_index = int((i / 30) / delta_t)
    #     t_index = min(t_index, t_final_rclbfqp)
    #     quad_clf.set_xy([x_sim_rclbfqp[t_index, 0, 0], x_sim_rclbfqp[t_index, 0, 1]])
    #     quad_clf._angle = -np.rad2deg(x_sim_rclbfqp[t_index, 0, 2])
    #     quad_mpc.set_xy([x_sim_mpc[t_index, 0, 0], x_sim_mpc[t_index, 0, 1]])
    #     quad_mpc._angle = -np.rad2deg(x_sim_mpc[t_index, 0, 2])
    #     return quad_clf, quad_mpc,

    # anim = FuncAnimation(fig, animate, interval=1000/30, frames=5 * 30)

    # plt.show()
    # plt.draw()
    # anim.save('logs/plots/pvtol/pvtol_obs.mov')
