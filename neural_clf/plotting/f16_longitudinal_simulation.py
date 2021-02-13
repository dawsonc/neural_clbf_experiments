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

# Simulate some results
with torch.no_grad():
    N_sim = 1
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = 550    # vt
    x_sim_start[:, 4] = 550    # alt

    t_sim = 15
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    print("Simulating robust LF controller...")
    x_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Nz_sim_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    Nz_nominal_rclfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclfqp[0, :, :] = x_sim_start
    t_final_rclfqp = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_rclfqp[tstep - 1, :, :]
            # Get the control input at the current state
            u, V, Vdot = clf_net(x_current)
            Nz_nominal_rclfqp[tstep, :, 0] = u[:, 0]

            u_sim_rclfqp[tstep, :, :] = u
            V_sim_rclfqp[tstep, :, 0] = V
            Vdot_sim_rclfqp[tstep, :, 0] = Vdot
            # Get the dynamics
            for i in range(N_sim):
                xdot, Nz = dynamics(x_current[i, :].unsqueeze(0), u, return_Nz=True)
                # Take one step to the future
                x_sim_rclfqp[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
                Nz_sim_rclfqp[tstep, i, :] = Nz

            t_final_rclfqp = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print("Simulating LQR controller...")
    x_sim_lqr = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_lqr[0, :, :] = x_sim_start
    u_sim_lqr = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_lqr = torch.zeros(num_timesteps, N_sim, 1)
    Nz_sim_lqr = torch.zeros(num_timesteps, N_sim, 1)
    Nz_nominal_lqr = torch.zeros(num_timesteps, N_sim, 1)
    t_final_lqr = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_lqr[tstep - 1, :, :]
            # Get the control input at the current state
            Nz_nominal, _, _, throttle_nominal = u_nominal(x_current)
            Nz_nominal_lqr[tstep, :, 0] = Nz_nominal
            u = torch.hstack((Nz_nominal.unsqueeze(-1), throttle_nominal.unsqueeze(-1)))
            # and measure the Lyapunov function value here
            V, _ = clf_net.compute_lyapunov(x_current)

            u_sim_lqr[tstep, :, :] = u
            V_sim_lqr[tstep, :, 0] = V
            # Get the dynamics
            for i in range(N_sim):
                xdot, Nz = dynamics(x_current[i, :].unsqueeze(0), u, return_Nz=True)
                # Take one step to the future
                x_sim_lqr[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
                Nz_sim_lqr[tstep, i, :] = Nz
            t_final_lqr = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    ax1.plot([], c=sns.color_palette("pastel")[1], label="LF")
    ax1.plot([], c=sns.color_palette("pastel")[0], label="LQR")
    ax1.plot(t[:t_final_rclfqp], x_sim_rclfqp[:t_final_rclfqp, :, 4],
             c=sns.color_palette("pastel")[1])
    ax1.plot(t[:t_final_lqr], x_sim_lqr[:t_final_lqr, :, 4], c=sns.color_palette("pastel")[0])
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$alt$")
    ax1.legend()
    ax1.set_xlim([0, t_sim])

    ax2 = axs[0, 1]
    ax2.plot([], c=sns.color_palette("pastel")[0], label="LQR V")
    ax2.plot([], c=sns.color_palette("pastel")[1], label="LF V")
    ax2.plot(t[:t_final_lqr], V_sim_lqr[:t_final_lqr, :, 0], c=sns.color_palette("pastel")[0])
    ax2.plot(t[:t_final_rclfqp], V_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax2.plot(t, t * 0.0, c="k")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$V$")
    ax2.legend()
    ax2.set_xlim([0, t_sim])

    ax3 = axs[1, 0]
    ax3.plot([], c=sns.color_palette("pastel")[0], label="LQR Nz")
    ax3.plot([], c=sns.color_palette("pastel")[1], label="LF Nz")
    ax3.plot()
    ax3.plot(t[:t_final_rclfqp], Nz_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax3.plot(t[:t_final_lqr], Nz_sim_lqr[:t_final_lqr, :, 0],
             c=sns.color_palette("pastel")[0])
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$Nz$")
    ax3.legend()
    ax3.set_xlim([0, t_sim])

    ax4 = axs[1, 1]
    ax4.plot([], c=sns.color_palette("pastel")[0], label="LQR Nz demanded")
    ax4.plot([], c=sns.color_palette("pastel")[1], label="LF Nz demanded")
    ax4.plot(t[:t_final_lqr], Nz_nominal_lqr[:t_final_lqr, :, 0],
             c=sns.color_palette("pastel")[0])
    ax4.plot(t[:t_final_rclfqp], Nz_nominal_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax4.plot(t, t * 0.0, c="k")
    ax4.set_xlabel("$t$")
    ax4.set_ylabel("$Nz demanded$")
    ax4.legend()

    fig.tight_layout()
    plt.show()
