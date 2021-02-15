import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from neural_clf.controllers.lf_net_f16_gcas import LF_Net
from models.f16_full_gcas import (
    dynamics,
    control_affine_dynamics,
    u_nominal,
    n_controls,
    n_dims,
)
from aerobench.util import StateIndex


# Beautify plots
sns.set_theme(context="talk", style="white")

# Load the model from file
filename = "logs/f16_lf_gcas.pth.tar"
checkpoint = torch.load(filename)
clf_net = LF_Net(n_dims,
                 checkpoint['n_hidden'],
                 n_controls,
                 checkpoint['clf_lambda'],
                 checkpoint['relaxation_penalty'],
                 dynamics,
                 control_affine_dynamics,
                 u_nominal)
clf_net.load_state_dict(checkpoint['lf_net'])

# Simulate some results
with torch.no_grad():
    N_sim = 1
    # Initial conditions
    power = 9  # engine power level (0-10)
    # Default alpha & beta
    alpha = 0.035            # Trim Angle of Attack (rad)
    beta = 0                 # Side slip angle (rad)
    # Initial Attitude
    alt = 1000        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = -np.pi/8           # Roll angle from wings level (rad)
    theta = (-np.pi/2)*0.3         # Pitch angle from nose level (rad)
    psi = 0   # Yaw angle from North (rad)
    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow, 3x integrator states]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power, 0, 0, 0]
    x_sim_start = torch.zeros(N_sim, n_dims)
    for sim in range(N_sim):
        x_sim_start[sim, :] = torch.tensor(init)

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

    print("Simulating nominal controller...")
    x_sim_nominal = torch.zeros(num_timesteps, N_sim, n_dims)
    x_sim_nominal[0, :, :] = x_sim_start
    u_sim_nominal = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_nominal = torch.zeros(num_timesteps, N_sim, 1)
    Nz_sim_nominal = torch.zeros(num_timesteps, N_sim, 1)
    Nz_nominal_nominal = torch.zeros(num_timesteps, N_sim, 1)
    t_final_nominal = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim_nominal[tstep - 1, :, :]
            # Get the control input at the current state
            u = u_nominal(x_current)
            # and measure the Lyapunov function value here
            # V, _ = clf_net.compute_lyapunov(x_current)

            u_sim_nominal[tstep, :, :] = u
            # V_sim_nominal[tstep, :, 0] = V
            # Get the dynamics
            for i in range(N_sim):
                xdot, Nz = dynamics(x_current[i, :].unsqueeze(0), u, return_Nz=True)
                # Take one step to the future
                x_sim_nominal[tstep, i, :] = x_current[i, :] + delta_t * xdot.squeeze()
                Nz_sim_nominal[tstep, i, :] = Nz
            t_final_nominal = tstep
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    fig, axs = plt.subplots(2, 2)
    t = np.linspace(0, t_sim, num_timesteps)
    ax1 = axs[0, 0]
    ax1.plot([], c=sns.color_palette("pastel")[1], label="LF")
    ax1.plot([], c=sns.color_palette("pastel")[0], label="Nominal")
    ax1.plot(t[:t_final_rclfqp], x_sim_rclfqp[:t_final_rclfqp, :, StateIndex.ALT],
             c=sns.color_palette("pastel")[1])
    ax1.plot(t[:t_final_nominal], x_sim_nominal[:t_final_nominal, :, StateIndex.ALT],
             c=sns.color_palette("pastel")[0])
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$alt$")
    ax1.legend()
    ax1.set_xlim([0, t_sim])

    ax2 = axs[0, 1]
    ax2.plot([], c=sns.color_palette("pastel")[0], label="Nominal V")
    ax2.plot([], c=sns.color_palette("pastel")[1], label="LF V")
    ax2.plot(t[:t_final_nominal], V_sim_nominal[:t_final_nominal, :, 0], c=sns.color_palette("pastel")[0])
    ax2.plot(t[:t_final_rclfqp], V_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax2.plot(t, t * 0.0, c="k")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$V$")
    ax2.legend()
    ax2.set_xlim([0, t_sim])

    ax3 = axs[1, 0]
    ax3.plot([], c=sns.color_palette("pastel")[0], label="Nominal Nz")
    ax3.plot([], c=sns.color_palette("pastel")[1], label="LF Nz")
    ax3.plot()
    ax3.plot(t[:t_final_rclfqp], Nz_sim_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax3.plot(t[:t_final_nominal], Nz_sim_nominal[:t_final_nominal, :, 0],
             c=sns.color_palette("pastel")[0])
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$Nz$")
    ax3.legend()
    ax3.set_xlim([0, t_sim])

    ax4 = axs[1, 1]
    ax4.plot([], c=sns.color_palette("pastel")[0], label="Nominal Nz demanded")
    ax4.plot([], c=sns.color_palette("pastel")[1], label="LF Nz demanded")
    ax4.plot(t[:t_final_nominal], Nz_nominal_nominal[:t_final_nominal, :, 0],
             c=sns.color_palette("pastel")[0])
    ax4.plot(t[:t_final_rclfqp], Nz_nominal_rclfqp[:t_final_rclfqp, :, 0],
             c=sns.color_palette("pastel")[1])
    ax4.plot(t, t * 0.0, c="k")
    ax4.set_xlabel("$t$")
    ax4.set_ylabel("$Nz$ demanded")
    ax4.legend()

    fig.tight_layout()
    plt.show()
