import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

from neural_clf.controllers.clf_qp_net import CLF_QP_Net
from neural_clf.controllers.mpc import Quad9dHoverMPC
from models.quad9d import (
    control_affine_dynamics,
    u_nominal,
    n_controls,
    n_dims,
    StateIndex,
    m_low,
    m_high,
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
filename = "logs/quad9d_robust_clf_qp.pth.tar"
checkpoint = torch.load(filename)
nominal_scenario = {"m": m_low}
scenarios = [
    {"m": m_low},
    {"m": m_high},
]

robust_clf_net = CLF_QP_Net(n_dims,
                            checkpoint['n_hidden'],
                            n_controls,
                            1.0,  # checkpoint['clf_lambda'],
                            1000.0,  # checkpoint['relaxation_penalty'],
                            control_affine_dynamics,
                            u_nominal,
                            scenarios,
                            nominal_scenario,
                            use_casadi=True)
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
    x_sim_start = torch.zeros(N_sim, n_dims) + 1.0
    x_sim_start[:, StateIndex.PZ] = -1.0

    t_sim = 10
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(m_low, m_high)
    # ms = torch.linspace(m_low, m_high, N_sim)

    print("Simulating robust CLBF QP controller...")
    x_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, n_dims)
    u_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, n_controls)
    V_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    Vdot_sim_rclbfqp = torch.zeros(num_timesteps, N_sim, 1)
    x_sim_rclbfqp[0, :, :] = x_sim_start
    t_final_rclbfqp = 0
    rclbf_runtime = 0.0
    rclbf_calls = 0.0
    for i in range(N_sim):
        try:
            for tstep in tqdm(range(1, num_timesteps)):
                # Get the current state
                x_current = x_sim_rclbfqp[tstep - 1, i, :]
                # Get the control input at the current state
                ts = time.time()
                u, r, V, Vdot = robust_clf_net(x_current.unsqueeze(0))
                tf = time.time()
                rclbf_runtime += tf - ts
                rclbf_calls += 1

                # u_sim_rclbfqp[tstep, :, :] = u
                # V_sim_rclbfqp[tstep, :, 0] = V
                # Vdot_sim_rclbfqp[tstep, :, 0] = Vdot.squeeze()
                # Get the dynamics
                f_val, g_val = control_affine_dynamics(x_current.unsqueeze(0), m=ms[i])
                # Take one step to the future
                xdot = f_val + (g_val @ u.T).squeeze()
                x_sim_rclbfqp[tstep, i, :] = x_current + delta_t * xdot.squeeze()

                t_final_rclbfqp = tstep
        except (Exception, KeyboardInterrupt):
            print("Controller failed")

    print("Simulating MPC controller...")
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
                f_val, g_val = control_affine_dynamics(x_current[i, :].unsqueeze(0), m=ms[i])

                # Get the control input at the current state if we're at the appropriate timing
                if tstep == 1 or tstep % mpc_update_frequency == 0:
                    ts = time.time()
                    u = torch.tensor(Quad9dHoverMPC(x_current[i, :].numpy()))
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
    except (Exception, KeyboardInterrupt):
        print("Controller failed")

    print(f"rCLBF qp total runtime = {rclbf_runtime} s ({rclbf_runtime / rclbf_calls} s per iteration)")
    print(f"MPC total runtime = {mpc_runtime} s ({mpc_runtime / mpc_calls} s per iteration)")

    rclbf_failures = 0
    mpc_failures = 0
    for i in range(N_sim):
        if torch.any(x_sim_rclbfqp[:, i, StateIndex.PZ] >= checkpoint["unsafe_z"]):
            rclbf_failures += 1
        if torch.any(x_sim_mpc[:, i, StateIndex.PZ] >= checkpoint["unsafe_z"]):
            mpc_failures += 1
    print(f"rCLBF QP safety failure rate: {rclbf_failures / N_sim}")
    print(f"MPC safety failure rate: {mpc_failures / N_sim}")

rclbf_goal_error, _ = x_sim_rclbfqp[:, :, StateIndex.PZ].abs()[7500:, :].min(dim=0)
mpc_goal_error, _ = x_sim_mpc[:, :, StateIndex.PZ].abs()[7500:, :].min(dim=0)
rclbf_goal_error = rclbf_goal_error.mean()
mpc_goal_error = mpc_goal_error.mean()
print(f"rCLBF QP goal error: {rclbf_goal_error}")
print(f"MPC goal error: {mpc_goal_error}")
