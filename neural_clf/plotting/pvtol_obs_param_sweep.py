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


# Also define the safe and unsafe regions
def is_unsafe_check(x, z):
    """Return the mask of x indicating safe regions"""
    unsafe_mask = torch.zeros_like(x, dtype=torch.bool)
    # We have a floor at z=-0.1 that we need to avoid
    unsafe_z = -0.3
    floor_mask = z <= unsafe_z
    unsafe_mask.logical_or_(floor_mask)
    # We also have a block obstacle to the left at ground level
    obs1_min_x, obs1_max_x = (-1.0, -0.5)
    obs1_min_z, obs1_max_z = (-0.4, 0.5)
    obs1_min_x, obs1_max_x = (-0.9, -0.6)
    obs1_min_z, obs1_max_z = (-0.4, 0.4)
    obs1_mask_x = torch.logical_and(x >= obs1_min_x, x <= obs1_max_x)
    obs1_mask_z = torch.logical_and(z >= obs1_min_z, z <= obs1_max_z)
    obs1_mask = torch.logical_and(obs1_mask_x, obs1_mask_z)
    unsafe_mask.logical_or_(obs1_mask)
    # We also have a block obstacle to the right in the air
    obs2_min_x, obs2_max_x = (0.05, 1.0)
    obs2_min_z, obs2_max_z = (0.8, 1.35)
    obs2_min_x, obs2_max_x = (0.15, 0.9)
    obs2_min_z, obs2_max_z = (0.9, 1.25)
    obs2_mask_x = torch.logical_and(x >= obs2_min_x, x <= obs2_max_x)
    obs2_mask_z = torch.logical_and(z >= obs2_min_z, z <= obs2_max_z)
    obs2_mask = torch.logical_and(obs2_mask_x, obs2_mask_z)
    unsafe_mask.logical_or_(obs2_mask)
    return unsafe_mask


# Simulate some results
with torch.no_grad():
    N_sim = 100
    x_sim_start = torch.zeros(N_sim, n_dims)
    x_sim_start[:, 0] = -1.5
    x_sim_start[:, 1] = 0.1

    # Get a random distribution of masses and inertias
    ms = torch.Tensor(N_sim, 1).uniform_(1.00 * low_m, 1.05 * low_m)
    inertias = torch.Tensor(N_sim, 1).uniform_(1.0 * low_I, 1.05 * low_I)
    # ms = torch.linspace(1.05 * low_m, 1.05 * low_m, N_sim)
    # inertias = torch.linspace(1.0 * low_I, 1.0 * low_I, N_sim)
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
    for i in range(N_sim):
        try:
            for tstep in tqdm(range(1, num_timesteps)):
                # Get the current state
                x_current = x_sim_rclfqp[tstep - 1, i, :]
                # Get the control input at the current state
                ts = time.time()
                u, r, V, Vdot = robust_clf_net(x_current.unsqueeze(0))
                tf = time.time()
                rclbf_runtime += tf - ts
                rclbf_calls += 1
                # Get the dynamics
                f_val, g_val = control_affine_dynamics(x_current.unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Take one step to the future
                xdot = f_val + g_val @ u[0, :]
                x_sim_rclfqp[tstep, i, :] = x_current + delta_t * xdot.squeeze()

                t_final_rclfqp = tstep
        except (Exception, KeyboardInterrupt):
            print("Controller failed")
            pass

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
    for i in range(N_sim):
        try:
            for tstep in tqdm(range(1, num_timesteps)):
                # Get the current state
                x_current = x_sim_mpc[tstep - 1, i, :]

                # and measure the Lyapunov function value here
                V, grad_V = robust_clf_net.compute_lyapunov(x_current.unsqueeze(0))

                u_sim_mpc[tstep, :, :] = u
                V_sim_mpc[tstep, :, 0] = V
                # Get the dynamics
                f_val, g_val = control_affine_dynamics(x_current.unsqueeze(0),
                                                       m=ms[i],
                                                       inertia=inertias[i])
                # Get the control input at the current state if we're at the appropriate timing
                if tstep == 1 or tstep % mpc_update_frequency == 0:
                    ts = time.time()
                    u = torch.tensor(PVTOLObsMPC(x_current.numpy(), obs_pos, obs_r))
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
                x_sim_mpc[tstep, i, :] = x_current + delta_t * xdot.squeeze()
                t_final_mpc = tstep
        except (Exception, KeyboardInterrupt):
            # print("Controller failed")
            pass

    print(f"rCLBF qp total runtime = {rclbf_runtime} s ({rclbf_runtime / rclbf_calls} s per iteration)")
    print(f"MPC total runtime = {mpc_runtime} s ({mpc_runtime / mpc_calls} s per iteration)")

rclbf_failures = 0
mpc_failures = 0
rclbf_reached = 0
mpc_reached = 0
for i in range(N_sim):
    max_t_rclbf = num_timesteps
    max_t_mpc = num_timesteps
    mpc_failed = False
    if torch.any(x_sim_rclfqp[:, i, :2].norm(dim=-1) <= 0.25, dim=0):
        rclbf_reached += 1
        max_t_rclbf = torch.nonzero(x_sim_rclfqp[:, i, :2].norm(dim=-1) <= 0.25).min()
    if torch.any(is_unsafe_check(x_sim_rclfqp[:max_t_rclbf, i, 0], x_sim_rclfqp[:max_t_rclbf, i, 1])):
        rclbf_failures += 1
    if torch.any(x_sim_mpc[:, i, :2].norm(dim=-1) <= 0.25, dim=0):
        reached_idx = torch.nonzero(x_sim_mpc[:, i, :2].norm(dim=-1) <= 0.25).min()
        xnorm = x_sim_mpc[:, i, :2].norm(dim=-1)
        if torch.abs(xnorm[reached_idx-1] - xnorm[reached_idx]) <= 0.1:
            mpc_reached += 1
            max_t_mpc = reached_idx
    if torch.any(is_unsafe_check(x_sim_mpc[:max_t_mpc, i, 0], x_sim_mpc[:max_t_mpc, i, 1])):
        mpc_failures += 1
    elif np.abs(np.diff(x_sim_mpc[:, i, :2].norm(dim=-1).numpy())).max() >= 0.1:
        mpc_failures += 1

print(f"rCLBF QP safety failure rate: {rclbf_failures / N_sim}")
print(f"MPC safety failure rate: {mpc_failures / N_sim}")
print(f"rCLBF QP reach rate: {rclbf_reached / N_sim}")
print(f"MPC reach rate: {mpc_reached / N_sim}")

rclbf_goal_error, _ = x_sim_rclfqp.norm(dim=-1)[3000:, :].min()
mpc_goal_error, _ = x_sim_mpc.norm(dim=-1)[3000:, :].min()
rclbf_goal_error = rclbf_goal_error.mean()
mpc_goal_error = mpc_goal_error.mean()
print(f"rCLBF QP goal error: {rclbf_goal_error}")
print(f"MPC safety failure rate: {mpc_goal_error}")
