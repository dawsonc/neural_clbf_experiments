"""
Implement nonlinear MPC-based controllers based on code from Yue
Original at https://gist.github.com/mengyuest/a0308c12977d2349c5e1f296f700210f
"""
import numpy as np
import torch
import casadi
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


sns.set_theme(context="talk", style="white")


# Define barrier function
def B_func(x):
    pz = x[1]
    px = x[0]

    B = 1.4 - torch.exp(-pz) - torch.exp(-((px + 0.75)**2 + (pz - 0.4)**2) / 0.3)
    grad_B = torch.zeros_like(x)
    grad_B[0] = 20/3 * (px + 0.75) * torch.exp(-((px + 0.75)**2 + (pz - 0.4)**2) / 0.3)
    grad_B[1] = torch.exp(-pz) + 20/3 * (pz - 0.4) * torch.exp(-((px + 0.75)**2 + (pz - 0.4)**2) / 0.3)
    return B, grad_B


# CLF
def V_func(x):
    pz = x[1]
    px = x[0]
    V = 0.5 * (px**2 + pz**2)
    grad_V = torch.zeros_like(x)
    grad_V[0] = px
    grad_V[1] = pz

    return V, grad_V


grav = 9.81
mass = 1.0
inertia = 0.01
r = 0.25  # lever arm


# Super simple single integrator dynamics
def f_func(x):
    f = torch.zeros_like(x)
    return f


def g_func(x):
    n_dims = 2
    n_controls = 2
    g = torch.zeros(n_dims, n_controls, dtype=x.dtype)
    g[0, 0] = 1.0
    g[1, 1] = 1.0

    return g


def cbf_clf_qp_linear_pvtol(x):
    x = torch.tensor(x)
    # Get B, V and gradients
    B, grad_B = B_func(x)
    V, grad_V = V_func(x)
    B = B.numpy()
    grad_B = grad_B.numpy()
    V = V.numpy()
    grad_V = grad_V.numpy()

    # set up qp
    opti = casadi.Opti()
    u = opti.variable(2)      # control (u1, u2)
    gamma = opti.variable(1)

    # Simple objective: actuator cost (relative to equilibrium) and terminal state cost
    opti.minimize(
        casadi.sumsqr(u) + 100 * casadi.sumsqr(gamma)
    )

    # Get clf cbf lie derivatives
    f = f_func(x).numpy()
    g = g_func(x).numpy()

    x = x.numpy()
    L_f_V = grad_V @ f
    L_g_V = grad_V @ g
    L_f_B = grad_B @ f
    L_g_B = grad_B @ g

    # Constrain QP
    clf_lambda = 1
    cbf_lambda = 10
    # import pdb; pdb.set_trace()
    opti.subject_to(L_f_V + L_g_V.reshape((1, 2)) @ u + clf_lambda * V <= gamma)
    opti.subject_to(L_f_B + L_g_B.reshape((1, 2)) @ u + cbf_lambda * B >= 0.0)

    # optimizer setting
    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    quiet = True
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)
    sol1 = opti.solve()

    # if B <= 0:
    #     import pdb; pdb.set_trace()

    return sol1.value(u)


if __name__ == '__main__':
    x_sim_start = torch.zeros(1, 2)
    x_sim_start[:, 0] = -1.5
    x_sim_start[:, 1] = 0.1

    t_sim = 5
    # t_sim = 0.1
    delta_t = 0.001
    num_timesteps = int(t_sim // delta_t)

    x_sim = torch.zeros(num_timesteps, 1, 2)
    x_sim[0, 0, :] = x_sim_start
    t_final = 0
    try:
        for tstep in tqdm(range(1, num_timesteps)):
            # Get the current state
            x_current = x_sim[tstep - 1, :, :]
            # Get the control input at the current state
            u = cbf_clf_qp_linear_pvtol(x_current[0, :])

            # Get the dynamics
            f_val, g_val = f_func(x_current[0, :]), g_func(x_current[0, :])
            # Take one step to the future if we're not done
            xdot = f_val + g_val @ u
            x_sim[tstep, 0, :] = x_current[0, :] + delta_t * xdot.squeeze()
            t_final = tstep
    except (Exception, KeyboardInterrupt):
        raise
        print("Controller failed")

    mpc_color = sns.color_palette("pastel")[2]
    obs_color = sns.color_palette("pastel")[3]

    t = np.linspace(0, t_sim, num_timesteps)
    fig, axs = plt.subplots(1, 1)
    ax1 = axs
    ax1.plot([], c=mpc_color, label="CLF-CBF-QP", linewidth=10.0)
    ax1.plot(x_sim[:t_final, :, 0], x_sim[:t_final, :, 1], c=mpc_color, linewidth=10.0)
    ax1.plot(x_sim_start[0, 0], x_sim_start[0, 1], 'kx', label="$x(0)$", markersize=15)
    ax1.plot(0.0, 0.0, 'ko', label="Goal", markersize=15)

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

    # plot CBF
    n_grid = 100
    x = torch.linspace(-4, 4, n_grid)
    z = torch.linspace(-4, 4, n_grid)
    grid_x, grid_z = torch.meshgrid(x, z)
    B_values = torch.zeros(n_grid, n_grid)
    print("Plotting V on grid...")
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            # Get the residual from running the model
            q = torch.zeros(2)
            # q = torch.tensor([[0.0129, -0.1149, -0.0083,  0.0534, -2.0552,  0.0201]])
            q[0] = x[i]
            q[1] = z[j]
            B_values[j, i], _ = B_func(q)

    ax1.plot([], [], c="blue", label="B(x) = 0")
    contours = ax1.contour(x, z, B_values, colors=["blue"], levels=[0.0], linewidths=[3.0])

    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$z$")
    ax1.legend(fontsize=25)
    ax1.set_xlim([-2.0, 1.0])
    ax1.set_ylim([-0.5, 1.5])

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(25)

    plt.show()
