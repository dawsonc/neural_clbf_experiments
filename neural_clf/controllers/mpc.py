"""
Implement nonlinear MPC-based controllers based on code from Yue
Original at https://gist.github.com/mengyuest/a0308c12977d2349c5e1f296f700210f
"""
import numpy as np
import casadi


def PVTOLObsMPC(x_current, obs_pos, obs_r):
    """
    Nonlinear MPC for navigating a quadrotor around obstacles. Needs circular representation
    of obstacles

    args:
        x_current - (6,) np state vector
        obs_pos - (n_obs, 2) np array of obstacle positions
        obs_r - (n_obs,) np array of obstacle radii
    """
    # parameter settings
    N_obs = obs_pos.shape[0]
    assert obs_r.shape[0] == N_obs
    T = 30  # planning horizon
    dt = 1/20.0

    # nominal model params
    g = 9.81
    mass = 1.0
    inertia = 0.01
    r = 0.25  # lever arm

    # Create opt problem and decision variables
    opti = casadi.Opti()
    x = opti.variable(T + 1, 6)  # state (x, z, theta, vx, vz, theta_dot)
    u = opti.variable(T, 2)      # control (u1, u2)
    gamma = opti.variable(T, N_obs)  # collision relaxation

    # Simple objective: actuator cost (relative to equilibrium) and terminal state cost
    opti.minimize(
        casadi.sumsqr(x[:, :2]) + casadi.sumsqr(u - mass * g / 2.0) + 100 * casadi.sumsqr(gamma)
    )

    # initial condition
    opti.subject_to(x[0, 0] == x_current[0])
    opti.subject_to(x[0, 1] == x_current[1])
    opti.subject_to(x[0, 2] == x_current[2])
    opti.subject_to(x[0, 3] == x_current[3])
    opti.subject_to(x[0, 4] == x_current[4])
    opti.subject_to(x[0, 5] == x_current[5])

    # No-crashing-into-floor constraint (yes that's the technical term)
    opti.subject_to(x[:, 1] >= -0.3)

    # dynamics
    for k in range(T):  # timesteps
        # xdot = vx
        opti.subject_to(x[k + 1, 0] == x[k, 0] + x[k, 3] * dt)
        # zdot = vz
        opti.subject_to(x[k + 1, 1] == x[k, 1] + x[k, 4] * dt)
        # thetadot = theta_dot
        opti.subject_to(x[k + 1, 2] == x[k, 2] + x[k, 5] * dt)
        # vx_dot = -(u1+u2)*sin(theta)/mass
        opti.subject_to(
            x[k + 1, 3] == x[k, 3] - (u[k, 0] + u[k, 1]) * casadi.sin(x[k, 2]) * dt / mass)
        # vz_dot = (u1+u2)*cos(theta)/mass - g
        opti.subject_to(
            x[k + 1, 4] == x[k, 4] + (u[k, 0] + u[k, 1]) * casadi.cos(x[k, 2]) * dt / mass - g * dt)
        # theta_ddot = (u1 - u2) * r/I
        opti.subject_to(x[k + 1, 5] == x[k, 5] + (u[k, 0] - u[k, 1]) * r / inertia * dt)

    # collision avoidance constraints
    for k in range(T):  # timesteps
        for i in range(N_obs):  # obstacles
            opti.subject_to(
                ((x[k + 1, 0] - obs_pos[i, 0]) / obs_r[i]) ** 2 +
                ((x[k + 1, 1] - obs_pos[i, 1]) / obs_r[i]) ** 2 + gamma[k, i] >= 1)

    opti.set_initial(x, np.random.rand(T+1, 6))
    opti.set_initial(u, np.random.rand(T, 2))

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

    # Return the first control input
    return sol1.value(u[0, :])
