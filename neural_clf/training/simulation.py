import numpy as np
import torch
from tqdm import tqdm


def simulate_rollout(x_init, net, control_affine_dynamics, timestep, rollout_length, scenario):
    """
    Simulate the execution of the given network controller from the given start points.
    Runs until a maximum time is reached

    args:
        x_init: N_rollouts x n_dims torch tensor of initial conditions
        net: a CLF net controller
        control_affine_dynamics: a function that takes n_batch x n_dims and returns a tuple of:
            f_func: a function n_batch x n_dims -> n_batch x n_dims that returns the
                    state-dependent part of the control-affine dynamics
            g_func: a function n_batch x n_dims -> n_batch x n_dims x n_controls that returns
                    the input coefficient matrix for the control-affine dynamics
        timestep: the timestep at which to simulate
        rollout_length: the amount of time to simulate for
        nominal_scenari: a dictionary specifying the parameters to pass to control_affine_dynamics
    """
    num_timesteps = int(rollout_length // timestep)
    N_sim = x_init.shape[0]
    n_dims = x_init.shape[1]

    x_rollout = torch.zeros(num_timesteps, N_sim, n_dims)
    x_rollout[0, :, :] = x_init
    print("Conducting rollout...")
    for tstep in tqdm(range(1, num_timesteps)):
        # Get the current state
        x_current = x_rollout[tstep - 1, :, :]
        # Get the control input at the current state
        u, r, V, Vdot = net(x_current)

        # Get the dynamics
        f_val, g_val = control_affine_dynamics(x_current, **scenario)

        # Take one step to the future
        xdot = f_val + torch.bmm(g_val, u.unsqueeze(-1)).squeeze()
        x_rollout[tstep, :, :] = x_current + timestep * xdot

    return x_rollout
