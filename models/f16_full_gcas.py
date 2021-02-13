import torch

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.lowlevel.low_level_controller import LowLevelController


# AeroBench dynamics for the F16 fighter jet
n_dims = 16
n_controls = 4


def dynamics(x, u, return_Nz=False):
    """
    Return the dynamics evaluated at the given state and control input

    Returns a tuple of xdot and Nz (true vertical acceleration) if return_Nz is True
    """
    # The f16 model is not batched, so we need to compute f and g for each x separately
    n_batch = x.size()[0]
    xdot = torch.zeros_like(x)
    Nz = torch.zeros((n_batch, 1))

    # Convert input to numpy
    x = x.cpu().detach().numpy()
    u = u.cpu().detach().numpy()
    for batch in range(n_batch):
        # Compute derivatives at this point points
        llc = LowLevelController()
        model = "stevens"  # look-up table
        # model = "morelli"  # polynomial fit
        t = 0.0
        xdot_i, _, Nz_i, _, _ = controlled_f16(t, x[batch, :], u[batch, :],
                                               llc, f16_model=model)
        xdot[batch, :] = torch.tensor(xdot_i)

        # Save the true z acceleration
        Nz[batch, 0] = Nz_i

    if return_Nz:
        return xdot, Nz
    else:
        return xdot


def u_nominal(x, alt_setpoint=500, vt_setpoint=500):
    """
    Return the nominal controller for the system at state x
    """
    gcas = GcasAutopilot()

    # The autopilot is not meant to be run on batches so we need to get control inputs separately
    n_batch = x.size()[0]
    x = x.cpu().detach().numpy()
    u = torch.zeros((n_batch, n_controls))

    for batch in range(n_batch):
        # The GCAS autopilot is implemented as a state machine that first rolls and then pulls up.
        # Here we unwrap the state machine logic to get a simpler mapping from state to control

        # If the plane is not hurtling towards the ground, no need to do anything.
        if gcas.is_nose_high_enough(x[batch, :]) or gcas.is_above_flight_deck(x[batch, :]):
            continue

        # If we are hurtling towards the ground and the plane isn't level, we need to try to
        # roll to get level
        if not gcas.is_roll_rate_low(x[batch, :]) or not gcas.are_wings_level(x[batch, :]):
            u[batch, :] = torch.tensor(gcas.roll_wings_level(x[batch, :]))
            continue

        # If we are hurtling towards the ground and the plane IS level, then we need to pull up
        u[batch, :] = torch.tensor(gcas.pull_nose_level())

    return u
