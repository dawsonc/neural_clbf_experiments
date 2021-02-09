import torch
import numpy as np

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.lowlevel.low_level_controller import LowLevelController


# AeroBench dynamics for the F16 fighter jet
n_dims = 16
n_controls = 4


def control_affine_dynamics(x):
    """
    Return the control-affine dynamics evaluated at the given state
    """
    pass


def u_nominal(x):
    """
    Return the nominal controller for the system at state x, given by LQR
    """
    pass
