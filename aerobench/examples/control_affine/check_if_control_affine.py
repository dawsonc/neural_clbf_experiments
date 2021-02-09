import numpy as np
from numpy import deg2rad

import sys
sys.path.insert(0, './code')

from aerobench.lowlevel.subf16_model import subf16_model
from aerobench.lowlevel.low_level_controller import CtrlLimits


# Check if the F16 model is control-affine

#         x[0] = air speed, VT    (ft/sec)
#         x[1] = angle of attack, alpha  (rad)
#         x[2] = angle of sideslip, beta (rad)
#         x[3] = roll angle, phi  (rad)
#         x[4] = pitch angle, theta  (rad)
#         x[5] = yaw angle, psi  (rad)
#         x[6] = roll rate, P  (rad/sec)
#         x[7] = pitch rate, Q  (rad/sec)
#         x[8] = yaw rate, R  (rad/sec)
#         x[9] = northward horizontal displacement, pn  (feet)
#         x[10] = eastward horizontal displacement, pe  (feet)
#         x[11] = altitude, h  (feet)
#         x[12] = engine thrust dynamics lag state, pow
#
#         u[0] = throttle command  0.0 < u(1) < 1.0
#         u[1] = elevator command in degrees
#         u[2] = aileron command in degrees
#         u[3] = rudder command in degrees

# Set initial conditions (taken from straight_and_level example)
power = 7.6             # engine power level (0-10)

# Default alpha & beta
alpha = deg2rad(1.8)    # Trim Angle of Attack (rad)
beta = 0                # Side slip angle (rad)

# Initial Attitude
alt = 3600              # altitude (ft)
vt = 500                # initial velocity (ft/sec)
phi = 0                 # Roll angle from wings level (rad)
theta = 0.03            # Pitch angle from nose level (rad)
psi = 0                 # Yaw angle from North (rad)

# Build Initial Condition Vectors
#    x = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
x_init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

# Our strategy for system identification: hold state constant, then get derivatives with a set of
# random control vectors. Then run a regression on the control vectors (augmented with a 1) to
# see how well the fit is

# Generate random inputs
u_lim = CtrlLimits()
N_samples = 1000
N_controls = 4
u = np.random.uniform(low=0.0, high=1.0, size=(N_samples, N_controls))
u[:, 0] = u[:, 0] * (u_lim.ThrottleMax - u_lim.ThrottleMin) + u_lim.ThrottleMin
u[:, 1] = u[:, 1] * (u_lim.ElevatorMaxDeg - u_lim.ElevatorMinDeg) + u_lim.ElevatorMinDeg
u[:, 2] = u[:, 2] * (u_lim.AileronMaxDeg - u_lim.AileronMinDeg) + u_lim.AileronMinDeg
u[:, 3] = u[:, 3] * (u_lim.RudderMaxDeg - u_lim.RudderMinDeg) + u_lim.RudderMinDeg

# get state derivatives for these inputs, using a specified model
# model = "stevens"  # look-up table
model = "morelli"  # polynomial fit
xdot = np.zeros((N_samples, len(x_init)))
for i in range(N_samples):
    xdot[i, :], _, _, _, _ = subf16_model(x_init, u[i, :], model)

# We want to look for a relationship of the form xdot = f(x) + g(x)*u, or xdot = [f, g]*[1, u]
# Augment the inputs with a one column for the control-independent part
regressors = np.hstack((np.ones((N_samples, 1)), u))
# Compute the least-squares fit and find A such that xdot^T = A phi^T, or xdot = phi A^T
A, residuals, _, _ = np.linalg.lstsq(regressors, xdot, rcond=None)
A = A.T
