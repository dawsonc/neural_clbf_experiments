"""Utility functions for defining models and nominal controllers"""
import numpy as np
import scipy.linalg


def lqr(A, B, Q, R, return_eigs=False):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u

    Code from Mark Wilfred Mueller at http://www.mwm.im/lqr-controllers-with-python/

    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A-B*K)
        return K, eigVals
