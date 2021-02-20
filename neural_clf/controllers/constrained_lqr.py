import cvxpy as cp
import numpy as np


class PVTOLSimpleMPC:
    """A controller that solves a constrained LQR problem online to navigate towards the goal,
    subject to convex state constraints, for the PVTOL simple landing task
    """

    def __init__(self, m, r, inertia, dt=0.001):
        """Initialize an MPC controller for the simple PVTOL landing task"""
        self.n_dims = 6
        self.n_controls = 2
        self.T = 500  # horizon

        # Define the optimization problem variables and parameters
        self.x = cp.Variable((self.T, self.n_dims), "x")
        self.crash_relax = cp.Variable(self.T, "relax")
        self.u = cp.Variable((self.T-1, self.n_controls), "u")
        self.x0 = cp.Parameter(self.n_dims, "x0")

        # Define the cost
        Q = 100 * np.eye(self.n_dims)
        R = 0.1 * np.eye(self.n_controls)
        g = 9.81
        u_eq = m * g / 2.0
        cost = 0.0
        for t in range(self.T):
            cost += cp.quad_form(self.x[t, :], Q)
        for t in range(self.T - 1):
            cost += cp.quad_form(self.u[t, :] - u_eq, R)

        # Define the dynamics constraints via direct transcription
        constraints = []
        A = np.zeros((self.n_dims, self.n_dims))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 2] = -g
        A *= dt

        B = np.zeros((self.n_dims, self.n_controls))
        B[4, 0] = 1.0 / m
        B[4, 1] = 1.0 / m
        B[5, 0] = r / inertia
        B[5, 1] = -r / inertia
        B *= dt

        for t in range(self.T-1):
            constraints.append(self.x[t+1, :] == self.x[t, :] + A @ self.x[t, :] + B @ self.u[t, :])

        # # Define the no-crashing constraint
        # for t in range(self.T):
        #     constraints.append(self.x[t, 1] >= -0.1 - self.crash_relax[t])
        #     constraints.append(self.crash_relax[t] >= 0)
        #     cost += self.crash_relax[t]**2 * 1e6

        # Define the initial condition constraint
        constraints.append(self.x[0, :] == self.x0)

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def step(self, x_current):
        """generate the next control input based on the current state"""
        self.x0.value = x_current
        self.problem.solve(warm_start=True)
        return self.u[0, :].value
