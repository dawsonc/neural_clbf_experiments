import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class CLF_QP_Net(nn.Module):
    """A neural network for simultaneously computing the Lyapunov function and the control input.
    The neural net makes the Lyapunov function, and the control input is computed by solving a QP.
    """

    def __init__(self, n_input, n_hidden, n_controls, clf_lambda, relaxation_penalty,
                 f_func, g_func, u_nominal, scenarios, nominal_scenario,
                 G_u=torch.tensor([]), h_u=torch.tensor([]), allow_relax=True):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            clf_lambda: desired exponential convergence rate
            relaxation_penalty: the penalty for relaxing the control lyapunov constraint
            f_func: a function n_batch x n_dims -> n_batch x n_dims that returns the state-dependent
                    part of the control-affine dynamics
            g_func: a function n_batch x n_dims -> n_batch x n_dims x n_controls that returns the
                    input coefficient matrix for the control-affine dynamics
            u_nominal: a function n_batch x n_dims -> n_batch x n_controls that returns the nominal
                       control input for the system (even LQR about origin is fine)
            scenarios: a list of dictionaries specifying the parameters to pass to f_func and g_func
            nominal_scenario: a dictionary specifying the parameters to pass to u_nominal
            G_u: a matrix of constraints on fesaible control inputs (n_constraints x n_controls)
            h_u: a vector of constraints on fesaible control inputs (n_constraints x 1)
                Given G_u and h_u, the CLF QP will additionally enforce G_u u <= h_u
        """
        super(CLF_QP_Net, self).__init__()

        # Save the dynamics and nominal controller functions
        self.f = f_func
        self.g = g_func
        self.u_nominal = u_nominal
        assert len(scenarios) > 0, "Must pass at least one scenario"
        self.scenarios = scenarios
        self.nominal_scenario = nominal_scenario

        # The network will have the following architecture
        #
        # n_input -> VFC1 (n_input x n_hidden) -> VFC2 (n_hidden, n_hidden)
        # -> VFC2 (n_hidden, n_hidden) -> V = x^T x --> QP -> u
        self.Vfc_layer_1 = nn.Linear(n_input, n_hidden)
        self.Vfc_layer_2 = nn.Linear(n_hidden, n_hidden)

        self.n_controls = n_controls
        self.clf_lambda = clf_lambda
        self.relaxation_penalty = relaxation_penalty
        assert G_u.size(0) == h_u.size(0), "G_u and h_u must have consistent dimensions"
        self.G_u = G_u.double()
        self.h_u = h_u.double()

        # To find the control input, we want to solve a QP, so we need to define the QP layer here
        # The decision variables are the control input and relaxation of the CLF condition for each
        # scenario
        u = cp.Variable(self.n_controls)
        relaxations = []
        for scenario in self.scenarios:
            relaxations.append(cp.Variable(1))
        # And it's parameterized by the lie derivatives and value of the Lyapunov function in each
        # scenario
        L_fs = []
        L_gs = []
        for scenario in self.scenarios:
            L_fs.append(cp.Parameter(1))
            L_gs.append(cp.Parameter(self.n_controls))

        V = cp.Parameter(1, nonneg=True)
        # To allow for gradual increasing of the cost of relaxations, set the relaxation penalty
        # as a parameter as well
        relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        # Also allow passing in a nominal controller to filter
        u_nominal = cp.Parameter(self.n_controls)

        # The QP is constrained by
        #
        # L_f V + L_g V u + lambda V <= 0
        #
        # To ensure that this QP is always feasible, we relax the constraint
        #
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        #
        # and later add the cost term relaxation_penalty * r.
        #
        # To encourage robustness to parameter variation, we have four instances of these
        # constraints for different parameters
        constraints = []
        if allow_relax:
            constraints = []
            for i in range(len(self.scenarios)):
                constraints.append(
                    L_fs[i] + L_gs[i] @ u + self.clf_lambda * V - relaxations[i] <= 0)
                constraints.append(relaxations[i] >= 0)
        else:
            for i in range(len(self.scenarios)):
                constraints.append(
                    L_fs[i] + L_gs[i] @ u + self.clf_lambda * V <= 0)
        # We also add the user-supplied constraints, if provided
        if len(self.G_u) > 0:
            constraints.append(self.G_u @ u <= self.h_u)

        # The cost is quadratic in the controls and linear in the relaxation
        objective_expression = cp.sum_squares(u - u_nominal)
        for r in relaxations:
            objective_expression += cp.multiply(relaxation_penalty_param, cp.square(r))
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem and the layer based on that
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + relaxations
        parameters = L_fs + L_gs + [V, u_nominal, relaxation_penalty_param]
        self.qp_layer = CvxpyLayer(problem, variables=variables, parameters=parameters)

    def forward(self, x):
        """
        Compute the forward pass of the controller

        args:
            x: the state at the current timestep [n_batch, 2]
        returns:
            u: the input at the current state [n_batch, 1]
            r: the relaxation required to satisfy the CLF inequality
            V: the value of the Lyapunov function at a given point
            Vdot: the time derivative of the Lyapunov function
        """
        # Use the first two layers to compute the Lyapunov function
        tanh = nn.Tanh()
        Vfc1_act = tanh(self.Vfc_layer_1(x))
        Vfc2_act = tanh(self.Vfc_layer_2(Vfc1_act))
        V = 0.5 * (Vfc2_act * Vfc2_act).sum(1)

        # We also need to calculate the Lie derivative of V along f and g
        #
        # L_f V = \grad V * f
        # L_g V = \grad V * g
        #
        # Since V = tanh(w2 * tanh(w1*x + b1) + b1),
        # grad V = d_tanh_dx(V) * w2 * d_tanh_dx(tanh(w1*x + b1)) * w1
        def d_tanh_dx(tanh):
            return torch.diag_embed(1 - tanh**2)

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        DVfc1_act = torch.matmul(d_tanh_dx(Vfc1_act), self.Vfc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        DVfc2_act = torch.bmm(torch.matmul(d_tanh_dx(Vfc2_act), self.Vfc_layer_2.weight), DVfc1_act)
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(Vfc2_act.unsqueeze(1), DVfc2_act)

        # Compute lie derivatives for each scenario
        L_fs = []
        L_gs = []
        for scenario in self.scenarios:
            L_fs.append(torch.bmm(grad_V, self.f(x, **scenario).unsqueeze(-1)).squeeze(-1))
            L_gs.append(torch.bmm(grad_V, self.g(x, **scenario)).squeeze(1))

        # Also get the nominal control input
        u_nominal = self.u_nominal(x, **self.nominal_scenario)

        # To find the control input, we need to solve a QP
        result = self.qp_layer(
            *L_fs, *L_gs,
            V.unsqueeze(-1),
            u_nominal,
            torch.tensor([self.relaxation_penalty]),
            solver_args={"max_iters": 50000})
        u = result[0]
        rs = result[1:]

        # Average across scenarios
        n_scenarios = len(self.scenarios)
        Vdot = L_fs[0].unsqueeze(-1) + torch.bmm(L_gs[0].unsqueeze(1), u.unsqueeze(-1))
        relaxation = rs[0]
        for i in range(1, n_scenarios):
            Vdot += L_fs[i].unsqueeze(-1) + torch.bmm(L_gs[i].unsqueeze(1), u.unsqueeze(-1))
            relaxation += rs[i]

        return u, relaxation, V, Vdot
