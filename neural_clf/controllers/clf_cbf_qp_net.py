import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def d_tanh_dx(tanh):
    return torch.diag_embed(1 - tanh**2)


class CLF_CBF_QP_Net(nn.Module):
    """A neural network for simultaneously computing the Lyapunov and barrier functions and the
    control input. The neural net makes the Lyapunov and barrier functions, and the control input
    is computed by solving a QP.
    """

    def __init__(self, n_input, n_hidden, n_controls, clf_lambda, cbf_lambda,
                 clf_relaxation_penalty, cbf_relaxation_penalty,
                 f_func, g_func, u_nominal, scenarios, nominal_scenario,
                 G_u=torch.tensor([]), h_u=torch.tensor([]),
                 allow_cbf_relax=True):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            clf_lambda: desired exponential convergence rate for the CLF
            cbf_lambda: desired exponential convergence rate for the CBF
            clf_relaxation_penalty: the penalty for relaxing the control lyapunov constraint
            cbf_relaxation_penalty: the penalty for relaxing the control barrier constraint
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
            allow_cbf_relax: set to True to allow the QP to relax the CBF constraint.
        """
        super(CLF_CBF_QP_Net, self).__init__()

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

        # The CBF network has a much simpler architecture, with only fully connected layers
        self.Hfc_layer_1 = nn.Linear(n_input, n_hidden)
        self.Hfc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.Hfc_layer_3 = nn.Linear(n_hidden, n_hidden)
        self.Hfc_layer_4 = nn.Linear(n_hidden, 1)

        self.n_controls = n_controls
        self.clf_lambda = clf_lambda
        self.cbf_lambda = cbf_lambda
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.cbf_relaxation_penalty = cbf_relaxation_penalty
        assert G_u.size(0) == h_u.size(0), "G_u and h_u must have consistent dimensions"
        self.G_u = G_u.double()
        self.h_u = h_u.double()

        # To find the control input, we want to solve a QP, so we need to define the QP layer here
        # The decision variables are the control input and relaxation of the CLF condition for each
        # scenario
        u = cp.Variable(self.n_controls)
        clf_relaxations = []
        cbf_relaxations = []
        for scenario in self.scenarios:
            clf_relaxations.append(cp.Variable(1))
            cbf_relaxations.append(cp.Variable(1))
        # And it's parameterized by the lie derivatives and value of the Lyapunov function in each
        # scenario
        L_f_Vs = []
        L_g_Vs = []
        for scenario in self.scenarios:
            L_f_Vs.append(cp.Parameter(1))
            L_g_Vs.append(cp.Parameter(self.n_controls))

        V = cp.Parameter(1, nonneg=True)

        # The QP is also parameterized by the lie derivatives and value of the barrier function
        # in each scenario
        L_f_Hs = []
        L_g_Hs = []
        for scenario in self.scenarios:
            L_f_Hs.append(cp.Parameter(1))
            L_g_Hs.append(cp.Parameter(self.n_controls))

        H = cp.Parameter(1)

        # To allow for gradual increasing of the cost of relaxations, set the relaxation penalty
        # as a parameter as well
        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        cbf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        # Also allow passing in a nominal controller to filter
        u_nominal = cp.Parameter(self.n_controls)

        # The QP is constrained by
        #
        # L_f V + L_g V u + lambda V <= 0
        # L_f H + L_g G u + lambda H >= 0
        #
        # To ensure that this QP is always feasible, we relax the CLF constraint
        #
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        #
        # and later add the cost term relaxation_penalty * r.
        #
        # To encourage robustness to parameter variation, we have four instances of these
        # constraints for different parameters
        constraints = []
        for i in range(len(self.scenarios)):
            if allow_cbf_relax:
                constraints.append(
                    L_f_Hs[i] + L_g_Hs[i] @ u + self.cbf_lambda * H + cbf_relaxations[i] >= 0)
            else:
                constraints.append(
                    L_f_Hs[i] + L_g_Hs[i] @ u + self.cbf_lambda * H >= 0)
            constraints.append(
                L_f_Vs[i] + L_g_Vs[i] @ u + self.clf_lambda * V - clf_relaxations[i] <= 0)
            constraints.append(clf_relaxations[i] >= 0)
            constraints.append(cbf_relaxations[i] >= 0)
        # We also add the user-supplied constraints, if provided
        if len(self.G_u) > 0:
            constraints.append(self.G_u @ u <= self.h_u)

        # The cost is quadratic in the controls and linear in the relaxation
        # objective_expression = cp.sum_squares(u - u_nominal)
        objective_expression = cp.sum_squares(u - u_nominal)
        for r in clf_relaxations:
            objective_expression += cp.multiply(clf_relaxation_penalty_param, r)
        for r in cbf_relaxations:
            objective_expression += cp.multiply(cbf_relaxation_penalty_param, r)
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem and the layer based on that
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + clf_relaxations + cbf_relaxations
        parameters = L_f_Vs + L_g_Vs + L_f_Hs + L_g_Hs + [V, H,
                                                          u_nominal,
                                                          clf_relaxation_penalty_param,
                                                          cbf_relaxation_penalty_param]
        self.qp_layer = CvxpyLayer(problem, variables=variables, parameters=parameters)

    def compute_barrier(self, x):
        """
        Computes the value and gradient of the barrier function

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            H: the value of the barrier at each provided point x [n_batch, 1]
            grad_H: the gradient of H [n_batch, n_dims]
        """
        tanh = nn.Tanh()
        Hfc1_act = tanh(self.Hfc_layer_1(x))
        Hfc2_act = tanh(self.Hfc_layer_2(Hfc1_act))
        Hfc3_act = tanh(self.Hfc_layer_3(Hfc2_act))
        H = self.Hfc_layer_4(Hfc3_act)

        # ...and the Lie derivatives of the barrier function along f and g

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        DHfc1_act = torch.matmul(d_tanh_dx(Hfc1_act), self.Hfc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        DHfc2_act = torch.bmm(torch.matmul(d_tanh_dx(Hfc2_act), self.Hfc_layer_2.weight), DHfc1_act)
        # Jacobian of third layer wrt input (n_batch x n_hidden x n_input)
        DHfc3_act = torch.bmm(torch.matmul(d_tanh_dx(Hfc3_act), self.Hfc_layer_3.weight), DHfc2_act)
        # Gradient of H wrt input (n_batch x 1 x n_input)
        grad_H = torch.matmul(self.Hfc_layer_4.weight, DHfc3_act)

        return H, grad_H

    def compute_lyapunov(self, x):
        """
        Computes the value and gradient of the Lyapunov function

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            V: the value of the Lyapunov at each provided point x [n_batch, 1]
            grad_V: the gradient of V [n_batch, n_dims]
        """
        # Use the first two layers to compute the Lyapunov function
        tanh = nn.Tanh()
        Vfc1_act = tanh(self.Vfc_layer_1(x))
        Vfc2_act = tanh(self.Vfc_layer_2(Vfc1_act))
        # Compute the Lyapunov function as the square norm of the last layer activations
        V = 0.5 * (Vfc2_act * Vfc2_act).sum(1)

        # We also need to calculate the Lie derivative of V along f and g
        #
        # L_f V = \grad V * f
        # L_g V = \grad V * g
        #
        # Since V = tanh(w2 * tanh(w1*x + b1) + b1),
        # grad V = d_tanh_dx(V) * w2 * d_tanh_dx(tanh(w1*x + b1)) * w1

        # Jacobian of first layer wrt input (n_batch x n_hidden x n_input)
        DVfc1_act = torch.matmul(d_tanh_dx(Vfc1_act), self.Vfc_layer_1.weight)
        # Jacobian of second layer wrt input (n_batch x n_hidden x n_input)
        DVfc2_act = torch.bmm(torch.matmul(d_tanh_dx(Vfc2_act), self.Vfc_layer_2.weight), DVfc1_act)
        # Gradient of V wrt input (n_batch x 1 x n_input)
        grad_V = torch.bmm(Vfc2_act.unsqueeze(1), DVfc2_act)

        return V, grad_V

    def forward(self, x):
        """
        Compute the forward pass of the controller

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            u: the input at the current state [n_batch, n_controls]
            r: the relaxation required to satisfy the CLF inequality
            V: the value of the Lyapunov function at a given point
            Vdot: the time derivative of the Lyapunov function
        """
        # Compute the Lyapunov and barrier functions
        V, grad_V = self.compute_lyapunov(x)
        H, grad_H = self.compute_barrier(x)

        # Compute lie derivatives for each scenario
        L_f_Vs = []
        L_g_Vs = []
        L_f_Hs = []
        L_g_Hs = []
        for scenario in self.scenarios:
            # Lyapunov Lie derivatives
            L_f_Vs.append(torch.bmm(grad_V, self.f(x, **scenario).unsqueeze(-1)).squeeze(-1))
            L_g_Vs.append(torch.bmm(grad_V, self.g(x, **scenario)).squeeze(1))

            # Barrier Lie derivaties
            L_f_Hs.append(torch.bmm(grad_H, self.f(x, **scenario).unsqueeze(-1)).squeeze(-1))
            L_g_Hs.append(torch.bmm(grad_H, self.g(x, **scenario)).squeeze(1))

        # Also get the nominal control input
        u_nominal = self.u_nominal(x, **self.nominal_scenario)

        # To find the control input, we need to solve a QP
        result = self.qp_layer(
            *L_f_Vs, *L_g_Vs,
            *L_f_Hs, *L_g_Hs,
            V.unsqueeze(-1),
            H,
            u_nominal,
            torch.tensor([self.clf_relaxation_penalty]),
            torch.tensor([self.cbf_relaxation_penalty]),
            solver_args={"max_iters": 5000000})
        u = result[0]
        rs = result[1:]

        # Average across scenarios
        n_scenarios = len(self.scenarios)
        Vdot = L_f_Vs[0].unsqueeze(-1) + torch.bmm(L_g_Vs[0].unsqueeze(1), u.unsqueeze(-1))
        Hdot = L_f_Hs[0].unsqueeze(-1) + torch.bmm(L_g_Hs[0].unsqueeze(1), u.unsqueeze(-1))
        relaxation = rs[0]
        for i in range(1, n_scenarios):
            Vdot += L_f_Vs[i].unsqueeze(-1) + torch.bmm(L_g_Vs[i].unsqueeze(1), u.unsqueeze(-1))
            Hdot += L_f_Hs[i].unsqueeze(-1) + torch.bmm(L_g_Hs[i].unsqueeze(1), u.unsqueeze(-1))
            relaxation += rs[i]

        Vdot /= n_scenarios
        Hdot /= n_scenarios
        relaxation /= n_scenarios

        return u, relaxation, V, Vdot, H, Hdot


def lyapunov_loss(x, x0, net, clf_lambda, timestep=0.001, print_loss=False):
    """
    Compute a loss to train the Lyapunov function

    args:
        x: the points at which to evaluate the loss
        x0: the origin
        net: a CLF_CBF_QP_Net instance
        clf_lambda: the rate parameter in the CLF condition
        timestep: the timestep used to compute a finite-difference approximation of the
                  Lyapunov function
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given Lyapunov function
    """
    # Compute loss based on...
    loss = 0.0
    #   1.) squared value of the Lyapunov function at the origin
    V0, _ = net.compute_lyapunov(x0)
    loss += V0.pow(2).squeeze()

    #   2.) A term to encourage satisfaction of CLF condition
    u, r, V, _, _, _ = net(x)
    # To compute the change in V, simulate x forward in time and check if V decreases in each
    # scenario
    lyap_descent_term = 0.0
    for s in net.scenarios:
        xdot = net.f(x, **s) + torch.bmm(net.g(x, **s), u.unsqueeze(-1)).squeeze()
        x_next = x + timestep * xdot
        V_next, _ = net.compute_lyapunov(x_next)
        Vdot = V_next - V
        lyap_descent_term += F.relu(Vdot.squeeze() + clf_lambda * V)
    loss += lyap_descent_term.mean()

    #   3.) A tuning term to encourage a quadratic-ish shape for V
    lyap_tuning_term = 0.1 * F.relu(0.1*(x*x).sum(1) - V)
    loss += lyap_tuning_term.mean()

    #   4.) A term to discourage relaxations of the CLF condition
    loss += r.mean()

    if print_loss:
        print(f"                     CLF origin: {V0.pow(2).squeeze().item()}")
        print(f"               CLF descent term: {lyap_descent_term.mean().item()}")
        print(f"                CLF tuning term: {lyap_tuning_term.mean().item()}")
        print(f"            CLF relaxation term: {r.mean().item()}")

    return loss


def barrier_loss(x,
                 safe_mask,
                 unsafe_mask,
                 net,
                 cbf_lambda,
                 timestep=0.001,
                 eps_safe=0.01,
                 eps_unsafe=0.1,
                 eps_dynamics=0.01,
                 print_loss=False):
    """
    Compute a loss to train the barrier function

    args:
        x: the points at which to evaluate the loss
        safe_mask: the points in x marked safe
        unsafe_mask: the points in x marked unsafe
        net: a CLF_CBF_QP_Net instance
        cbf_lambda: the rate parameter in the CBF condition
        timestep: the timestep used to compute a finite-difference approximation of the
                  barrier function
        eps: a parameter setting the buffer for strict inequalities
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given barrier function
    """
    # Compute loss based on...
    loss = 0.0
    #   1.) term to encourage barrier H >= 0 in the safe region
    H_safe, _ = net.compute_barrier(x[safe_mask])
    safe_region_barrier_term = F.relu(eps_safe - H_safe)
    loss += safe_region_barrier_term.mean()

    #   2.) term to encourage barrier H < 0 in the unsafe region
    H_unsafe, _ = net.compute_barrier(x[unsafe_mask])
    unsafe_region_barrier_term = F.relu(eps_unsafe + H_unsafe)
    loss += 5 * unsafe_region_barrier_term.mean()

    #   3.) term to encourage satisfaction of CBF condition
    u, _, _, _, H, _ = net(x)
    # To compute the change in H, simulate x forward in time and check if the CBF condition is
    # satisfied
    barrier_dynamics_term = 0.0
    for s in net.scenarios:
        xdot = net.f(x, **s) + torch.bmm(net.g(x, **s), u.unsqueeze(-1)).squeeze()
        x_next = x + timestep * xdot
        H_next, _ = net.compute_barrier(x_next)
        Hdot = H_next - H
        barrier_dynamics_term += F.relu(eps_dynamics - Hdot.squeeze() - cbf_lambda * H.squeeze())
    loss += barrier_dynamics_term.mean()

    if print_loss:
        print(f"               safe region term: {safe_region_barrier_term.mean().item()}")
        print(f"             unsafe region term: {unsafe_region_barrier_term.mean().item()}")
        print(f"          barrier dynamics term: {barrier_dynamics_term.mean().item()}")

    return loss
