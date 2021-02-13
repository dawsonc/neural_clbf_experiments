import torch
import torch.nn as nn
import torch.nn.functional as F


def d_tanh_dx(tanh):
    return torch.diag_embed(1 - tanh**2)


class LF_Net(nn.Module):
    """A neural network for simultaneously computing the Lyapunov function and the
    control input.
    """

    def __init__(self, n_input, n_hidden, n_controls, clf_lambda, clf_relaxation_penalty,
                 dynamics, u_nominal):
        """
        Initialize the network

        args:
            n_input: number of states the system has
            n_hidden: number of hiddent layers to use
            n_controls: number of control outputs to use
            clf_lambda: desired exponential convergence rate for the CLF
            clf_relaxation_penalty: the penalty for relaxing the control lyapunov constraint
            dynamics: a function that takes n_batch x n_dims state and n_batch x n_controls controls
                      and returns the state derivatives
            u_nominal: a function n_batch x n_dims -> n_batch x n_controls that returns the nominal
                       control input for the system (even LQR about origin is fine)
        """
        super(LF_Net, self).__init__()

        # Save the dynamics and nominal controller functions
        self.dynamics = dynamics
        self.u_nominal = u_nominal

        # The network will have the following architecture
        #
        # n_input -> VFC1 (n_input x n_hidden) -> VFC2 (n_hidden, n_hidden)
        # -> VFC2 (n_hidden, n_hidden) -> V = x^T x --> QP -> u
        self.Vfc_layer_1 = nn.Linear(n_input, n_hidden)
        self.Vfc_layer_2 = nn.Linear(n_hidden, n_hidden)

        # We also train a controller to learn the nominal control input
        self.Ufc_layer_1 = nn.Linear(n_input, n_hidden)
        self.Ufc_layer_2 = nn.Linear(n_hidden, n_hidden)
        self.Ufc_layer_3 = nn.Linear(n_hidden, n_controls)

        self.n_controls = n_controls
        self.clf_lambda = clf_lambda
        self.clf_relaxation_penalty = clf_relaxation_penalty

    def compute_controls(self, x):
        """
        Computes the control input (for use in the QP filter)

        args:
            x: the state at the current timestep [n_batch, n_dims]
        returns:
            u: the value of the barrier at each provided point x [n_batch, n_controls]
        """
        tanh = nn.Tanh()
        Ufc1_act = tanh(self.Ufc_layer_1(x))
        Ufc2_act = tanh(self.Ufc_layer_2(Ufc1_act))
        U = self.Ufc_layer_3(Ufc2_act)

        return U

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

        # We also need to calculate the gradient of V
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
        u_learned = self.compute_controls(x)

        # Compute lie derivative
        f = self.dynamics(x, u_learned)
        L_f_V = torch.bmm(grad_V, f.unsqueeze(-1)).squeeze(-1)

        u = u_learned

        Vdot = L_f_V.unsqueeze(-1)

        return u, V, Vdot


def lyapunov_loss(x,
                  x_goal,
                  net,
                  clf_lambda,
                  safe_level=1.0,
                  timestep=0.001,
                  print_loss=False):
    """
    Compute a loss to train the Lyapunov function

    args:
        x: the points at which to evaluate the loss
        x_goal: the origin
        net: a CLF_CBF_QP_Net instance
        clf_lambda: the rate parameter in the CLF condition
        safe_level: defines the safe region as the sublevel set of the lyapunov function
        timestep: the timestep used to compute a finite-difference approximation of the
                  Lyapunov function
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given Lyapunov function
    """
    # Compute loss based on...
    loss = 0.0
    #   1.) squared value of the Lyapunov function at the goal
    V0, _ = net.compute_lyapunov(x_goal)
    loss += V0.pow(2).mean()

    #   2.) A term to encourage satisfaction of CLF condition
    u, V, _ = net(x)
    # To compute the change in V, simulate x forward in time and check if V decreases
    xdot = net.dynamics(x, u)
    x_next = x + timestep * xdot
    V_next, _ = net.compute_lyapunov(x_next)
    Vdot = (V_next.squeeze() - V.squeeze()) / timestep
    lyap_descent_term = F.relu(Vdot + clf_lambda * V.squeeze())
    # lyap_descent_term *= 100
    loss += lyap_descent_term.mean()

    #   3.) term to encourage V <= safe_level in the safe region
    safe_mask = x[:, 11] >= 100
    V_safe, _ = net.compute_lyapunov(x[safe_mask[:, 0], :])
    safe_region_lyapunov_term = F.relu(V_safe - safe_level)
    if safe_region_lyapunov_term.numel() > 0:
        loss += safe_region_lyapunov_term.mean()

    #   4.) term to encourage V >= safe_level in the unsafe region
    unsafe_mask = x[:, 11] <= 0
    V_unsafe, _ = net.compute_lyapunov(x[unsafe_mask[:, 0], :])
    unsafe_region_lyapunov_term = F.relu(safe_level - V_unsafe)
    if unsafe_region_lyapunov_term.numel() > 0:
        loss += unsafe_region_lyapunov_term.mean()

    if print_loss:
        print(f"                     CLF origin: {V0.pow(2).mean().item()}")
        print(f"           CLF safe region term: {safe_region_lyapunov_term.mean().item()}")
        print(f"         CLF unsafe region term: {unsafe_region_lyapunov_term.mean().item()}")
        print(f"               CLF descent term: {lyap_descent_term.mean().item()}")

    return loss


def controller_loss(x, net, print_loss=False):
    """
    Compute a loss to train the filtered controller

    args:
        x: the points at which to evaluate the loss
        net: a CLF_CBF_QP_Net instance
        print_loss: True to enable printing the values of component terms
    returns:
        loss: the loss for the given controller function
    """
    Nz_nominal, _, _, throttle_nominal = net.u_nominal(x)
    u_nominal = torch.hstack((Nz_nominal.unsqueeze(-1), throttle_nominal.unsqueeze(-1)))
    u_learned, _, _ = net(x)

    # Compute loss based on difference from nominal controller (e.g. LQR) at all points
    controller_squared_error = 1e-3 * ((u_nominal - u_learned)**2).sum(dim=-1)
    loss = controller_squared_error.mean()

    if print_loss:
        print(f"                controller term: {controller_squared_error.mean().item()}")

    return loss
