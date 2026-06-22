import torch


class Example3DDynamics:
    """
    Continuous-time 3-state polynomial system:

        x1dot = -x1 + x3
        x2dot = x1^2 - x2 - 2 * x1 * x3 + x3
        x3dot = -x2 + u

    The origin x* = 0 is an equilibrium with u* = 0.

    Note on the linearization at the origin:
        A = [[-1,  0,  1],
             [ 0, -1,  1],
             [ 0, -1,  0]],   B = [0, 0, 1]^T
    The pair (A, B) is not fully controllable, but the uncontrollable mode
    z = x1 - x2 obeys zdot = -z and is therefore stable, so (A, B) is
    stabilizable and an LQR controller (via the continuous-time ARE) exists.
    """

    def __init__(self):
        self.nx = 3
        self.nu = 1

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        x: size is (batch, 3)
        u: size is (batch, 1)
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        x3 = x[:, 2:3]

        x1dot = -x1 + x3
        x2dot = x1**2 - x2 - 2 * x1 * x3 + x3
        x3dot = -x2 + u
        return torch.cat((x1dot, x2dot, x3dot), dim=1)

    def linearized_dynamics(self, x, u):
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx), device=device)
        B = torch.zeros((batch_size, self.nx, self.nu), device=device)
        x1 = x[:, 0]
        x3 = x[:, 2]
        # Row 0: d/dx (-x1 + x3)
        A[:, 0, 0] = -1.0
        A[:, 0, 2] = 1.0
        # Row 1: d/dx (x1^2 - x2 - 2*x1*x3 + x3)
        A[:, 1, 0] = 2 * x1 - 2 * x3
        A[:, 1, 1] = -1.0
        A[:, 1, 2] = 1.0 - 2 * x1
        # Row 2: d/dx (-x2 + u)
        A[:, 2, 1] = -1.0
        B[:, 2, 0] = 1.0
        return A.to(device), B.to(device)

    @property
    def x_equilibrium(self):
        return torch.zeros((3,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
