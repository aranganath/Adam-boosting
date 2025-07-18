import torch
from torch.optim.optimizer import Optimizer

class LayerWiseCompactLBFGS(Optimizer):
    def __init__(self, params, lr=1.0, history_size=1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.history_size = history_size

    def _compute_direction(self, g, S, Y, gamma):
        """
        Compute d = -H_k @ g using compact L-BFGS update per parameter tensor.
        """
        m = S.shape[1]
        if m == 0:
            return -gamma * g

        R = torch.triu(S.T @ Y)             # [m, m]
        D = torch.diag(S.T @ Y)             # [m]
        H0g = gamma * g                     # [d]
        H0Y = gamma * Y                     # [d, m]
        alpha = S.T @ g                     # [m]
        beta = Y.T @ H0g                    # [m]
        YTH0Y = Y.T @ H0Y                   # [m, m]
        M = torch.diag(D) + YTH0Y           # [m, m]

        z = torch.linalg.solve(R, alpha)    # [m]
        Mz = M @ z                          # [m]
        u = torch.linalg.solve(R.T, Mz - beta)
        v = torch.linalg.solve(R, beta)

        direction = -H0g - S @ u + H0Y @ v  # [d]
        return direction

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            # from pdb import set_trace
            # set_trace()
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev_param'] = p.data.clone()
                    state['prev_grad'] = grad.clone()
                    state['S'] = []
                    state['Y'] = []
                    state['H0'] = 1.0  # scalar init

                state['step'] += 1
                s_k = p.data - state['prev_param']
                y_k = grad - state['prev_grad']

                if s_k.reshape(-1).dot(y_k.reshape(-1)) > 1e-12:
                    # Maintain fixed-size buffer
                    if len(state['S']) >= self.history_size:
                        state['S'].pop(0)
                        state['Y'].pop(0)
                    state['S'].append(s_k.clone())
                    state['Y'].append(y_k.clone())

                    # Update gamma
                    ys = torch.dot(y_k.flatten(), s_k.flatten())
                    yy = torch.dot(y_k.flatten(), y_k.flatten())
                    if yy > 1e-12:
                        state['H0'] = ys / yy

                # Stack history
                if len(state['S']) > 0:
                    S = torch.stack([s.view(-1) for s in state['S']], dim=1)  # [d, m]
                    Y = torch.stack([y.view(-1) for y in state['Y']], dim=1)
                    d = self._compute_direction(grad.view(-1), S, Y, state['H0']).view_as(p)
                else:
                    d = -state['H0'] * grad  # fallback

                # Update parameters
                p.data += lr * d

                # Save for next step
                state['prev_param'] = p.data.clone()
                state['prev_grad'] = grad.clone()

        return loss
