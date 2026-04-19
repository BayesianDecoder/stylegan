import torch 


class NGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                # Sanitize params that may have NaN/inf from a prior phase
                if param.isnan().any() or param.isinf().any():
                    torch.nan_to_num_(param, nan=0.0, posinf=0.0, neginf=0.0)
                g = param.grad
                # Skip step entirely if gradients are still invalid
                if g.isnan().any() or g.isinf().any():
                    continue
                norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                g = g / norm
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                param -= group["lr"] * g


class LBFGSPhase(torch.optim.LBFGS):
    def __init__(self, params, lr=1.0, max_iter=20, history_size=10,
                 line_search_fn="strong_wolfe"):
        super().__init__(params, lr=lr, max_iter=max_iter,
                         history_size=history_size,
                         line_search_fn=line_search_fn)
