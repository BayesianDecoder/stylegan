from .prelude import *
from lpips import LPIPS

try:
    from DISTS_pytorch import DISTS as _DISTSModel
    _DISTS_AVAILABLE = True
except ImportError:
    _DISTS_AVAILABLE = False


class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        # Fine→coarse: emphasise high-frequency detail at full resolution,
        # taper off at coarser scales.
        level_weights: List[float] = [2.0, 1.5, 1.0, 0.75, 0.5, 0.25],
        l1_weight: float = 0.05,
        dists_weight: float = 0.1,
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.dists_weight = dists_weight
        self.lpips_network = LPIPS(net="vgg", verbose=False).to(DEVICE)
        if _DISTS_AVAILABLE and dists_weight > 0:
            try:
                self.dists_network = _DISTSModel().to(DEVICE)
            except Exception as _e:
                print(f"[DISTS] init failed ({_e}); disabling DISTS loss.")
                self.dists_network = None
        else:
            self.dists_network = None

    def measure_lpips(self, x, y, mask):
        if mask is not None:
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)
        return self.lpips_network(x, y, normalize=True).mean()

    def _measure_dists(self, x, y):
        # DISTS expects [0, 1]; clamp to be safe
        return self.dists_network(x.clamp(0, 1), y.clamp(0, 1)).mean()

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None):
        x = f_hat(x_clean)

        losses = []

        if mask is not None:
            mask = F.interpolate(mask, y.shape[-1], mode="area")

        # DISTS at full resolution: captures both structure and texture globally
        if self.dists_network is not None and y.shape[-1] >= 32:
            losses.append(self.dists_weight * self._measure_dists(x, y))

        for weight in self.weights:
            if y.shape[-1] <= self.min_loss_res:
                break

            if weight > 0:
                loss = self.measure_lpips(x, y, mask)
                losses.append(weight * loss)

            if mask is not None:
                mask = F.avg_pool2d(mask, 2)

            x = F.avg_pool2d(x, 2)
            x_clean = F.avg_pool2d(x_clean, 2)
            y = F.avg_pool2d(y, 2)

        total = torch.stack(losses).sum(dim=0) if losses else torch.tensor(0.0, device=DEVICE)
        l1 = self.l1_weight * F.l1_loss(x, y)

        return total + l1
