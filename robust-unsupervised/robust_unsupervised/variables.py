from .prelude import *


# Gaussian noise σ added to the encoder's W output before optimization.
# Larger degradation → more noise to encourage escaping poor local minima
# caused by the unreliable degraded-image signal.
_NOISE_SCALE = {
    "XS": 0.005,
    "S":  0.010,
    "M":  0.020,
    "L":  0.050,
    "XL": 0.100,
}


def warm_start_W(
    G,
    image_01: torch.Tensor,
    level_str: str,
    encoder=None,
) -> "WVariable":
    """Return a WVariable initialised from an encoder + adaptive Gaussian noise.

    Args:
        G: StyleGAN generator.
        image_01: degraded target image in [0, 1], shape [1, 3, H, W].
        level_str: degradation level key ("XS"/"S"/"M"/"L"/"XL").
        encoder: callable image→W [1, 512] (e4e or compatible).
                 Falls back to G.mapping.w_avg when None.
    """
    noise_std = _NOISE_SCALE.get(level_str, 0.02)

    if encoder is not None:
        try:
            with torch.no_grad():
                # e4e expects [-1, 1], 256×256
                img = (image_01 * 2.0 - 1.0).clamp(-1, 1)
                img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
                w = encoder(img)          # [1, 512] or [1, 18, 512]
                if w.dim() == 3:
                    w = w[:, 0, :]        # take first layer as W estimate
        except Exception:
            encoder = None               # fall through to w_avg

    if encoder is None:
        w = G.mapping.w_avg.reshape(1, G.w_dim).clone().to(DEVICE)

    w = w + noise_std * torch.randn_like(w)
    return WVariable(G, nn.Parameter(w.to(DEVICE)))


class Variable(nn.Module):
    def __init__(self, G: networks.Generator, data: torch.Tensor):
        super().__init__()
        self.G = G
        self.data = data

    # ------------------------------------

    @staticmethod
    def sample_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    @staticmethod
    def sample_random_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    def to_input_tensor(self):
        raise NotImplementedError

    # ------------------------------------

    def parameters(self):
        return [self.data]

    def to_image(self):
        return self.render_image(self.to_input_tensor())

    def render_image(self, ws: torch.Tensor):
        """
        ws shape: [batch_size, num_layers, 512]
        """
        return (self.G.synthesis(ws, noise_mode="const", force_fp32=True) + 1.0) / 2.0

    def detach(self):
        data = self.data.detach().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data
        return self.__class__(self.G, data)

    def clone(self):
        data = self.data.detach().clone().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else self.data
        return self.__class__(self.G, data)

    def interpolate(self, other: "Variable", alpha: float = 0.5):
        assert self.G == other.G
        return self.__class__(self.G, self.data.lerp(other.data, alpha))

    def __add__(self, other: "Variable"):
        return self.from_data(self.data + other.data)

    def __sub__(self, other: "Variable"):
        return self.from_data(self.data - other.data)

    def __mul__(self, scalar: float):
        return self.from_data(self.data * scalar)

    def unbind(self):
        return [
            self.__class__(
                self.G,
                nn.Parameter(p.unsqueeze(0))
                if isinstance(self.data, nn.Parameter)
                else p.unsqueeze(0),
            )
            for p in self.data
        ]


class WVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping.w_avg.reshape(1, G.w_dim).repeat(batch_size, 1)
        return WVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = G.mapping(
            torch.randn(batch_size, G.z_dim).to(DEVICE),
            None,
            skip_w_avg_update=True,
        )[:, 0]
        return WVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data.unsqueeze(1).repeat(1, self.G.num_ws, 1)

    @torch.no_grad()
    def truncate(self, truncation: float = 1.0):
        assert 0.0 <= truncation <= 1.0
        self.data.lerp_(self.G.mapping.w_avg.reshape(1, 512), 1.0 - truncation)
        return self


class WpVariable(Variable):
    def __init__(self, G, data: torch.Tensor):
        super().__init__(G, data)

    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.to_input_tensor(WVariable.sample_from(G, batch_size))
        return WpVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            G.mapping(
                (torch.randn(batch_size * G.mapping.num_ws, G.z_dim).to(DEVICE)),
                None,
                skip_w_avg_update=True,
            )
            .mean(dim=1)
            .reshape(batch_size, G.mapping.num_ws, G.w_dim)
        )
        return WpVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        return self.data

    def mix(self, other: "WpVariable", num_layers: float):
        return WpVariable(
            self.G,
            torch.cat(
                (self.data[:, :num_layers, :], other.data[:, num_layers:, :]), dim=1
            ),
        )

    @staticmethod
    def from_W(W: WVariable):
        return WpVariable(W.G, nn.parameter.Parameter(W.to_input_tensor()))

    @torch.no_grad()
    def truncate(self, truncation=1.0, *, layer_start=0, layer_end: Optional[int] = None):
        assert 0.0 <= truncation <= 1.0
        mu = self.G.mapping.w_avg
        target = mu.reshape(1, 1, 512).repeat(1, self.G.mapping.num_ws, 1)
        self.data[:, layer_start:layer_end].lerp_(target[:, layer_start:layer_end], 1.0 - truncation)
        return self


class WppVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        data = WVariable.sample_from(G, batch_size).to_input_tensor().repeat(1, 512, 1)
        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1):
        data = (
            WVariable.sample_random_from(G, batch_size)
            .to_input_tensor()
            .repeat(1, 512, 1)
        )
        return WppVariable(G, nn.Parameter(data))

    @staticmethod
    def from_w(W: WVariable):
        data = W.data.detach().repeat(1, 512 * W.G.num_ws, 1)
        return WppVariable(W.G, nn.parameter.Parameter(data))

    @staticmethod
    def from_Wp(Wp: WpVariable):
        data = Wp.data.detach().repeat_interleave(512, dim=1)
        return WppVariable(Wp.G, nn.parameter.Parameter(data))

    def to_input_tensor(self):
        return self.data
