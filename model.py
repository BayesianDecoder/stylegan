import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


TYPES  = ['upsample', 'denoise', 'deartifact', 'inpaint']
LEVELS = ['XS', 'S', 'M', 'L', 'XL']


# ─────────────────────────────────────────────────────────────────────────────
# Main dual-head estimator (type + severity)
# ─────────────────────────────────────────────────────────────────────────────

class DegradationEstimator(nn.Module):
    """Dual-head MobileNetV3-Small: predicts degradation type + severity."""

    def __init__(self, dropout_rate: float = 0.3, freeze_backbone: bool = True):
        super().__init__()
        backbone      = models.mobilenet_v3_small(weights='DEFAULT')
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        feature_dim   = 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.type_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 4)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 5)
        )

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        return self.type_head(feat), self.severity_head(feat)

    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            type_logits, sev_logits = self.forward(x)
            type_probs = torch.sigmoid(type_logits[0])
            predicted_types = [TYPES[i] for i, p in enumerate(type_probs) if p > threshold]
            if not predicted_types:
                predicted_types = [TYPES[type_probs.argmax().item()]]
            sev_probs = torch.softmax(sev_logits[0], dim=0)
            return predicted_types, LEVELS[sev_probs.argmax().item()], type_probs, sev_probs


def load_model(model_path, device):
    model = DegradationEstimator(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Specialist backbone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_head(in_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout * 0.5),
        nn.Linear(256, 5),
    )


def _partial_unfreeze(features, n_blocks_to_unfreeze=1):
    """Freeze all blocks then unfreeze the last n — prevents destroying early features."""
    for p in features.parameters():
        p.requires_grad = False
    blocks = list(features.children())
    for block in blocks[-n_blocks_to_unfreeze:]:
        for p in block.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# SeveritySpecialist — MobileNetV3-Small (upsample only, already accurate)
# ─────────────────────────────────────────────────────────────────────────────

class SeveritySpecialist(nn.Module):
    """MobileNetV3-Small specialist — kept for upsample (already >90%)."""

    def __init__(self, deg_type, dropout_rate=0.4, freeze_backbone=True):
        super().__init__()
        self.deg_type = deg_type
        backbone      = models.mobilenet_v3_small(weights='DEFAULT')
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        feat_dim      = 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.severity_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 5)
        )

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        return self.severity_head(feat)

    def predict_severity(self, x):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x)[0], dim=0)
            return LEVELS[probs.argmax().item()], probs


# ─────────────────────────────────────────────────────────────────────────────
# DenoiseSpecialist — statistics-only MLP, NO backbone
#
# WHY no backbone: ImageNet pretraining teaches every CNN to IGNORE noise
# (noise is "corruption" in ImageNet). Frozen backbone features actively
# mislead the head (val acc stuck ~59%). Fine-tuning is slow and still
# fights against years of noise-invariant pretraining.
#
# Fix: skip the backbone entirely. Compute 8 hand-crafted noise statistics
# (Laplacian, Sobel gradients, local variance, HF energy) and feed them to
# a tiny MLP. These statistics directly measure noise power at multiple
# scales and cleanly separate XS/S/M/L/XL severity levels.
# ─────────────────────────────────────────────────────────────────────────────

class DenoiseSpecialist(nn.Module):
    """
    Statistics-only MLP for noise severity. NO backbone, NO mixup.

    Mixup corrupts noise statistics: std(lam*A + (1-lam)*B) ≠ lam*std(A) +
    (1-lam)*std(B), so the model learns wrong stats→label mapping during
    training and predicts randomly on val (pure images, no mixing).

    Uses 10 multi-scale statistics: 3-scale Laplacian + Sobel + local variance.
    Multi-scale separates noise from image edges — at 4× downsampling, noise
    averages out but edges remain, giving a clean noise-only signal.
    """

    def __init__(self, deg_type="denoise", dropout_rate=0.3, freeze_backbone=True):
        super().__init__()
        self.deg_type = deg_type

        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        sx  = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sy  = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("_lap", lap.view(1, 1, 3, 3))
        self.register_buffer("_sx",  sx.view(1, 1, 3, 3))
        self.register_buffer("_sy",  sy.view(1, 1, 3, 3))

        # 10 stats → MLP. No BatchNorm — avoids running-stat mismatch on val.
        self.mlp = nn.Sequential(
            nn.Linear(10, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 5),
        )

    def _noise_stats(self, x):
        gray = x.mean(dim=1, keepdim=True)          # (B,1,H,W)

        # 3-scale Laplacian std — noise averages out at lower scales but
        # edges persist, so scale ratio isolates noise contribution
        def lap_std(g):
            return F.conv2d(g, self._lap, padding=1).flatten(1).std(dim=1, keepdim=True)

        s1 = lap_std(gray)                               # full scale
        s2 = lap_std(F.avg_pool2d(gray, 2))              # 2× down
        s4 = lap_std(F.avg_pool2d(gray, 4))              # 4× down
        # ratio s1/s4: if mostly noise, ratio ≈ 4; if edges, ratio ≈ 1
        ratio = s1 / (s4 + 1e-6)

        # Sobel gradient magnitude
        gx      = F.conv2d(gray, self._sx, padding=1)
        gy      = F.conv2d(gray, self._sy, padding=1)
        grad    = (gx ** 2 + gy ** 2).sqrt()
        g_mu    = grad.flatten(1).mean(dim=1, keepdim=True)
        g_std   = grad.flatten(1).std(dim=1, keepdim=True)

        # Local residual variance (noise in smooth regions)
        lmu     = F.avg_pool2d(gray, 8, stride=4, padding=2)
        lmu_up  = F.interpolate(lmu, size=gray.shape[2:], mode='nearest')
        resid   = (gray - lmu_up) ** 2
        lv_mu   = resid.flatten(1).mean(dim=1, keepdim=True)
        lv_std  = resid.flatten(1).std(dim=1, keepdim=True)

        # High-frequency energy
        blur    = F.avg_pool2d(gray, 5, stride=1, padding=2)
        hf      = (gray - blur).abs()
        hf_mu   = hf.flatten(1).mean(dim=1, keepdim=True)
        hf_std  = hf.flatten(1).std(dim=1, keepdim=True)

        return torch.cat([s1, s2, s4, ratio, g_mu, g_std,
                          lv_mu, lv_std, hf_mu, hf_std], dim=1)  # (B, 10)

    def unfreeze_backbone(self):
        pass

    def forward(self, x):
        return self.mlp(self._noise_stats(x))

    def predict_severity(self, x):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x)[0], dim=0)
            return LEVELS[probs.argmax().item()], probs


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNetSpecialist — B0 for deartifact, B2 for inpaint
#
# deartifact: EfficientNet-B0 is much better than MobileNetV3 at capturing
#   blocking/ringing artifacts (texture-frequency sensitivity).
# inpaint: EfficientNet-B2 has a larger receptive field — needed to detect
#   the spatial extent of inpainting masks (small=XS vs large=XL).
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetSpecialist(nn.Module):

    _FEAT_DIM = {"b0": 1280, "b2": 1408}

    def __init__(self, deg_type, variant="b0", dropout_rate=0.45, freeze_backbone=True):
        super().__init__()
        self.deg_type = deg_type
        self.variant  = variant

        if variant == "b0":
            backbone = models.efficientnet_b0(weights='DEFAULT')
        elif variant == "b2":
            backbone = models.efficientnet_b2(weights='DEFAULT')
        else:
            raise ValueError(f"Unknown variant {variant}")

        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        feat_dim      = self._FEAT_DIM[variant]

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.severity_head = _make_head(feat_dim, dropout_rate)

    def unfreeze_backbone(self):
        _partial_unfreeze(self.features, n_blocks_to_unfreeze=3)
        print(f"[{self.deg_type}] Partial unfreeze — last 3 EfficientNet-{self.variant} blocks")

    def forward(self, x):
        feat = self.avgpool(self.features(x)).flatten(1)
        return self.severity_head(feat)

    def predict_severity(self, x):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x)[0], dim=0)
            return LEVELS[probs.argmax().item()], probs


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_specialist(deg_type, freeze_backbone=True):
    """Returns the best specialist model for each degradation type."""
    if deg_type == "upsample":
        return SeveritySpecialist(deg_type, dropout_rate=0.4,  freeze_backbone=freeze_backbone)
    elif deg_type == "denoise":
        return DenoiseSpecialist(deg_type,  dropout_rate=0.5,  freeze_backbone=freeze_backbone)
    elif deg_type == "deartifact":
        return EfficientNetSpecialist(deg_type, variant="b0", dropout_rate=0.45, freeze_backbone=freeze_backbone)
    elif deg_type == "inpaint":
        return EfficientNetSpecialist(deg_type, variant="b2", dropout_rate=0.45, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown deg_type: {deg_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_specialist(model_path, deg_type, device):
    model = build_specialist(deg_type, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded [{deg_type}] specialist from {model_path}")
    return model.to(device).eval()


def load_all_specialists(specialists_dir, device):
    specialists = {}
    for deg_type in TYPES:
        path = os.path.join(specialists_dir, f"{deg_type}_specialist.pth")
        if os.path.exists(path):
            specialists[deg_type] = load_specialist(path, deg_type, device)
        else:
            print(f"Warning: specialist not found at {path}")
    return specialists
