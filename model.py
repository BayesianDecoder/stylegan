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
        gray     = x.mean(dim=1, keepdim=True)                      # (B,1,H,W)
        img_std  = gray.flatten(1).std(dim=1, keepdim=True) + 1e-6  # (B,1) global scale

        def lap_std(g):
            return F.conv2d(g, self._lap, padding=1).flatten(1).std(dim=1, keepdim=True)

        s1    = lap_std(gray)
        s2    = lap_std(F.avg_pool2d(gray, 2))
        s4    = lap_std(F.avg_pool2d(gray, 4))
        # ratio isolates noise vs edges; s1/s4 ≈ 4 for pure noise, ≈ 1 for edges
        ratio = s1 / (s4 + 1e-6)

        gx    = F.conv2d(gray, self._sx, padding=1)
        gy    = F.conv2d(gray, self._sy, padding=1)
        grad  = (gx ** 2 + gy ** 2).sqrt()
        g_mu  = grad.flatten(1).mean(dim=1, keepdim=True)
        g_std = grad.flatten(1).std(dim=1, keepdim=True)

        lmu    = F.avg_pool2d(gray, 8, stride=4, padding=2)
        lmu_up = F.interpolate(lmu, size=gray.shape[2:], mode='nearest')
        resid  = (gray - lmu_up) ** 2
        lv_mu  = resid.flatten(1).mean(dim=1, keepdim=True)
        lv_std = resid.flatten(1).std(dim=1, keepdim=True)

        blur   = F.avg_pool2d(gray, 5, stride=1, padding=2)
        hf     = (gray - blur).abs()
        hf_mu  = hf.flatten(1).mean(dim=1, keepdim=True)
        hf_std = hf.flatten(1).std(dim=1, keepdim=True)

        # Normalise by image global std → scale-invariant statistics.
        # Removes the ~15% noise level difference caused by Resize(256)→crop
        # vs Resize(224) in different dataset splits.
        stats = torch.cat([s1, s2, s4, ratio, g_mu, g_std,
                           lv_mu, lv_std, hf_mu, hf_std], dim=1)   # (B, 10)
        return stats / img_std

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
# DeartifactSpecialist — statistics-only MLP with 8×8 block boundary features
#
# Same conclusion as denoise: frozen EfficientNet suppresses JPEG artifact
# signals (ImageNet training optimises away frequency artefacts as noise).
# JPEG artifacts live in 8×8 block boundaries and ringing patterns — directly
# measurable with block residuals and boundary discontinuity statistics.
# ─────────────────────────────────────────────────────────────────────────────

class DeartifactSpecialist(nn.Module):

    def __init__(self, deg_type="deartifact", dropout_rate=0.3, freeze_backbone=True):
        super().__init__()
        self.deg_type = deg_type
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("_lap", lap.view(1, 1, 3, 3))

        # 9 artifact statistics → MLP
        self.mlp = nn.Sequential(
            nn.Linear(9, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 5),
        )

    def _artifact_stats(self, x):
        gray    = x.mean(dim=1, keepdim=True)
        img_std = gray.flatten(1).std(dim=1, keepdim=True) + 1e-6

        # 8×8 block residual — core JPEG blocking signal
        blk     = F.avg_pool2d(gray, 8, stride=8)
        blk_up  = F.interpolate(blk, size=gray.shape[2:], mode='nearest')
        resid   = (gray - blk_up).abs()
        br_mu   = resid.flatten(1).mean(dim=1, keepdim=True)
        br_std  = resid.flatten(1).std(dim=1, keepdim=True)

        # Boundary discontinuities at 8-pixel intervals (JPEG block edges)
        h_diff  = (gray[:, :, :, 8:] - gray[:, :, :, :-8]).abs()
        v_diff  = (gray[:, :, 8:, :] - gray[:, :, :-8, :]).abs()
        h_mu    = h_diff.flatten(1).mean(dim=1, keepdim=True)
        v_mu    = v_diff.flatten(1).mean(dim=1, keepdim=True)

        # Ringing: Laplacian HF energy (increases with artifact severity)
        lap_std = F.conv2d(gray, self._lap, padding=1).flatten(1).std(dim=1, keepdim=True)

        # Multi-scale: at 2× down, 8px blocks → 4px — separates blocks from content
        gray2   = F.avg_pool2d(gray, 2)
        blk2    = F.avg_pool2d(gray2, 4, stride=4)
        blk2_up = F.interpolate(blk2, size=gray2.shape[2:], mode='nearest')
        br2_mu  = (gray2 - blk2_up).abs().flatten(1).mean(dim=1, keepdim=True)
        lap2_std = F.conv2d(gray2, self._lap, padding=1).flatten(1).std(dim=1, keepdim=True)

        # Block edge sharpness ratio (sharp block edges = more artifacts)
        sharp   = (h_mu + v_mu) / (br_mu + 1e-6)

        stats = torch.cat([br_mu, br_std, h_mu, v_mu, lap_std,
                           br2_mu, lap2_std, sharp,
                           gray.flatten(1).std(dim=1, keepdim=True)], dim=1)  # (B,9)
        return stats / img_std

    def unfreeze_backbone(self):
        pass  # no backbone

    def forward(self, x):
        return self.mlp(self._artifact_stats(x))

    def predict_severity(self, x):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x)[0], dim=0)
            return LEVELS[probs.argmax().item()], probs


# ─────────────────────────────────────────────────────────────────────────────
# InpaintSpecialist — EfficientNet-B2, full backbone fine-tune from Phase B
#
# Inpaint severity = mask spatial extent (XS=tiny, XL=huge).
# Statistics lose spatial information so backbone IS needed here.
# Full backbone unfreeze at Phase B with low LR forces the model to learn
# mask-extent features while ImageNet pretraining provides a useful warm start.
# ─────────────────────────────────────────────────────────────────────────────

class InpaintSpecialist(nn.Module):

    def __init__(self, deg_type="inpaint", dropout_rate=0.45, freeze_backbone=True):
        super().__init__()
        self.deg_type = deg_type
        backbone      = models.efficientnet_b2(weights='DEFAULT')
        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.severity_head = _make_head(1408, dropout_rate)

    def unfreeze_backbone(self):
        # Unfreeze ALL layers — mask extent needs full spatial feature adaptation
        for p in self.features.parameters():
            p.requires_grad = True
        print(f"[{self.deg_type}] Full backbone unfreeze — all EfficientNet-B2 layers")

    def forward(self, x):
        return self.severity_head(self.avgpool(self.features(x)).flatten(1))

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
        return SeveritySpecialist(deg_type,   dropout_rate=0.4,  freeze_backbone=freeze_backbone)
    elif deg_type == "denoise":
        return DenoiseSpecialist(deg_type,    dropout_rate=0.3,  freeze_backbone=freeze_backbone)
    elif deg_type == "deartifact":
        return DeartifactSpecialist(deg_type, dropout_rate=0.3,  freeze_backbone=freeze_backbone)
    elif deg_type == "inpaint":
        return InpaintSpecialist(deg_type,    dropout_rate=0.45, freeze_backbone=freeze_backbone)
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
