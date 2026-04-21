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


def _partial_unfreeze(features, n_blocks_to_unfreeze=3):
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
# DenoiseSpecialist — EfficientNet-B0 + Laplacian noise branch
#
# WHY the noise branch: ImageNet pretraining teaches the backbone to IGNORE
# noise (it's treated as irrelevant for object recognition). The Laplacian
# residual directly measures high-frequency noise power — bypassing the
# backbone's learned noise-invariance — giving the head an explicit severity
# signal the backbone would otherwise suppress.
# ─────────────────────────────────────────────────────────────────────────────

class DenoiseSpecialist(nn.Module):

    def __init__(self, deg_type="denoise", dropout_rate=0.5, freeze_backbone=True):
        super().__init__()
        self.deg_type = deg_type
        backbone      = models.efficientnet_b0(weights='DEFAULT')
        self.features = backbone.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)
        feat_dim      = 1280

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Laplacian kernel — fixed, not learned
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("_lap", lap.view(1, 1, 3, 3))

        # +1 for the scalar Laplacian noise estimate
        self.severity_head = _make_head(feat_dim + 1, dropout_rate)

    def _noise_estimate(self, x):
        gray = x.mean(dim=1, keepdim=True)
        lap  = F.conv2d(gray, self._lap, padding=1)
        return lap.flatten(1).std(dim=1, keepdim=True)   # (B, 1)

    def unfreeze_backbone(self):
        _partial_unfreeze(self.features, n_blocks_to_unfreeze=3)
        print(f"[{self.deg_type}] Partial unfreeze — last 3 EfficientNet blocks")

    def forward(self, x):
        feat      = self.avgpool(self.features(x)).flatten(1)   # (B, 1280)
        noise_est = self._noise_estimate(x)                      # (B, 1)
        return self.severity_head(torch.cat([feat, noise_est], dim=1))

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
