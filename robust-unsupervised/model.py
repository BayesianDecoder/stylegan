# """
# model.py
# --------
# Degradation Estimator — dual-head CNN classifier.

# Architecture:
#     Shared backbone : MobileNetV3-Small (pretrained on ImageNet)
#     Head 1          : Type head    — sigmoid, 4 outputs (multi-label)
#     Head 2          : Severity head — softmax, 5 outputs (single-label)

# Why MobileNetV3-Small:
#     - Lightweight — trains fast on M3
#     - Pretrained — already knows edges and textures
#     - 576-dim feature vector — enough for this task
# """

# import torch
# import torch.nn as nn
# import torchvision.models as models


# TYPES  = ['upsample', 'denoise', 'deartifact', 'inpaint']
# LEVELS = ['XS', 'S', 'M', 'L', 'XL']


# class DegradationEstimator(nn.Module):
#     """
#     Dual-head classifier for degradation type and severity.

#     Forward returns:
#         type_logits     : (batch, 4)  — raw logits, no sigmoid
#         severity_logits : (batch, 5)  — raw logits, no softmax

#     Both activations are applied inside the loss functions:
#         type_logits     → BCEWithLogitsLoss (handles sigmoid internally)
#         severity_logits → CrossEntropyLoss  (handles softmax internally)
#     """

#     def __init__(self, dropout_rate: float = 0.3, freeze_backbone: bool = True):
#         super().__init__()

#         # ── Backbone ────────────────────────────────────────────────────────
#         backbone = models.mobilenet_v3_small(weights='DEFAULT')

#         # Extract feature extractor and pooling — remove classifier
#         self.features = backbone.features   # CNN feature extractor
#         self.avgpool  = backbone.avgpool    # Global average pool → (batch, 576, 1, 1)
#         feature_dim   = 576

#         # Optionally freeze backbone for first training phase
#         # (train heads only → prevents early overfitting)
#         if freeze_backbone:
#             for param in self.features.parameters():
#                 param.requires_grad = False

#         # ── Type head — multi-label ──────────────────────────────────────────
#         # Each of 4 types is an independent yes/no question
#         # NO sigmoid here — BCEWithLogitsLoss handles it
#         self.type_head = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, 4)
#         )

#         # ── Severity head — single-label ─────────────────────────────────────
#         # Only one severity level is true at a time
#         # NO softmax here — CrossEntropyLoss handles it
#         self.severity_head = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, 5)
#         )

#     def unfreeze_backbone(self):
#         """Call this after initial training to fine-tune the full model."""
#         for param in self.features.parameters():
#             param.requires_grad = True
#         print("Backbone unfrozen — full fine-tuning enabled")

#     def forward(self, x: torch.Tensor):
#         """
#         Args:
#             x : (batch, 3, 224, 224) normalized image tensor

#         Returns:
#             type_logits     : (batch, 4)
#             severity_logits : (batch, 5)
#         """
#         # Extract shared features
#         feat = self.features(x)           # (batch, 576, 7, 7)
#         feat = self.avgpool(feat)         # (batch, 576, 1, 1)
#         feat = feat.flatten(1)            # (batch, 576)

#         type_logits     = self.type_head(feat)      # (batch, 4)
#         severity_logits = self.severity_head(feat)  # (batch, 5)

#         return type_logits, severity_logits

#     def predict(self, x: torch.Tensor, threshold: float = 0.5):
#         """
#         Inference-time prediction with human-readable output.

#         Returns:
#             predicted_types    : list of type names present
#             predicted_severity : severity level name
#             type_probs         : (4,) sigmoid probabilities
#             severity_probs     : (5,) softmax probabilities
#         """
#         self.eval()
#         with torch.no_grad():
#             type_logits, severity_logits = self.forward(x)

#             # Type: sigmoid → threshold
#             type_probs = torch.sigmoid(type_logits[0])
#             predicted_types = [
#                 TYPES[i] for i, p in enumerate(type_probs)
#                 if p.item() > threshold
#             ]
#             # Fallback: if nothing above threshold, pick highest probability
#             if not predicted_types:
#                 predicted_types = [TYPES[type_probs.argmax().item()]]

#             # Severity: softmax → argmax
#             severity_probs = torch.softmax(severity_logits[0], dim=0)
#             severity_idx = severity_probs.argmax().item()
#             predicted_severity = LEVELS[severity_idx]

#         return predicted_types, predicted_severity, type_probs, severity_probs


# def load_model(model_path: str, device: torch.device) -> DegradationEstimator:
#     """Loads a trained model from disk."""
#     model = DegradationEstimator(freeze_backbone=False)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()
#     print(f"Loaded model from {model_path}")
#     return model


"""
model.py
--------
Degradation Estimator — dual-head CNN classifier.
 
Architecture:
    Shared backbone : MobileNetV3-Small (pretrained on ImageNet)
    Head 1          : Type head    — sigmoid, 4 outputs (multi-label)
    Head 2          : Severity head — softmax, 5 outputs (single-label)
 
Why MobileNetV3-Small:
    - Lightweight — trains fast on M3
    - Pretrained — already knows edges and textures
    - 576-dim feature vector — enough for this task
"""
 
import torch
import torch.nn as nn
import torchvision.models as models
 
 
TYPES  = ['upsample', 'denoise', 'deartifact', 'inpaint']
LEVELS = ['XS', 'S', 'M', 'L', 'XL']
 
 
class DegradationEstimator(nn.Module):
    """
    Dual-head classifier for degradation type and severity.
 
    Forward returns:
        type_logits     : (batch, 4)  — raw logits, no sigmoid
        severity_logits : (batch, 5)  — raw logits, no softmax
 
    Both activations are applied inside the loss functions:
        type_logits     → BCEWithLogitsLoss (handles sigmoid internally)
        severity_logits → CrossEntropyLoss  (handles softmax internally)
    """
 
    def __init__(self, dropout_rate: float = 0.3, freeze_backbone: bool = True):
        super().__init__()
 
        # ── Backbone ────────────────────────────────────────────────────────
        backbone = models.mobilenet_v3_small(weights='DEFAULT')
 
        # Extract feature extractor and pooling — remove classifier
        self.features = backbone.features   # CNN feature extractor
        self.avgpool  = backbone.avgpool    # Global average pool → (batch, 576, 1, 1)
        feature_dim   = 576
 
        # Optionally freeze backbone for first training phase
        # (train heads only → prevents early overfitting)
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
 
        # ── Type head — multi-label ──────────────────────────────────────────
        # Each of 4 types is an independent yes/no question
        # NO sigmoid here — BCEWithLogitsLoss handles it
        self.type_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4)
        )
 
        # ── Severity head — single-label ─────────────────────────────────────
        # Only one severity level is true at a time
        # NO softmax here — CrossEntropyLoss handles it
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 5)
        )
 
    def unfreeze_backbone(self):
        """Call this after initial training to fine-tune the full model."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — full fine-tuning enabled")
 
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (batch, 3, 224, 224) normalized image tensor
 
        Returns:
            type_logits     : (batch, 4)
            severity_logits : (batch, 5)
        """
        # Extract shared features
        feat = self.features(x)           # (batch, 576, 7, 7)
        feat = self.avgpool(feat)         # (batch, 576, 1, 1)
        feat = feat.flatten(1)            # (batch, 576)
 
        type_logits     = self.type_head(feat)      # (batch, 4)
        severity_logits = self.severity_head(feat)  # (batch, 5)
 
        return type_logits, severity_logits
 
    def predict(self, x: torch.Tensor, threshold: float = 0.5):
        """
        Inference-time prediction with human-readable output.
 
        Returns:
            predicted_types    : list of type names present
            predicted_severity : severity level name
            type_probs         : (4,) sigmoid probabilities
            severity_probs     : (5,) softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            type_logits, severity_logits = self.forward(x)
 
            # Type: sigmoid → threshold
            type_probs = torch.sigmoid(type_logits[0])
            predicted_types = [
                TYPES[i] for i, p in enumerate(type_probs)
                if p.item() > threshold
            ]
            # Fallback: if nothing above threshold, pick highest probability
            if not predicted_types:
                predicted_types = [TYPES[type_probs.argmax().item()]]
 
            # Severity: softmax → argmax
            severity_probs = torch.softmax(severity_logits[0], dim=0)
            severity_idx = severity_probs.argmax().item()
            predicted_severity = LEVELS[severity_idx]
 
        return predicted_types, predicted_severity, type_probs, severity_probs
 
 
def load_model(model_path: str, device: torch.device) -> DegradationEstimator:
    """Loads a trained model from disk."""
    model = DegradationEstimator(freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")
    return model
 
 
# ═══════════════════════════════════════════════════════════════════════════
# NEW — Specialist Models (Phase 2 improvement)
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY: The shared severity head in DegradationEstimator above tries to learn
# severity for ALL 4 degradation types at once. This is hard because:
#   - "mild noise" looks completely different from "mild blur"
#   - "heavy noise" looks completely different from "heavy blur"
#   - One shared head cannot learn all these distinct patterns well
#
# FIX: Train 4 separate specialist models — one per degradation type.
# Each specialist only sees images of its own type, so it learns the
# subtle visual differences between severity levels much more accurately.
#
# HOW IT WORKS IN PRACTICE:
#   Step 1: DegradationEstimator (existing) predicts TYPE  → e.g. "denoise"
#   Step 2: Route to the matching specialist               → denoise specialist
#   Step 3: Specialist predicts SEVERITY                   → e.g. "M"
#
# Each specialist is a simpler model — just one head predicting 5 classes.
# The backbone is still pretrained MobileNetV3 but only the severity head
# is attached. No type head needed since we already know the type.
# ═══════════════════════════════════════════════════════════════════════════
 
class SeveritySpecialist(nn.Module):
    """
    Specialist model for severity estimation of ONE degradation type.
 
    Simpler than DegradationEstimator — only predicts severity (5 classes).
    No type head needed since we already know the type from the main model.
 
    One instance is trained per degradation type:
        specialist_upsample   → trained only on blurry images
        specialist_denoise    → trained only on noisy images
        specialist_deartifact → trained only on JPEG-compressed images
        specialist_inpaint    → trained only on masked images
 
    Expected accuracy improvement:
        Shared model severity:    ~62%
        Specialist model severity: ~85%+
    """
 
    def __init__(self, deg_type: str, dropout_rate: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()
 
        # Which degradation type this specialist handles
        # Stored so we know which images to train/test on
        self.deg_type = deg_type
 
        # Same backbone as main model — pretrained MobileNetV3-Small
        # This ensures fair comparison: same feature extractor, different head
        backbone = models.mobilenet_v3_small(weights='DEFAULT')
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        feature_dim   = 576
 
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
 
        # Only ONE head — severity only (5 classes: XS, S, M, L, XL)
        # Simpler than the shared model which has two heads
        # More capacity can go into getting severity right for this one type
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 5)   # 5 severity levels: XS S M L XL
        )
 
    def unfreeze_backbone(self):
        """Call after initial training to fine-tune full model."""
        for param in self.features.parameters():
            param.requires_grad = True
        print(f"[{self.deg_type} specialist] Backbone unfrozen")
 
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (batch, 3, 224, 224) normalized image tensor
                Must be images of this specialist's degradation type only
 
        Returns:
            severity_logits : (batch, 5) — raw logits, no softmax
        """
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.flatten(1)            # (batch, 576)
        return self.severity_head(feat)   # (batch, 5)
 
    def predict_severity(self, x: torch.Tensor) -> tuple:
        """
        Predicts severity level for one image.
 
        Returns:
            predicted_severity : severity level name e.g. 'M'
            severity_probs     : (5,) softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            severity_logits = self.forward(x)
            severity_probs  = torch.softmax(severity_logits[0], dim=0)
            severity_idx    = severity_probs.argmax().item()
            predicted_severity = LEVELS[severity_idx]
        return predicted_severity, severity_probs
 
 
def load_specialist(model_path: str, deg_type: str,
                    device: torch.device) -> SeveritySpecialist:
    """Loads a trained specialist model from disk."""
    model = SeveritySpecialist(deg_type=deg_type, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded [{deg_type}] specialist from {model_path}")
    return model
 
 
def load_all_specialists(specialists_dir: str,
                          device: torch.device) -> dict:
    """
    Loads all 4 specialist models at once.
 
    Args:
        specialists_dir : folder containing the 4 .pth files
                          e.g. 'specialists/'
                          Expected files:
                            specialists/upsample_specialist.pth
                            specialists/denoise_specialist.pth
                            specialists/deartifact_specialist.pth
                            specialists/inpaint_specialist.pth
 
    Returns:
        dict mapping type name → loaded specialist model
        e.g. {'upsample': model, 'denoise': model, ...}
    """
    specialists = {}
    for deg_type in TYPES:
        path = os.path.join(specialists_dir, f"{deg_type}_specialist.pth")
        if os.path.exists(path):
            specialists[deg_type] = load_specialist(path, deg_type, device)
        else:
            print(f"Warning: specialist not found at {path}")
    return specialists
 
 
# Need os for load_all_specialists
import os