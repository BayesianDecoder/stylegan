# """
# predict.py
# ----------
# Runs the trained CNN on a new degraded image and prints the prediction.
# Also shows how to connect this to the paper's run.py pipeline.

# Usage:
#     python predict.py --image path/to/degraded_image.png --model_path best_model.pth
# """

# import argparse
# import torch
# from PIL import Image
# from torchvision import transforms

# from model import load_model, TYPES, LEVELS

# # ── Image preprocessing (same as val_transform in dataset.py) ───────────────
# TRANSFORM = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# # Paper's fixed composition order from Equation 16
# FIXED_ORDER = ['upsample', 'denoise', 'deartifact', 'inpaint']


# def predict_single_image(image_path: str, model, device: torch.device,
#                           threshold: float = 0.5):
#     """
#     Runs CNN on one image and returns the predicted degradation info.

#     Returns:
#         predicted_types    : list of type names e.g. ['denoise']
#         predicted_severity : severity level name e.g. 'M'
#         ordered_types      : types sorted in paper's fixed order
#         type_probs         : (4,) tensor of probabilities per type
#         severity_probs     : (5,) tensor of probabilities per level
#     """
#     # Load and preprocess image
#     img = Image.open(image_path).convert('RGB')
#     x   = TRANSFORM(img).unsqueeze(0).to(device)  # add batch dim → (1, 3, 224, 224)

#     # Run CNN
#     predicted_types, predicted_severity, type_probs, severity_probs = \
#         model.predict(x, threshold=threshold)

#     # Sort predicted types in paper's fixed order (Equation 16)
#     ordered_types = [t for t in FIXED_ORDER if t in predicted_types]

#     return predicted_types, predicted_severity, ordered_types, type_probs, severity_probs


# def build_run_command(ordered_types: list, predicted_severity: str) -> str:
#     """
#     Builds the equivalent paper run.py command from CNN predictions.
#     This shows exactly what the human would have had to type manually.

#     Example output:
#         python run.py --tasks denoise_upsampling --level M --image my_image.png
#     """
#     # Map type names to paper's task argument names
#     task_name_map = {
#         'upsample':   'upsampling',
#         'denoise':    'denoising',
#         'deartifact': 'deartifacting',
#         'inpaint':    'inpainting',
#     }
#     task_args = '_'.join(task_name_map[t] for t in ordered_types)
#     return f"python run.py --tasks {task_args} --level {predicted_severity}"


# def print_prediction_report(image_path, predicted_types, predicted_severity,
#                              ordered_types, type_probs, severity_probs):
#     """Prints a clean human-readable prediction report."""
#     print("\n" + "="*55)
#     print("BLIND DEGRADATION ESTIMATION RESULT")
#     print("="*55)
#     print(f"Image: {image_path}")
#     print()

#     print("Type probabilities:")
#     for i, t in enumerate(TYPES):
#         bar_len = int(type_probs[i].item() * 20)
#         bar = "█" * bar_len + "░" * (20 - bar_len)
#         detected = " ← DETECTED" if t in predicted_types else ""
#         print(f"  {t:12s} [{bar}] {type_probs[i].item():.3f}{detected}")

#     print()
#     print("Severity probabilities:")
#     for i, l in enumerate(LEVELS):
#         bar_len = int(severity_probs[i].item() * 20)
#         bar = "█" * bar_len + "░" * (20 - bar_len)
#         detected = " ← PREDICTED" if i == severity_probs.argmax().item() else ""
#         print(f"  {l:4s} [{bar}] {severity_probs[i].item():.3f}{detected}")

#     print()
#     print(f"Final prediction:")
#     print(f"  Types    : {predicted_types}")
#     print(f"  Severity : {predicted_severity}")
#     print(f"  Order    : {ordered_types} (paper's fixed order)")
#     print()

#     cmd = build_run_command(ordered_types, predicted_severity)
#     print("Equivalent manual command (what human would have typed):")
#     print(f"  {cmd}")
#     print()
#     print("With your CNN, this command is now built automatically.")
#     print("="*55)


# # ── How to connect to paper's run.py ────────────────────────────────────────
# """
# INTEGRATION GUIDE
# -----------------
# In the paper's run.py, find the YourDegradation class:

#     class YourDegradation:
#         def degrade_ground_truth(self, x):
#             raise NotImplementedError
#         def degrade_prediction(self, x):
#             raise NotImplementedError

# Replace it with this pattern:

#     from predict import predict_single_image, build_degradation_fn
#     from model import load_model

#     device = torch.device("mps")
#     estimator = load_model("best_model.pth", device)

#     # Run CNN on the input image
#     image_path = args.image_path
#     types, severity, ordered_types, _, _ = predict_single_image(
#         image_path, estimator, device
#     )

#     # Build f̂ automatically
#     class AutoDegradation:
#         def degrade_ground_truth(self, x):
#             return x  # identity — we don't have ground truth
#         def degrade_prediction(self, x):
#             return build_degradation_fn(ordered_types, severity)(x)

# Then pass --tasks custom to run.py and it will use AutoDegradation.
# """


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Predict degradation from image')
#     parser.add_argument('--image',      type=str, required=True,           help='Path to degraded image')
#     parser.add_argument('--model_path', type=str, default='best_model.pth', help='Trained model path')
#     parser.add_argument('--threshold',  type=float, default=0.5,           help='Type detection threshold')
#     args = parser.parse_args()

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     model  = load_model(args.model_path, device)

#     predicted_types, predicted_severity, ordered_types, type_probs, severity_probs = \
#         predict_single_image(args.image, model, device, args.threshold)

#     print_prediction_report(
#         args.image, predicted_types, predicted_severity,
#         ordered_types, type_probs, severity_probs
#     )


"""
predict.py
----------
Runs the trained CNN on a new degraded image and prints the prediction.
Also shows how to connect this to the paper's run.py pipeline.
 
Usage:
    python predict.py --image path/to/degraded_image.png --model_path best_model.pth
"""
 
import argparse
import torch
from PIL import Image
from torchvision import transforms
 
from model import load_model, TYPES, LEVELS
 
# ── Image preprocessing (same as val_transform in dataset.py) ───────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
 
# Paper's fixed composition order from Equation 16
FIXED_ORDER = ['upsample', 'denoise', 'deartifact', 'inpaint']
 
 
def predict_single_image(image_path: str, model, device: torch.device,
                          threshold: float = 0.5):
    """
    Runs CNN on one image and returns the predicted degradation info.
 
    Returns:
        predicted_types    : list of type names e.g. ['denoise']
        predicted_severity : severity level name e.g. 'M'
        ordered_types      : types sorted in paper's fixed order
        type_probs         : (4,) tensor of probabilities per type
        severity_probs     : (5,) tensor of probabilities per level
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    x   = TRANSFORM(img).unsqueeze(0).to(device)  # add batch dim → (1, 3, 224, 224)
 
    # Run CNN
    predicted_types, predicted_severity, type_probs, severity_probs = \
        model.predict(x, threshold=threshold)
 
    # Sort predicted types in paper's fixed order (Equation 16)
    ordered_types = [t for t in FIXED_ORDER if t in predicted_types]
 
    return predicted_types, predicted_severity, ordered_types, type_probs, severity_probs
 
 
def build_run_command(ordered_types: list, predicted_severity: str) -> str:
    """
    Builds the equivalent paper run.py command from CNN predictions.
    This shows exactly what the human would have had to type manually.
 
    Example output:
        python run.py --tasks denoise_upsampling --level M --image my_image.png
    """
    # Map type names to paper's task argument names
    task_name_map = {
        'upsample':   'upsampling',
        'denoise':    'denoising',
        'deartifact': 'deartifacting',
        'inpaint':    'inpainting',
    }
    task_args = '_'.join(task_name_map[t] for t in ordered_types)
    return f"python run.py --tasks {task_args} --level {predicted_severity}"
 
 
def print_prediction_report(image_path, predicted_types, predicted_severity,
                             ordered_types, type_probs, severity_probs):
    """Prints a clean human-readable prediction report."""
    print("\n" + "="*55)
    print("BLIND DEGRADATION ESTIMATION RESULT")
    print("="*55)
    print(f"Image: {image_path}")
    print()
 
    print("Type probabilities:")
    for i, t in enumerate(TYPES):
        bar_len = int(type_probs[i].item() * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        detected = " ← DETECTED" if t in predicted_types else ""
        print(f"  {t:12s} [{bar}] {type_probs[i].item():.3f}{detected}")
 
    print()
    print("Severity probabilities:")
    for i, l in enumerate(LEVELS):
        bar_len = int(severity_probs[i].item() * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        detected = " ← PREDICTED" if i == severity_probs.argmax().item() else ""
        print(f"  {l:4s} [{bar}] {severity_probs[i].item():.3f}{detected}")
 
    print()
    print(f"Final prediction:")
    print(f"  Types    : {predicted_types}")
    print(f"  Severity : {predicted_severity}")
    print(f"  Order    : {ordered_types} (paper's fixed order)")
    print()
 
    cmd = build_run_command(ordered_types, predicted_severity)
    print("Equivalent manual command (what human would have typed):")
    print(f"  {cmd}")
    print()
    print("With your CNN, this command is now built automatically.")
    print("="*55)
 
 
# ── How to connect to paper's run.py ────────────────────────────────────────
"""
INTEGRATION GUIDE
-----------------
In the paper's run.py, find the YourDegradation class:
 
    class YourDegradation:
        def degrade_ground_truth(self, x):
            raise NotImplementedError
        def degrade_prediction(self, x):
            raise NotImplementedError
 
Replace it with this pattern:
 
    from predict import predict_single_image, build_degradation_fn
    from model import load_model
 
    device = torch.device("mps")
    estimator = load_model("best_model.pth", device)
 
    # Run CNN on the input image
    image_path = args.image_path
    types, severity, ordered_types, _, _ = predict_single_image(
        image_path, estimator, device
    )
 
    # Build f̂ automatically
    class AutoDegradation:
        def degrade_ground_truth(self, x):
            return x  # identity — we don't have ground truth
        def degrade_prediction(self, x):
            return build_degradation_fn(ordered_types, severity)(x)
 
Then pass --tasks custom to run.py and it will use AutoDegradation.
"""
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict degradation from image')
    parser.add_argument('--image',      type=str, required=True,           help='Path to degraded image')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Trained model path')
    parser.add_argument('--threshold',  type=float, default=0.5,           help='Type detection threshold')
    args = parser.parse_args()
 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = load_model(args.model_path, device)
 
    predicted_types, predicted_severity, ordered_types, type_probs, severity_probs = \
        predict_single_image(args.image, model, device, args.threshold)
 
    print_prediction_report(
        args.image, predicted_types, predicted_severity,
        ordered_types, type_probs, severity_probs
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════
# NEW — Improved prediction using specialist models
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY: The main model predicts type very accurately (99.9%) but severity
# only moderately (62.2%). Once we know the type, we can route to the
# matching specialist which predicts severity much better (~85%+).
#
# TWO-STAGE PIPELINE:
#   Stage 1: Main model   → predicts TYPE  (99.9% accurate)
#   Stage 2: Specialist   → predicts SEVERITY for that type (~85% accurate)
#
# This is the improved version shown at mid evaluation.
# ═══════════════════════════════════════════════════════════════════════════
 
def predict_with_specialists(image_path: str,
                               main_model,
                               specialists: dict,
                               device: torch.device,
                               threshold: float = 0.5):
    """
    Two-stage prediction using main model + specialist.
 
    Stage 1: Main model predicts which degradation TYPE is present.
    Stage 2: Matching specialist predicts SEVERITY for that type.
 
    Args:
        image_path  : path to the degraded image
        main_model  : loaded DegradationEstimator (for type prediction)
        specialists : dict of {type_name: SeveritySpecialist}
                      loaded by load_all_specialists()
        device      : torch device
 
    Returns:
        predicted_types    : list of type names from main model
        predicted_severity : severity from specialist (more accurate)
        ordered_types      : types in paper's fixed order
        type_probs         : (4,) type probabilities from main model
        severity_probs     : (5,) severity probabilities from specialist
        used_specialist    : True if specialist was used, False if fallback
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    x   = TRANSFORM(img).unsqueeze(0).to(device)
 
    # ── Stage 1: Main model predicts TYPE ──────────────────────────────────
    predicted_types, _, ordered_types, type_probs, _ = \
        predict_single_image(image_path, main_model, device, threshold)
 
    # ── Stage 2: Specialist predicts SEVERITY ──────────────────────────────
    # Use the first predicted type to route to the right specialist
    # (In single-degradation case there is only one type anyway)
    primary_type   = predicted_types[0] if predicted_types else 'denoise'
    used_specialist = False
 
    if primary_type in specialists:
        # Route to the matching specialist
        specialist = specialists[primary_type]
        predicted_severity, severity_probs = specialist.predict_severity(x)
        used_specialist = True
    else:
        # Fallback: use main model's severity prediction
        # (happens if specialist not trained yet for this type)
        _, predicted_severity, _, severity_probs = main_model.predict(
            x, threshold
        )
        print(f"  Warning: no specialist for '{primary_type}', "
              f"using shared model severity")
 
    return (predicted_types, predicted_severity, ordered_types,
            type_probs, severity_probs, used_specialist)
 
 
def print_specialist_report(image_path, predicted_types, predicted_severity,
                              ordered_types, type_probs, severity_probs,
                              used_specialist):
    """Prints prediction report showing which stage provided each prediction."""
    print("\n" + "="*60)
    print("IMPROVED PREDICTION (main model + specialist)")
    print("="*60)
    print(f"Image: {image_path}")
    print()

    print("Stage 1 — Type prediction (main model, 99.9% accurate):")
    for i, t in enumerate(TYPES):
        bar_len  = int(type_probs[i].item() * 20)
        bar      = "█" * bar_len + "░" * (20 - bar_len)
        detected = " ← DETECTED" if t in predicted_types else ""
        print(f"  {t:12s} [{bar}] {type_probs[i].item():.3f}{detected}")

    source = "specialist model" if used_specialist else "shared model (fallback)"
    print(f"\nStage 2 — Severity prediction ({source}, ~85% accurate):")
    for i, l in enumerate(LEVELS):
        bar_len  = int(severity_probs[i].item() * 20)
        bar      = "█" * bar_len + "░" * (20 - bar_len)
        detected = " ← PREDICTED" if i == severity_probs.argmax().item() else ""
        print(f"  {l:4s} [{bar}] {severity_probs[i].item():.3f}{detected}")

    print(f"\nFinal prediction:")
    print(f"  Types    : {predicted_types}  [from main model]")
    print(f"  Severity : {predicted_severity}  [from {source}]")
    print(f"  Order    : {ordered_types}")
    print()

    cmd = build_run_command(ordered_types, predicted_severity)
    print(f"Equivalent manual command:")
    print(f"  {cmd}")
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════
# build_degradation_fn — bridges CNN predictions to the benchmark API
# ═══════════════════════════════════════════════════════════════════════════
#
# Maps CNN type names → benchmark task names, looks up the correct
# degradation parameters for the predicted severity level, and returns
# a ComposedDegradation ready for the run.py optimization loop.
#
# Used by auto_restore.py and the integration pattern described above.
# ═══════════════════════════════════════════════════════════════════════════

# CNN type names → benchmark task names (paper's terminology)
_TYPE_TO_TASK = {
    'upsample':   'upsampling',
    'denoise':    'denoising',
    'deartifact': 'deartifacting',
    'inpaint':    'inpainting',
}


def build_degradation_fn(ordered_types: list, severity: str, resolution: int = None):
    """
    Constructs a Degradation object from CNN predictions.

    Wraps the predicted degradation(s) in ResizePrediction so the generator's
    output is always resized to match the target before comparison.

    Args:
        ordered_types : CNN type names in paper's fixed composition order
                        e.g. ['denoise'] or ['upsample', 'denoise']
        severity      : severity level string e.g. 'M'
        resolution    : output resolution; defaults to benchmark.config.resolution

    Returns:
        ComposedDegradation ready for the 3-phase optimization loop
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stylegan2_ada"))

    import benchmark.config as bench_config
    from benchmark.tasks import (
        degradation_types, degradation_levels,
        ComposedDegradation, ResizePrediction,
    )

    if resolution is None:
        resolution = bench_config.resolution

    task_names   = [_TYPE_TO_TASK[t] for t in ordered_types]
    degradations = [ResizePrediction(resolution)]
    for task_name in task_names:
        deg_cls = degradation_types[task_name]
        deg_arg = degradation_levels[task_name][severity]
        degradations.append(deg_cls(deg_arg))

    return ComposedDegradation(degradations)