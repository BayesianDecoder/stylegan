"""
generate_dataset.py
-------------------
Creates the labeled training dataset for the Blind Degradation Estimator.
Uses the paper's own degradation logic — labels are automatic.

Usage:
    python generate_dataset.py --input_dir data/ffhq --output_dir data/degraded --num_images 1000

Output:
    data/degraded/
        images/          <- all degraded images
        labels.csv       <- filename, type_label, severity_label
"""

import os
import io
import csv
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Degradation type and level definitions ──────────────────────────────────

TYPES  = ['upsample', 'denoise', 'deartifact', 'inpaint']
LEVELS = ['XS', 'S', 'M', 'L', 'XL']

# Parameters taken directly from Section 5.1 of the paper
PARAMS = {
    'upsample': {
        'XS': {'factor': 2},
        'S':  {'factor': 4},
        'M':  {'factor': 8},
        'L':  {'factor': 16},
        'XL': {'factor': 32},
    },
    'denoise': {
        'XS': {'k_p': 96,  'k_b': 0.04},
        'S':  {'k_p': 48,  'k_b': 0.08},
        'M':  {'k_p': 24,  'k_b': 0.16},
        'L':  {'k_p': 12,  'k_b': 0.32},
        'XL': {'k_p': 6,   'k_b': 0.64},
    },
    'deartifact': {
        'XS': {'quality': 18},
        'S':  {'quality': 15},
        'M':  {'quality': 12},
        'L':  {'quality': 9},
        'XL': {'quality': 6},
    },
    'inpaint': {
        'XS': {'strokes': 1},
        'S':  {'strokes': 5},
        'M':  {'strokes': 9},
        'L':  {'strokes': 13},
        'XL': {'strokes': 17},
    },
}

# ── Individual degradation functions ────────────────────────────────────────

def apply_upsample(img: Image.Image, factor: int) -> Image.Image:
    """
    Simulates a low-resolution image by downsampling then upsampling.
    This is the degradation the paper tries to invert (super-resolution).
    """
    w, h = img.size
    # Downsample to small size using random filter (bilinear/bicubic/lanczos)
    filter_choice = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
    small = img.resize((w // factor, h // factor), filter_choice)
    # Upsample back — this is what the network receives as input y
    return small.resize((w, h), Image.BILINEAR)


def apply_denoise(img: Image.Image, k_p: float, k_b: float) -> Image.Image:
    """
    Applies Poisson + Bernoulli noise — simulates camera sensor noise.
    Poisson = shot noise (signal-dependent)
    Bernoulli = dead/hot pixels (random black pixels)
    
    NOTE: These are the non-differentiable real degradations.
    The paper uses Gaussian approximations of these during optimization.
    Here we apply the REAL noise since we just need a noisy image for training.
    """
    arr = np.array(img).astype(np.float32) / 255.0

    # Poisson noise — randomize k_p slightly for robustness
    k_p_jitter = k_p * random.uniform(0.8, 1.2)
    noisy = np.random.poisson(k_p_jitter * arr) / k_p_jitter
    noisy = np.clip(noisy, 0, 1)

    # Bernoulli noise — dead pixels
    k_b_jitter = k_b * random.uniform(0.8, 1.2)
    dead_mask = np.random.binomial(1, k_b_jitter, arr.shape[:2])
    noisy[dead_mask == 1] = 0

    # Clamp and convert back
    noisy = np.clip(noisy, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def apply_deartifact(img: Image.Image, quality: int) -> Image.Image:
    """
    Applies JPEG compression artifacts by saving/loading at low quality.
    Simulates heavily compressed images from social media or old cameras.
    """
    buf = io.BytesIO()
    # Jitter quality slightly for robustness
    q = max(1, quality + random.randint(-2, 2))
    img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_inpaint(img: Image.Image, strokes: int) -> Image.Image:
    """
    Applies random brush stroke masks — simulates missing regions.
    Each stroke connects two random points in the outer thirds of the image.
    """
    arr = np.array(img).copy()
    w, h = img.size

    for _ in range(strokes):
        # Random stroke connecting two points in outer thirds
        x1 = random.randint(0, w // 3)
        x2 = random.randint(2 * w // 3, w - 1)
        y1 = random.randint(h // 4, 3 * h // 4)
        y2 = random.randint(h // 4, 3 * h // 4)

        # Stroke thickness = 8% of image size (from paper)
        thickness = max(1, int(0.08 * min(w, h)))

        # Draw the stroke as a filled rectangle between the two points
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min = max(0, min(y1, y2) - thickness // 2)
        y_max = min(h, max(y1, y2) + thickness // 2)
        arr[y_min:y_max, x_min:x_max] = 0

    return Image.fromarray(arr)


def apply_degradation(img: Image.Image, deg_type: str, level: str) -> Image.Image:
    """Applies one degradation type at one severity level."""
    params = PARAMS[deg_type][level]
    if deg_type == 'upsample':
        return apply_upsample(img, **params)
    elif deg_type == 'denoise':
        return apply_denoise(img, **params)
    elif deg_type == 'deartifact':
        return apply_deartifact(img, **params)
    elif deg_type == 'inpaint':
        return apply_inpaint(img, **params)
    else:
        raise ValueError(f"Unknown degradation type: {deg_type}")


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(input_dir: str, output_dir: str, num_images: int):
    """
    Main dataset generation loop.
    
    For each clean image, applies all 4 types × 5 levels = 20 degraded versions.
    Total images = num_images × 20
    Labels are automatic — no human annotation needed.
    """
    img_save_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_save_dir, exist_ok=True)

    # Collect all image files from input directory
    all_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:num_images]

    print(f"Found {len(all_files)} clean images")
    print(f"Will generate {len(all_files) * 20} degraded images")
    print(f"Saving to: {output_dir}")

    records = []  # stores (filename, type_idx, severity_idx)

    for img_file in tqdm(all_files, desc="Processing images"):
        # Load clean image
        clean_path = os.path.join(input_dir, img_file)
        try:
            clean_img = Image.open(clean_path).convert('RGB')
        except Exception as e:
            print(f"  Skipping {img_file}: {e}")
            continue

        # Resize to 256x256 for faster training
        # (paper uses 1024x1024 but 256x256 is fine for classification)
        clean_img = clean_img.resize((256, 256), Image.LANCZOS)

        img_stem = os.path.splitext(img_file)[0]

        for type_idx, deg_type in enumerate(TYPES):
            for level_idx, level in enumerate(LEVELS):
                # Apply degradation
                degraded = apply_degradation(clean_img, deg_type, level)

                # Save with structured filename encoding the label
                # Format: {original_name}_t{type_idx}_s{level_idx}.png
                # Example: 00001_t1_s2.png = image 1, denoise, level M
                save_name = f"{img_stem}_t{type_idx}_s{level_idx}.png"
                save_path = os.path.join(img_save_dir, save_name)
                degraded.save(save_path)

                records.append((save_name, type_idx, level_idx))

    # Save label CSV
    csv_path = os.path.join(output_dir, 'labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'type_label', 'severity_label'])
        writer.writerows(records)

    print(f"\nDone! Generated {len(records)} images")
    print(f"Labels saved to: {csv_path}")
    print(f"\nLabel mapping:")
    for i, t in enumerate(TYPES):
        print(f"  type {i} = {t}")
    for i, l in enumerate(LEVELS):
        print(f"  severity {i} = {l}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate degraded dataset')
    parser.add_argument('--input_dir',   type=str, required=True,  help='Directory of clean FFHQ images')
    parser.add_argument('--output_dir',  type=str, required=True,  help='Where to save degraded images + labels')
    parser.add_argument('--num_images',  type=int, default=1000,   help='How many clean images to use')
    args = parser.parse_args()

    generate_dataset(args.input_dir, args.output_dir, args.num_images)
