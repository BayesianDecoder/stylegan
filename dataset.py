# """
# dataset.py
# ----------
# PyTorch Dataset for loading the generated degraded images.

# Handles train / val / test splits automatically.
# Applies augmentation only during training.
# """

# import os
# import pandas as pd
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms


# TYPES  = ['upsample', 'denoise', 'deartifact', 'inpaint']
# LEVELS = ['XS', 'S', 'M', 'L', 'XL']


# class DegradationDataset(Dataset):
#     """
#     Loads degraded images and their type + severity labels.

#     type_label  : multi-hot vector of shape (4,)
#                   e.g. [0, 1, 0, 0] = denoise only
#                   Uses BCEWithLogitsLoss during training

#     severity_label : single integer 0-4
#                      Uses CrossEntropyLoss during training
#     """

#     def __init__(self, data_dir: str, split: str = 'train'):
#         """
#         Args:
#             data_dir : path to folder containing images/ and labels.csv
#             split    : 'train' (80%), 'val' (10%), or 'test' (10%)
#         """
#         self.img_dir = os.path.join(data_dir, 'images')

#         # Load labels CSV
#         csv_path = os.path.join(data_dir, 'labels.csv')
#         df = pd.read_csv(csv_path)
#         df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

#         # Split
#         n = len(df)
#         if split == 'train':
#             self.df = df.iloc[:int(0.8 * n)]
#         elif split == 'val':
#             self.df = df.iloc[int(0.8 * n):int(0.9 * n)]
#         else:
#             self.df = df.iloc[int(0.9 * n):]

#         self.df = self.df.reset_index(drop=True)

#         # ── Transforms ──────────────────────────────────────────────────────
#         # Training: augmentation to prevent overfitting
#         # Val/Test: only resize and normalize — no augmentation
#         if split == 'train':
#             self.transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.ColorJitter(
#                     brightness=0.15,
#                     contrast=0.15,
#                     saturation=0.1
#                 ),
#                 transforms.ToTensor(),
#                 # ImageNet normalization — works well with pretrained MobileNetV3
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 ),
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 ),
#             ])

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]

#         # Load image
#         img_path = os.path.join(self.img_dir, row['filename'])
#         image = Image.open(img_path).convert('RGB')
#         image = self.transform(image)

#         # type_label: multi-hot vector for BCEWithLogitsLoss
#         # Example: type_idx=1 (denoise) → [0.0, 1.0, 0.0, 0.0]
#         type_vec = torch.zeros(4, dtype=torch.float32)
#         type_vec[int(row['type_label'])] = 1.0

#         # severity_label: single integer for CrossEntropyLoss
#         severity = torch.tensor(int(row['severity_label']), dtype=torch.long)

#         return image, type_vec, severity


# def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
#     """Creates train, val, and test DataLoaders."""
#     train_ds = DegradationDataset(data_dir, split='train')
#     val_ds   = DegradationDataset(data_dir, split='val')
#     test_ds  = DegradationDataset(data_dir, split='test')

#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size,
#         shuffle=True, num_workers=num_workers, pin_memory=False
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=batch_size,
#         shuffle=False, num_workers=num_workers, pin_memory=False
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=batch_size,
#         shuffle=False, num_workers=num_workers, pin_memory=False
#     )

#     print(f"Dataset sizes — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
#     return train_loader, val_loader, test_loader




"""
dataset.py
----------
PyTorch Dataset for loading the generated degraded images.
 
Handles train / val / test splits automatically.
Applies augmentation only during training.
"""
 
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
 
 
TYPES  = ['upsample', 'denoise', 'deartifact', 'inpaint']
LEVELS = ['XS', 'S', 'M', 'L', 'XL']
 
 
class DegradationDataset(Dataset):
    """
    Loads degraded images and their type + severity labels.
 
    type_label  : multi-hot vector of shape (4,)
                  e.g. [0, 1, 0, 0] = denoise only
                  Uses BCEWithLogitsLoss during training
 
    severity_label : single integer 0-4
                     Uses CrossEntropyLoss during training
    """
 
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir : path to folder containing images/ and labels.csv
            split    : 'train' (80%), 'val' (10%), or 'test' (10%)
        """
        self.img_dir = os.path.join(data_dir, 'images')
 
        # Load labels CSV
        csv_path = os.path.join(data_dir, 'labels.csv')
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
 
        # Split
        n = len(df)
        if split == 'train':
            self.df = df.iloc[:int(0.8 * n)]
        elif split == 'val':
            self.df = df.iloc[int(0.8 * n):int(0.9 * n)]
        else:
            self.df = df.iloc[int(0.9 * n):]
 
        self.df = self.df.reset_index(drop=True)
 
        # ── Transforms ──────────────────────────────────────────────────────
        # Training: augmentation to prevent overfitting
        # Val/Test: only resize and normalize — no augmentation
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1
                ),
                transforms.ToTensor(),
                # ImageNet normalization — works well with pretrained MobileNetV3
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
 
        # Load image
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
 
        # type_label: multi-hot vector for BCEWithLogitsLoss
        # Example: type_idx=1 (denoise) → [0.0, 1.0, 0.0, 0.0]
        type_vec = torch.zeros(4, dtype=torch.float32)
        type_vec[int(row['type_label'])] = 1.0
 
        # severity_label: single integer for CrossEntropyLoss
        severity = torch.tensor(int(row['severity_label']), dtype=torch.long)
 
        return image, type_vec, severity
 
 
def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Creates train, val, and test DataLoaders."""
    train_ds = DegradationDataset(data_dir, split='train')
    val_ds   = DegradationDataset(data_dir, split='val')
    test_ds  = DegradationDataset(data_dir, split='test')
 
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
 
    print(f"Dataset sizes — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
 
 
# ═══════════════════════════════════════════════════════════════════════════
# NEW — Specialist Dataset (filters by degradation type)
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY: Each specialist model must be trained on images of ONE type only.
# For example the denoise specialist should only see noisy images —
# not blurry or compressed ones. This class filters the full 50,000
# image dataset down to only the relevant images for each specialist.
#
# From your 50,000 total images:
#   type_idx=0 (upsample)   → 12,500 images for upsample specialist
#   type_idx=1 (denoise)    → 12,500 images for denoise specialist
#   type_idx=2 (deartifact) → 12,500 images for deartifact specialist
#   type_idx=3 (inpaint)    → 12,500 images for inpaint specialist
#
# Each filtered set is split 80/10/10 for train/val/test.
# No new data generation needed — reuses your existing data_main/degraded/
# ═══════════════════════════════════════════════════════════════════════════
 
class SpecialistDataset(Dataset):
    """
    Filtered dataset for training one severity specialist.
 
    Only loads images of ONE degradation type (e.g. only noisy images).
    Labels are severity only (0-4) — no type label needed since
    all images in this dataset are the same type.
 
    Usage:
        # For denoise specialist (type_idx=1)
        ds = SpecialistDataset('data_main/degraded', type_idx=1, split='train')
        # Returns 10,000 training images (80% of 12,500 denoise images)
    """
 
    def __init__(self, data_dir: str, type_idx: int, split: str = 'train'):
        """
        Args:
            data_dir  : same folder as DegradationDataset (data_main/degraded)
            type_idx  : which degradation type to filter
                        0=upsample, 1=denoise, 2=deartifact, 3=inpaint
            split     : 'train' (80%), 'val' (10%), or 'test' (10%)
        """
        self.img_dir  = os.path.join(data_dir, 'images')
        self.type_idx = type_idx
 
        # Load full labels CSV
        csv_path = os.path.join(data_dir, 'labels.csv')
        df = pd.read_csv(csv_path)
 
        # FILTER — keep only rows matching the requested type.
        df = df[df['type_label'] == type_idx].copy()

        # Stratified 80/10/10 split by severity_label keeps all classes balanced
        # across train/val/test, which improves specialist val stability.
        rng = np.random.default_rng(42)
        split_rows = []
        for sev in range(len(LEVELS)):
            grp = df[df['severity_label'] == sev]
            if grp.empty:
                continue
            idx = grp.index.to_numpy(copy=True)
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            if split == 'train':
                chosen = idx[:n_train]
            elif split == 'val':
                chosen = idx[n_train:n_train + n_val]
            else:
                chosen = idx[n_train + n_val:]
            if chosen.size:
                split_rows.append(df.loc[chosen])

        if split_rows:
            self.df = pd.concat(split_rows, axis=0).sample(frac=1.0, random_state=42)
        else:
            self.df = df.iloc[0:0]
 
        self.df = self.df.reset_index(drop=True)
 
        # Both train and val use the same Resize(256)→CenterCrop(224) path so
        # noise statistics are at identical scale. Previously val used
        # Resize(224) directly which gave ~15% less noise than train images
        # at the same severity label — causing statistics MLP to fail on val.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if self.type_idx == TYPES.index('deartifact'):
            # Keep the full 256×256 frame — no crop so 8-pixel block grid stays aligned.
            # Strong augmentation for train: rotation and colour jitter prevent the
            # model memorising face identity → quality-level associations, forcing it
            # to learn the actual frequency/blocking features.
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                           saturation=0.1, hue=0.02),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    normalize,
                ])
        elif self.type_idx == TYPES.index('inpaint'):
            # Strong augmentation for inpaint too — mask coverage (not position)
            # is the severity signal; rotation/flip diversify mask position without
            # changing coverage, preventing spatial memorisation.
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),   # same resize ratio as train
                    transforms.ToTensor(),
                    normalize,
                ])
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
 
        img_path = os.path.join(self.img_dir, row['filename'])
        image    = Image.open(img_path).convert('RGB')
        image    = self.transform(image)
 
        # Only severity label — no type label needed
        # All images in this dataset are the same type
        severity = torch.tensor(int(row['severity_label']), dtype=torch.long)
 
        return image, severity
 
 
def get_specialist_dataloaders(data_dir: str, type_idx: int,
                                batch_size: int = 32,
                                num_workers: int = 4):
    """
    Creates train/val/test DataLoaders for ONE specialist.
 
    Args:
        data_dir  : path to degraded dataset (same as main dataset)
        type_idx  : 0=upsample, 1=denoise, 2=deartifact, 3=inpaint
 
    Returns:
        train_loader, val_loader, test_loader
        Each contains only images of the specified type
    """
    train_ds = SpecialistDataset(data_dir, type_idx, split='train')
    val_ds   = SpecialistDataset(data_dir, type_idx, split='val')
    test_ds  = SpecialistDataset(data_dir, type_idx, split='test')
 
    type_name = TYPES[type_idx]
    print(f"[{type_name} specialist] "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
 
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
 
    return train_loader, val_loader, test_loader
 