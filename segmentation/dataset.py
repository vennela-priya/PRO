"""segmentation/dataset.py — Brain tumour segmentation dataset."""
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


class BrainSegDataset(Dataset):
    """
    Loads MRI images and their pseudo-mask pairs.

    Parameters
    ----------
    image_dir    : folder containing class subfolders (glioma/, meningioma/, ...)
    pseudo_cache : folder with .npy or .png masks (same stem as image)
    image_size   : square resize target
    train        : True -> augmentation; False -> resize + normalise only

    __getitem__ returns
    -------------------
    {'image': FloatTensor (3,H,W), 'mask': FloatTensor (1,H,W)}
    """

    def __init__(self, image_dir, pseudo_cache, image_size=224, train=True):
        self.pseudo_cache = Path(pseudo_cache)
        self.image_size   = image_size
        self.train        = train

        image_dir    = Path(image_dir)
        self.samples = []

        # Try class subfolders first
        found_class = False
        for cls in CLASSES:
            cls_dir = image_dir / cls
            if cls_dir.exists():
                found_class = True
                for p in cls_dir.rglob('*'):
                    if p.is_file() and p.suffix.lower() in IMG_EXTS:
                        self.samples.append(p)

        # Fallback: flat directory scan
        if not found_class:
            for p in image_dir.rglob('*'):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append(p)

        if not self.samples:
            raise FileNotFoundError(
                'No images found in: ' + str(image_dir) +
                '  Expected subfolders: ' + ', '.join(CLASSES)
            )

        if train:
            self.tfm = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.10, scale_limit=0.15,
                    rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(p=0.3),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])
        else:
            self.tfm = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, stem, h, w):
        npy = self.pseudo_cache / (stem + '.npy')
        png = self.pseudo_cache / (stem + '.png')
        if npy.exists():
            mask = np.load(str(npy)).astype(np.float32)
        elif png.exists():
            m    = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            mask = (m / 255.0).astype(np.float32) if m is not None else None
        else:
            mask = None
        if mask is None:
            mask = np.zeros((h, w), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        bgr      = cv2.imread(str(img_path))
        if bgr is None:
            bgr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w     = img.shape[:2]
        mask     = self._load_mask(img_path.stem, h, w)

        result       = self.tfm(image=img, mask=mask)
        image_tensor = result['image'].float()
        mask_tensor  = result['mask'].float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return {'image': image_tensor, 'mask': mask_tensor}
