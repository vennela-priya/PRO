"""segmentation/pseudo_masks.py — Generate pseudo masks from Grad-CAM."""
import cv2
import numpy as np
from pathlib import Path

CLASSES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def build_pseudo_dataset(
    images_dir,
    output_dir,
    gradcam_fn,
    threshold    = 0.40,
    skip_notumor = True,
    save_npy     = True,
    save_png     = True,
    verbose      = True,
):
    """
    Build pseudo-mask dataset from Grad-CAM activations.

    Parameters
    ----------
    images_dir   : root folder containing class subfolders
    output_dir   : where to write mask files
    gradcam_fn   : callable(img_rgb_hwc) -> 2-D float array in [0,1]
    threshold    : CAM binarisation threshold
    skip_notumor : if True, saves zero mask for notumor class
    save_npy     : save float32 .npy files
    save_png     : save uint8 .png files (0 or 255)
    verbose      : print progress

    Returns
    -------
    count of masks written
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    count        = 0

    for cls in CLASSES:
        cls_dir    = images_dir / cls
        is_notumor = (cls == 'notumor')

        if not cls_dir.exists():
            if verbose:
                print('  Skipping ' + cls + ': folder not found')
            continue

        img_paths = [p for p in cls_dir.rglob('*')
                     if p.suffix.lower() in IMG_EXTS]
        if verbose:
            print('  ' + cls + ': ' + str(len(img_paths)) + ' images')

        for img_path in img_paths:
            stem = img_path.stem

            # Skip if already generated
            if save_npy and (output_dir / (stem + '.npy')).exists():
                count += 1
                continue
            if save_png and not save_npy and (output_dir / (stem + '.png')).exists():
                count += 1
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            H, W = bgr.shape[:2]

            if is_notumor and skip_notumor:
                mask = np.zeros((H, W), dtype=np.float32)
            else:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                try:
                    cam = gradcam_fn(img_rgb)
                except Exception as e:
                    if verbose:
                        print('    GradCAM failed for ' + stem + ': ' + str(e))
                    mask = np.zeros((H, W), dtype=np.float32)
                else:
                    cam = cv2.resize(cam.astype(np.float32), (W, H))
                    mn, mx = cam.min(), cam.max()
                    if mx > mn:
                        cam = (cam - mn) / (mx - mn)

                    binary = (cam >= threshold).astype(np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_clean)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)

                    # Remove blobs smaller than 200 px
                    n_c, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
                    clean = np.zeros_like(binary)
                    for lbl in range(1, n_c):
                        if stats[lbl, cv2.CC_STAT_AREA] >= 200:
                            clean[labels == lbl] = 1
                    mask = clean.astype(np.float32)

            if save_npy:
                np.save(str(output_dir / (stem + '.npy')), mask)
            if save_png:
                cv2.imwrite(str(output_dir / (stem + '.png')),
                            (mask * 255).astype(np.uint8))
            count += 1

    if verbose:
        print('Total masks written: ' + str(count) + ' -> ' + str(output_dir))
    return count
