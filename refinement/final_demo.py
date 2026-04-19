"""
refinement/final_demo.py
=========================
Step 10 — Final demo: run the pipeline on 5 different MRI images
(glioma, meningioma, pituitary, notumor, glioma) and produce a mosaic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from refinement.pipeline import run_pipeline


def run_final_demo(
    unet_path:        str,
    classifier_path:  str  = "",
    output_dir:       str  = "outputs/",
) -> None:
    """
    Run the refinement pipeline on 5 representative MRI images and save mosaic.

    Parameters
    ----------
    unet_path        : str  Path to the AttentionUNet checkpoint.
    classifier_path  : str  Path to the ResNetCBAM classifier checkpoint (auto-found if empty).
    output_dir       : str  Directory to save the mosaic and per-image results.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover classifier if not given
    if not classifier_path:
        _clf = _PROJECT_ROOT / "checkpoints" / "resnet_cbam" / "best_ep008_auc0.9952.pt"
        classifier_path = str(_clf) if _clf.exists() else ""
        if classifier_path:
            print(f"[Demo] Auto-found classifier: {_clf.name}")
        else:
            print("[Demo] WARNING: No classifier found — GradCAM will be skipped")

    # 5 images: glioma, meningioma, pituitary, notumor, glioma (second one)
    base = _PROJECT_ROOT / "Project" / "Testing"
    demo_images = [
        ("glioma",      str(base / "glioma"     / "Te-glTr_0000.jpg")),
        ("meningioma",  str(base / "meningioma"  / "Te-meTr_0000.jpg")),
        ("pituitary",   str(base / "pituitary"   / "Te-piTr_0000.jpg")),
        ("notumor",     str(base / "notumor"     / "Te-noTr_0000.jpg")),
        ("glioma",      str(base / "glioma"      / "Te-glTr_0001.jpg")),
    ]

    # Fallback to data/ folder images if Project/ does not exist
    alt_base = _PROJECT_ROOT / "data" / "Testing - 副本"
    if not base.exists():
        demo_images = [
            ("glioma",     str(alt_base / "glioma"     / "Te-glTr_0000.jpg")),
            ("meningioma", str(alt_base / "meningioma"  / "Te-meTr_0000.jpg")),
            ("pituitary",  str(alt_base / "pituitary"   / "Te-piTr_0000.jpg")),
            ("notumor",    str(alt_base / "notumor"     / "Te-noTr_0000.jpg")),
            ("glioma",     str(alt_base / "glioma"      / "Te-glTr_0001.jpg")),
        ]

    panels = []
    print(f"\n{'='*70}")
    print(f"{'FINAL DEMO — Pseudo-Mask Refinement Pipeline':^70}")
    print(f"{'='*70}\n")

    for label, img_path in demo_images:
        if not Path(img_path).exists():
            print(f"  [SKIP] {img_path} not found")
            continue

        t0 = time.time()
        result = run_pipeline(
            image_path      = img_path,
            unet_path       = unet_path,
            classifier_path = classifier_path or None,
            output_dir      = str(out_dir),
            skip_iterative  = True,
            skip_crf        = False,
        )
        elapsed = time.time() - t0

        print(
            f"Image: {Path(img_path).name:<28} | "
            f"Dice before: {result['dice_before']:.2f} | "
            f"Dice after: {result['dice_after']:.2f} | "
            f"Improvement: {result['dice_improvement']:+.2f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Build mosaic panel for this image (3-column: original | refined mask | boundary)
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        s     = 200   # panel size
        orig  = cv2.resize(rgb, (s, s))

        ref   = result["refined_mask"]
        ref_v = cv2.resize((ref * 255).astype(np.uint8), (s, s))
        ref_rgb = np.stack([ref_v] * 3, axis=-1)

        bnd = result["boundary_image"]
        bnd_v = cv2.resize(bnd, (s, s))

        # Label strip
        lh  = 20
        row = np.concatenate([orig, ref_rgb, bnd_v], axis=1)
        lbl = np.ones((lh, row.shape[1], 3), dtype=np.uint8) * 30
        cv2.putText(lbl, f"{label} | dice={result['dice_after']:.2f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 50), 1)
        panel = np.concatenate([lbl, row], axis=0)
        panels.append(panel)

    if panels:
        mosaic   = np.concatenate(panels, axis=0)
        out_path = str(out_dir / "final_results_mosaic.png")
        cv2.imwrite(out_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
        print(f"\nMosaic saved -> {out_path}")

    print("\nPIPELINE COMPLETE — all outputs saved to outputs/")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--unet",       default="checkpoints/unet/best_unet.pth")
    p.add_argument("--classifier", default="")
    p.add_argument("--output_dir", default="outputs/")
    args = p.parse_args()

    run_final_demo(unet_path=args.unet,
                   classifier_path=args.classifier,
                   output_dir=args.output_dir)
