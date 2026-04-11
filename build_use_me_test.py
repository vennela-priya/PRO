"""
build_use_me_test.py
--------------------
Runs every image in Project/Testing/ through the best ResNetCBAM checkpoint,
keeps only the CORRECT predictions, ranks them by softmax confidence, and
copies the top TOP_N_PER_CLASS images per class into  USE-Me Test/<class>/.

Run from the project root:
    py -3 build_use_me_test.py
"""

import os, sys, shutil, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── config ──────────────────────────────────────────────────────────────────
BASE            = Path(__file__).parent
TEST_ROOT       = BASE / "Project" / "Testing"
DEST_ROOT       = BASE / "USE-Me Test"
CKPT_DIR        = BASE / "checkpoints" / "resnet_cbam"

CLASSES         = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE        = 224
TOP_N_PER_CLASS = 60     # highest-confidence CORRECT images per class

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── model (mirrors app.py exactly) ──────────────────────────────────────────
try:
    import timm
    _TIMM = True
except ImportError:
    _TIMM = False

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False))
    def forward(self, x):
        B, C, H, W = x.shape
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])
        return x * torch.sigmoid(self.fc(avg) + self.fc(mx)).view(B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx  = x.amax(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.ca = ChannelAttention(c, r)
        self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ResNetCBAM(nn.Module):
    def __init__(self, nc):
        super().__init__()
        base = timm.create_model("resnet50", pretrained=False,
                                 num_classes=0, global_pool="")
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.act1    = base.act1
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.cbam3   = CBAM(1024)
        self.cbam4   = CBAM(2048)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, nc))
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        return self.head(self.pool(x).flatten(1))

# ── checkpoint helpers ───────────────────────────────────────────────────────
def _remap(raw_sd):
    """Strip backbone. prefix; squeeze Conv2d CBAM -> Linear."""
    out = {}
    for k, v in raw_sd.items():
        nk = k[len("backbone."):] if k.startswith("backbone.") else k
        if "cbam" in nk and "ca.fc" in nk and v.dim() == 4:
            v = v.squeeze(-1).squeeze(-1)
        out[nk] = v
    return out

def load_model():
    if not _TIMM:
        print("timm not installed -- aborting"); sys.exit(1)
    ckpt_files = sorted(CKPT_DIR.glob("best_*.pt"), reverse=True)
    if not ckpt_files:
        print(f"No checkpoint in {CKPT_DIR}"); sys.exit(1)
    ckpt_path = ckpt_files[0]
    print(f"Checkpoint : {ckpt_path.name}")
    raw    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = raw["model_state_dict"]
    print(f"  saved AUC: {raw.get('auc', 'n/a'):.4f}   acc: {raw.get('acc', 'n/a'):.4f}")
    m = ResNetCBAM(len(CLASSES))
    try:
        m.load_state_dict(raw_sd, strict=True)
        print("  -> direct load OK")
    except Exception:
        remapped = _remap(raw_sd)
        mis = m.load_state_dict(remapped, strict=False)
        print(f"  -> remapped load OK  (missing={len(mis.missing_keys)})")
    m.eval()
    return m

# ── preprocessing ────────────────────────────────────────────────────────────
def preprocess(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    model = load_model()

    # collect all test images with ground-truth label
    test_images = []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = TEST_ROOT / cls_name
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} missing -- skipping"); continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            for p in sorted(cls_dir.glob(ext)):
                test_images.append((p, cls_idx, cls_name))

    print(f"\nTest images  : {len(test_images)}  ({TEST_ROOT})")
    print(f"Strategy     : top {TOP_N_PER_CLASS} highest-confidence CORRECT per class\n")

    # ── inference ────────────────────────────────────────────────────────────
    results = []   # (path, gt_idx, gt_name, pred_idx, confidence)
    with torch.no_grad():
        for i, (img_path, gt_idx, gt_name) in enumerate(test_images):
            if i % 100 == 0:
                print(f"  [{i:4d}/{len(test_images)}] ...")
            try:
                t      = preprocess(img_path)
                logits = model(t)
                probs  = F.softmax(logits.float(), dim=1).squeeze().numpy()
                pred   = int(np.argmax(probs))
                conf   = float(probs[pred])
                results.append((img_path, gt_idx, gt_name, pred, conf))
            except Exception as e:
                print(f"  SKIP {img_path.name}: {e}")

    # ── per-class accuracy summary ────────────────────────────────────────────
    print()
    total_c   = {c: 0 for c in CLASSES}
    correct_c = {c: 0 for c in CLASSES}
    for _, gi, gn, pi, _ in results:
        total_c[gn] += 1
        if pi == gi: correct_c[gn] += 1
    for cls_name in CLASSES:
        t = total_c[cls_name]; c2 = correct_c[cls_name]
        print(f"  {cls_name:12s}: {c2:3d}/{t} correct ({c2/t*100:.1f}%)")

    # ── select top-N correct per class ────────────────────────────────────────
    correct = [(p, gi, gn, pi, c) for p, gi, gn, pi, c in results if pi == gi]
    correct.sort(key=lambda x: -x[4])   # descending confidence

    # rebuild destination folder
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir()

    per_class = {c: 0 for c in CLASSES}
    selected  = []

    for img_path, gt_idx, gt_name, pred_idx, conf in correct:
        if per_class[gt_name] >= TOP_N_PER_CLASS:
            continue
        dest_cls = DEST_ROOT / gt_name
        dest_cls.mkdir(exist_ok=True)
        shutil.copy2(img_path, dest_cls / img_path.name)
        per_class[gt_name] += 1
        selected.append((img_path, gt_name, conf))

    copied = sum(per_class.values())

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  Folder : {DEST_ROOT}")
    print(f"{'='*58}")
    for cls_name in CLASSES:
        n   = per_class[cls_name]
        grp = [c for _, gn, c in selected if gn == cls_name]
        lo  = min(grp) * 100 if grp else 0
        hi  = max(grp) * 100 if grp else 0
        print(f"  {cls_name:12s}  {n:3d} images   conf {lo:.1f}% -- {hi:.1f}%")
    print(f"{'='*58}")
    print(f"  TOTAL  : {copied} images  (top-{TOP_N_PER_CLASS} correct per class)")
    print(f"{'='*58}")

    # ── manifest CSV ──────────────────────────────────────────────────────────
    manifest = DEST_ROOT / "manifest.csv"
    rank = {c: 0 for c in CLASSES}
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "confidence_pct", "rank_in_class"])
        for img_path, gt_name, conf in selected:
            rank[gt_name] += 1
            w.writerow([img_path.name, gt_name,
                        f"{conf * 100:.2f}", rank[gt_name]])
    print(f"\n  Manifest -> {manifest}")

if __name__ == "__main__":
    main()
