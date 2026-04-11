"""
Patches app.py with two fixes:
  1. Gradio lambda crash (numpy array as default arg)
  2. Checkpoint key remapping (old train.py: backbone.* prefix + Conv2d CBAM)
"""
import re

path = 'C:/Users/HP/Desktop/PRO(B)/BrainTumorAI/app.py'
src  = open(path, encoding='utf-8').read()

# ── FIX 1: replace bad lambda with proper closure ─────────────────────────
old_ex = """        # Example buttons
        for btn, arr in ex_btns:
            btn.click(fn=lambda a=arr: a, inputs=[], outputs=[img_in])"""

new_ex = """        # Example buttons — use closure to avoid numpy-array default-arg bug
        def _make_loader(a):
            def _load(): return a
            return _load
        for btn, arr in ex_btns:
            btn.click(fn=_make_loader(arr), inputs=[], outputs=[img_in])"""

assert old_ex in src, "Pattern 1 not found"
src = src.replace(old_ex, new_ex)
print("Fix 1 applied: example-button lambda fixed")

# ── FIX 2: replace load_model() with key-remapping version ────────────────
old_load = '''def load_model() -> Tuple[Union[Ensemble, MockEnsemble], str]:
    if not _TIMM:
        logger.info("timm missing — demo mode")
        return MockEnsemble(), "demo"

    fns = {
        "efficientnet": _make_efficientnet,
        "resnet_cbam":  _make_resnet_cbam,
        "densenet":     _make_densenet,
    }

    for base_str in SEARCH_PATHS:
        base = Path(base_str)
        if not base.exists():
            continue

        # ── Try full ensemble checkpoint ───────────────────────────────
        ens_ckpt = base / "ensemble_final.pt"
        if ens_ckpt.exists():
            try:
                mdls = {k: fn(NUM_CLASSES) for k, fn in fns.items()}
                ens  = Ensemble(mdls, ENS_WEIGHTS).to(DEVICE)
                ens.load_state_dict(
                    torch.load(ens_ckpt, map_location=DEVICE, weights_only=False))
                ens.eval()
                logger.info(f"Loaded full ensemble: {ens_ckpt}")
                return ens, "live"
            except Exception as e:
                logger.warning(f"Ensemble load failed: {e}")

        # ── Try individual checkpoints ─────────────────────────────────
        loaded: Dict[str, nn.Module] = {}
        for mname, mfn in fns.items():
            ckpt = _best_ckpt(base / mname)
            if ckpt is None:
                continue
            try:
                m     = mfn(NUM_CLASSES)
                state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
                m.load_state_dict(state["model_state_dict"])
                m.eval()
                loaded[mname] = m.to(DEVICE)
                logger.info(f"  {mname}: {ckpt.name}")
            except Exception as e:
                logger.warning(f"  {mname} failed: {e}")

        if loaded:
            w   = {k: ENS_WEIGHTS[k] for k in loaded}
            ens = Ensemble(loaded, w).to(DEVICE)
            ens.eval()
            mode = "live" if len(loaded) == 3 else "partial"
            logger.info(f"Assembled {mode} ensemble: {list(loaded.keys())}")
            return ens, mode

    logger.info("No checkpoints — demo mode")
    return MockEnsemble(), "demo"'''

new_load = '''def _remap_keys(raw_sd: dict, mname: str) -> dict:
    """
    Remap checkpoint saved by OLD train.py (backbone.* prefix, Conv2d CBAM)
    to match the current model architecture.
    Old format:
      - EfficientNet/DenseNet: backbone.* + head.4.weight/head.4.bias
      - ResNet+CBAM: backbone.* + cbam*.ca.fc.*.weight (4-D Conv2d) + head.*
    New format:
      - EfficientNet/DenseNet: flat timm keys + classifier.weight/bias
      - ResNet+CBAM: flat keys + cbam*.ca.fc.*.weight (2-D Linear) + head.*
    """
    new_sd = {}
    for k, v in raw_sd.items():
        nk = k
        # Strip backbone. prefix
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        # EfficientNet / DenseNet: map final head layer → classifier
        if mname in ("efficientnet", "densenet"):
            if nk == "head.4.weight":  nk = "classifier.weight"
            if nk == "head.4.bias":    nk = "classifier.bias"
            # Drop intermediate head layers (not in timm flat model)
            if nk.startswith("head.") and nk not in ("classifier.weight","classifier.bias"):
                continue
        # ResNet+CBAM: CBAM channel-attention weights were Conv2d [out,in,1,1]
        # Current model uses Linear [out,in] — squeeze spatial dims
        if "cbam" in nk and "ca.fc" in nk and v.dim() == 4:
            v = v.squeeze(-1).squeeze(-1)
        new_sd[nk] = v
    return new_sd


def load_model() -> Tuple[Union[Ensemble, MockEnsemble], str]:
    if not _TIMM:
        logger.info("timm missing — demo mode")
        return MockEnsemble(), "demo"

    fns = {
        "efficientnet": _make_efficientnet,
        "resnet_cbam":  _make_resnet_cbam,
        "densenet":     _make_densenet,
    }

    for base_str in SEARCH_PATHS:
        base = Path(base_str)
        if not base.exists():
            continue

        # ── Try full ensemble checkpoint ───────────────────────────────
        ens_ckpt = base / "ensemble_final.pt"
        if ens_ckpt.exists():
            try:
                mdls = {k: fn(NUM_CLASSES) for k, fn in fns.items()}
                ens  = Ensemble(mdls, ENS_WEIGHTS).to(DEVICE)
                ens.load_state_dict(
                    torch.load(ens_ckpt, map_location=DEVICE, weights_only=False))
                ens.eval()
                logger.info(f"Loaded full ensemble: {ens_ckpt}")
                return ens, "live"
            except Exception as e:
                logger.warning(f"Ensemble load failed: {e}")

        # ── Try individual checkpoints (with key remapping) ────────────
        loaded: Dict[str, nn.Module] = {}
        for mname, mfn in fns.items():
            ckpt = _best_ckpt(base / mname)
            if ckpt is None:
                continue
            try:
                m      = mfn(NUM_CLASSES)
                raw    = torch.load(ckpt, map_location=DEVICE, weights_only=False)
                raw_sd = raw["model_state_dict"]

                # Try direct load first (Colab-trained checkpoints)
                try:
                    m.load_state_dict(raw_sd, strict=True)
                    logger.info(f"  {mname}: {ckpt.name} (direct)")
                except Exception:
                    # Remap keys for old train.py checkpoints
                    remapped = _remap_keys(raw_sd, mname)
                    m.load_state_dict(remapped, strict=False)
                    missing  = [k for k in m.state_dict() if k not in remapped]
                    extra    = [k for k in remapped if k not in m.state_dict()]
                    logger.info(f"  {mname}: {ckpt.name} (remapped) "
                                f"missing={len(missing)} extra={len(extra)}")

                m.eval()
                loaded[mname] = m.to(DEVICE)
            except Exception as e:
                logger.warning(f"  {mname} failed: {e}")

        if loaded:
            w    = {k: ENS_WEIGHTS[k] for k in loaded}
            ens  = Ensemble(loaded, w).to(DEVICE)
            ens.eval()
            mode = "live" if len(loaded) == 3 else "partial"
            logger.info(f"Assembled {mode} ensemble: {list(loaded.keys())}")
            return ens, mode

    logger.info("No checkpoints — demo mode")
    return MockEnsemble(), "demo"'''

assert old_load in src, "Pattern 2 (load_model) not found"
src = src.replace(old_load, new_load)
print("Fix 2 applied: key-remapping load_model() added")

open(path, 'w', encoding='utf-8').write(src)
print("app.py saved.")

# verify syntax
import ast
try:
    ast.parse(src)
    print(f"Syntax OK — {src.count(chr(10))} lines")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
