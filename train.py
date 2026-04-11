"""
train.py — Brain Tumor MRI Classification
3-phase ensemble training (CPU-compatible)

Models:  EfficientNet-B3 + ResNet50+CBAM + DenseNet121
Dataset: MainProject/Project  (4 classes: glioma, meningioma, notumor, pituitary)
Usage:   python train.py
"""
from __future__ import annotations
import logging, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics import AUROC, Accuracy
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def train_tfm(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=15, p=0.6, border_mode=0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
        A.OneOf([A.GaussianBlur(p=1), A.GaussNoise(p=1)], p=0.25),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=0, p=0.3),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, fill_value=0, p=0.15),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def val_tfm(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["path"]))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, int(row["label"])

    def class_weights(self):
        counts = self.df["label"].value_counts().sort_index().values.astype(float)
        w = 1.0 / counts; w /= w.sum()
        return torch.tensor(w, dtype=torch.float32)

    def sample_weights(self):
        counts = self.df["label"].value_counts()
        return (len(self.df) / self.df["label"].map(counts)).tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
import timm

def make_efficientnet(nc, pretrained=True):
    """EfficientNet-B3 — primary backbone."""
    backbone = timm.create_model("efficientnet_b3", pretrained=pretrained,
                                  num_classes=0, global_pool="avg")
    in_f = backbone.num_features  # 1536
    head = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, nc))

    class EffB3(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone; self.head=head
        def forward(self, x): return self.head(self.backbone(x))
        def freeze(self):
            for p in self.backbone.parameters(): p.requires_grad = False
        def unfreeze(self):
            for p in self.backbone.parameters(): p.requires_grad = True
        @property
        def target_layer(self): return self.backbone.blocks[-1]

    return EffB3()


def make_resnet_cbam(nc, pretrained=True):
    """ResNet-50 + CBAM attention — secondary backbone."""
    class CA(nn.Module):
        def __init__(self, c, r=16):
            super().__init__()
            self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,bias=False), nn.ReLU(),
                                    nn.Conv2d(c//r,c,1,bias=False))
            self.sig = nn.Sigmoid()
        def forward(self, x):
            return self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

    class SA(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2,1,7,padding=3,bias=False); self.sig = nn.Sigmoid()
        def forward(self, x):
            return self.sig(self.conv(torch.cat([x.mean(1,True), x.max(1,True)[0]],1)))

    class CBAM(nn.Module):
        def __init__(self, c): super().__init__(); self.ca=CA(c); self.sa=SA()
        def forward(self, x): return x * self.sa(x * self.ca(x))

    base = timm.create_model("resnet50", pretrained=pretrained,
                              num_classes=0, global_pool="")
    cbam3=CBAM(1024); cbam4=CBAM(2048)
    pool=nn.AdaptiveAvgPool2d(1)
    head=nn.Sequential(nn.Dropout(0.3), nn.Linear(2048,512), nn.ReLU(),
                       nn.Dropout(0.2), nn.Linear(512, nc))

    class ResNetCBAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=base; self.cbam3=cbam3; self.cbam4=cbam4
            self.pool=pool; self.head=head
        def forward(self, x):
            b=self.backbone
            x=b.act1(b.bn1(b.conv1(x))); x=b.maxpool(x)
            x=b.layer1(x); x=b.layer2(x)
            x=self.cbam3(b.layer3(x)); x=self.cbam4(b.layer4(x))
            return self.head(self.pool(x).flatten(1))
        def freeze(self):
            for p in self.backbone.parameters(): p.requires_grad = False
        def unfreeze(self):
            for p in self.backbone.parameters(): p.requires_grad = True
        @property
        def target_layer(self): return self.backbone.layer4[-1]

    return ResNetCBAM()


def make_densenet(nc, pretrained=True):
    """DenseNet-121 — tertiary backbone (CPU-efficient)."""
    backbone = timm.create_model("densenet121", pretrained=pretrained,
                                  num_classes=0, global_pool="avg")
    in_f = backbone.num_features  # 1024
    head = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, nc))

    class DenseNet(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=backbone; self.head=head
        def forward(self, x): return self.head(self.backbone(x))
        def freeze(self):
            for p in self.backbone.parameters(): p.requires_grad = False
        def unfreeze(self):
            for p in self.backbone.parameters(): p.requires_grad = True
        @property
        def target_layer(self): return self.backbone.features.denseblock4

    return DenseNet()


# ─────────────────────────────────────────────────────────────────────────────
# MixUp / CutMix
# ─────────────────────────────────────────────────────────────────────────────
def mixup(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    _, _, H, W = x.shape
    r = np.sqrt(1-lam)
    cx,cy = np.random.randint(W), np.random.randint(H)
    x1,x2 = max(0,int(cx-W*r/2)), min(W,int(cx+W*r/2))
    y1,y2 = max(0,int(cy-H*r/2)), min(H,int(cy+H*r/2))
    xm=x.clone(); xm[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    lam=1-(x2-x1)*(y2-y1)/(W*H)
    return xm, y, y[idx], lam

def mixed_loss(crit, logits, ya, yb, lam):
    return lam*crit(logits,ya)+(1-lam)*crit(logits,yb)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience=patience; self.best=None; self.counter=0
    def __call__(self, score):
        if self.best is None or score > self.best+1e-4:
            self.best=score; self.counter=0; return False
        self.counter+=1
        if self.counter>=self.patience:
            logger.info(f"Early stopping triggered."); return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Train one model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, nc, name, cfg):
    writer = SummaryWriter(log_dir=f"runs/{name}")
    ckpt_dir = Path("checkpoints") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    val_acc_m = Accuracy(task="multiclass", num_classes=nc).to(device)
    val_auc_m = AUROC(task="multiclass",   num_classes=nc).to(device)

    cw = train_loader.dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    best_auc=0.0; best_path=None

    def one_epoch(opt, ep, total_ep, aug=True):
        model.train(); total_loss=0.0; n=0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            r=random.random()
            if aug and r<0.3:
                imgs,ya,yb,lam=cutmix(imgs,labels); do_mix=True
            elif aug and r<0.5:
                imgs,ya,yb,lam=mixup(imgs,labels); do_mix=True
            else: do_mix=False
            opt.zero_grad()
            logits=model(imgs)
            loss=(mixed_loss(criterion,logits,ya,yb,lam) if do_mix
                  else criterion(logits,labels))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total_loss+=loss.item(); n+=1
        return total_loss/n

    def validate(ep):
        nonlocal best_auc, best_path
        model.eval(); val_acc_m.reset(); val_auc_m.reset(); vloss=0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs,labels=imgs.to(device),labels.to(device)
                logits=model(imgs)
                vloss+=criterion(logits,labels).item()
                probs=torch.softmax(logits,-1)
                val_acc_m.update(probs.argmax(1),labels)
                val_auc_m.update(probs,labels)
        vacc=val_acc_m.compute().item()
        vauc=val_auc_m.compute().item()
        vl=vloss/len(val_loader)
        if vauc>best_auc:
            best_auc=vauc
            best_path=ckpt_dir/f"best_ep{ep:03d}_auc{vauc:.4f}.pt"
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),
                        "auc":vauc,"acc":vacc}, best_path)
            logger.info(f"  ✓ Saved best: auc={vauc:.4f} acc={vacc:.4f}")
        return vacc, vauc, vl

    # ── Resume from existing checkpoint if available ─────────────────────
    existing = sorted(ckpt_dir.glob("best_*.pt"), reverse=True)
    if existing:
        ckpt = torch.load(existing[0], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_auc = ckpt.get("auc", 0.0)
        best_path = existing[0]
        logger.info(f"[{name}] Resuming from {existing[0].name} (auc={best_auc:.4f}) — skipping Phase 1")
        skip_phase1 = True
    else:
        skip_phase1 = False

    # ── Phase 1: frozen backbone, warm up head ──────────────────────────
    if not skip_phase1:
        logger.info(f"\n{'─'*50}\n[{name}] Phase 1 — head warm-up (3 epochs)\n{'─'*50}")
        model.freeze()
        opt1 = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),
                                lr=1e-3)
    for ep in range(1, 4 if not skip_phase1 else 0):
        t0=time.time()
        tl=one_epoch(opt1, ep, 10, aug=True)
        vacc,vauc,vl=validate(ep)
        logger.info(f"[{name}] P1 ep{ep:02d}/03 "
                    f"train_loss={tl:.4f} val_loss={vl:.4f} "
                    f"val_acc={vacc:.4f} val_auc={vauc:.4f} "
                    f"({time.time()-t0:.0f}s)")
        writer.add_scalars(f"{name}/loss",{"train":tl,"val":vl},ep)
        writer.add_scalars(f"{name}/metrics",{"acc":vacc,"auc":vauc},ep)

    # ── Phase 2: full fine-tune with cosine LR ──────────────────────────
    logger.info(f"\n{'─'*50}\n[{name}] Phase 2 — full fine-tune (5 epochs)\n{'─'*50}")
    model.unfreeze()
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5, eta_min=1e-6)
    es = EarlyStopping(patience=3)

    for ep in range(1,6):
        t0=time.time()
        tl=one_epoch(opt2, ep, 5, aug=True)
        vacc,vauc,vl=validate(ep+3)
        sched.step()
        lr_now=opt2.param_groups[0]["lr"]
        logger.info(f"[{name}] P2 ep{ep:02d}/05 "
                    f"train_loss={tl:.4f} val_loss={vl:.4f} "
                    f"val_acc={vacc:.4f} val_auc={vauc:.4f} "
                    f"lr={lr_now:.2e} ({time.time()-t0:.0f}s)")
        writer.add_scalars(f"{name}/loss",{"train":tl,"val":vl},ep+3)
        writer.add_scalars(f"{name}/metrics",{"acc":vacc,"auc":vauc},ep+3)
        if es(vauc): break

    writer.close()
    logger.info(f"[{name}] Best AUC={best_auc:.4f} → {best_path}")
    return best_path, best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────
class Ensemble(nn.Module):
    def __init__(self, models: dict, weights: dict):
        super().__init__()
        self.mods = nn.ModuleDict(models)
        self.weights = weights
        self.temps = nn.ParameterDict(
            {k: nn.Parameter(torch.ones(1)) for k in models})

    def forward(self, x):
        out=None
        for k,m in self.mods.items():
            logits=m(x)/self.temps[k].clamp(min=0.1)
            p=torch.softmax(logits,-1)*self.weights[k]
            out=p if out is None else out+p
        return out

    def calibrate(self, val_loader, device):
        for m in self.mods.values(): m.eval()
        self.to(device)
        cached={k:[] for k in self.mods}; labs=[]
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs=imgs.to(device)
                for k,m in self.mods.items(): cached[k].append(m(imgs).cpu())
                labs.append(labels)
        for k in self.mods: cached[k]=torch.cat(cached[k]).to(device)
        labs=torch.cat(labs).to(device)
        nll=nn.CrossEntropyLoss()
        opt=torch.optim.LBFGS([self.temps[k] for k in self.mods], lr=0.01, max_iter=50)
        def closure():
            opt.zero_grad(); out=None
            for k in self.mods:
                p=torch.softmax(cached[k]/self.temps[k].clamp(min=0.1),-1)*self.weights[k]
                out=p if out is None else out+p
            loss=nll(torch.log(out+1e-8),labs); loss.backward(); return loss
        opt.step(closure)
        for k in self.mods: logger.info(f"  Temp[{k}] = {self.temps[k].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             recall_score, confusion_matrix, classification_report)

def evaluate(model, loader, device, nc, class_names, label=""):
    model.eval()
    all_prob, all_true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs=imgs.to(device)
            out=model(imgs)
            p=torch.softmax(out,-1) if out.min()<0 or out.max()>1 else out
            all_prob.append(p.cpu().numpy())
            all_true.append(labels.numpy())
    y_prob=np.concatenate(all_prob)
    y_true=np.concatenate(all_true)
    y_pred=y_prob.argmax(1)

    acc  = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except: auc=0.0
    sens = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    specs=[]
    for c in range(nc):
        tn=((y_true!=c)&(y_pred!=c)).sum(); fp=((y_true!=c)&(y_pred==c)).sum()
        specs.append(tn/(tn+fp+1e-8))
    spec=float(np.mean(specs))

    logger.info(f"\n{'='*55}")
    logger.info(f"  {label}")
    logger.info(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    logger.info(f"  AUC (macro) : {auc:.4f}")
    logger.info(f"  Sensitivity : {sens:.4f}  ({sens*100:.2f}%)")
    logger.info(f"  Specificity : {spec:.4f}  ({spec*100:.2f}%)")
    logger.info(f"  F1 (macro)  : {f1:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=class_names, digits=4)}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info(f"{'='*55}")

    return {"accuracy":acc,"auc":auc,"sensitivity":sens,"specificity":spec,
            "f1_macro":f1,"y_true":y_true,"y_pred":y_pred,"y_prob":y_prob,"cm":cm}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with open("config.yaml") as f:
        config=yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc=config["data"]["num_classes"]
    names=config["data"]["class_names"]
    size=config["data"]["image_size"]
    logger.info(f"Device: {device} | Classes: {nc} → {names}")

    # ── Data ──────────────────────────────────────────────────────────────
    tv_csv=Path(config["data"]["trainval_csv"])
    if not tv_csv.exists():
        logger.info("trainval.csv not found — running prepare_data.py…")
        import subprocess
        subprocess.run([sys.executable,"prepare_data.py"],check=True)

    df=pd.read_csv(tv_csv)
    train_df=df[df["split"]=="train"]
    val_df  =df[df["split"]=="val"]
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
    logger.info(f"Train distribution:\n{train_df['class_name'].value_counts().to_string()}")

    bs=config["training"]["batch_size"]
    nw=config["data"]["num_workers"]
    pm=config["data"]["pin_memory"]

    train_ds=MRIDataset(train_df, train_tfm(size))
    val_ds  =MRIDataset(val_df,   val_tfm(size))

    sampler=WeightedRandomSampler(train_ds.sample_weights(),len(train_ds),replacement=True)
    train_loader=DataLoader(train_ds,batch_size=bs,sampler=sampler,
                            num_workers=nw,pin_memory=pm,drop_last=True)
    val_loader  =DataLoader(val_ds,  batch_size=bs,shuffle=False,
                            num_workers=nw,pin_memory=pm)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # ── Train 3 models ────────────────────────────────────────────────────
    model_fns={"efficientnet":make_efficientnet,
               "resnet_cbam": make_resnet_cbam,
               "densenet":    make_densenet}

    best_paths={}; best_aucs={}
    for mname,mfn in model_fns.items():
        logger.info(f"\n{'#'*55}\nTraining: {mname.upper()}\n{'#'*55}")
        model=mfn(nc, pretrained=True).to(device)
        bp,ba=train_model(model,train_loader,val_loader,device,nc,mname,config)
        best_paths[mname]=bp; best_aucs[mname]=ba

    # ── Reload best weights ───────────────────────────────────────────────
    loaded={}
    for mname,mfn in model_fns.items():
        m=mfn(nc,pretrained=False).to(device)
        ckpt=torch.load(best_paths[mname],map_location=device)
        m.load_state_dict(ckpt["model_state_dict"]); m.eval()
        loaded[mname]=m

    # ── Ensemble ──────────────────────────────────────────────────────────
    logger.info(f"\n{'#'*55}\nBuilding & Calibrating Ensemble\n{'#'*55}")
    ens_weights={"efficientnet":0.4,"resnet_cbam":0.3,"densenet":0.3}
    ensemble=Ensemble(loaded,ens_weights).to(device)
    logger.info("Calibrating temperatures on val set…")
    ensemble.calibrate(val_loader, device)
    torch.save(ensemble.state_dict(),"checkpoints/ensemble_final.pt")
    logger.info("Ensemble saved → checkpoints/ensemble_final.pt")

    # ── Val evaluation ─────────────────────────────────────────────────────
    logger.info(f"\n{'#'*55}\nValidation Evaluation\n{'#'*55}")
    rows=[]
    for mname,m in loaded.items():
        res=evaluate(m,val_loader,device,nc,names,label=f"{mname} (VAL)")
        rows.append({"model":mname,"split":"val","accuracy":res["accuracy"],
                     "auc":res["auc"],"sensitivity":res["sensitivity"],
                     "specificity":res["specificity"],"f1_macro":res["f1_macro"]})

    ens_res=evaluate(ensemble,val_loader,device,nc,names,label="ENSEMBLE (VAL)")
    rows.append({"model":"ensemble","split":"val","accuracy":ens_res["accuracy"],
                 "auc":ens_res["auc"],"sensitivity":ens_res["sensitivity"],
                 "specificity":ens_res["specificity"],"f1_macro":ens_res["f1_macro"]})

    df_res=pd.DataFrame(rows)
    df_res.to_csv("results/metrics_table.csv",index=False)

    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULTS SUMMARY")
    print(f"{'='*65}")
    print(df_res[["model","accuracy","auc","sensitivity","specificity","f1_macro"]].to_string(index=False))
    print(f"{'='*65}")

    ens=rows[-1]
    logger.info(f"""
{'='*55}
  TRAINING COMPLETE

  Ensemble VAL metrics:
    Accuracy    : {ens['accuracy']:.4f} ({ens['accuracy']*100:.2f}%)
    AUC (macro) : {ens['auc']:.4f}
    Sensitivity : {ens['sensitivity']:.4f}
    Specificity : {ens['specificity']:.4f}
    F1 (macro)  : {ens['f1_macro']:.4f}

  If satisfied with val accuracy → run:
    python final_test.py
{'='*55}""")


if __name__ == "__main__":
    main()
