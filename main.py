import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    balanced_accuracy_score, matthews_corrcoef
)
import pandas as pd

from cds_patchnet_test import CDSPatchNet, CDSPatchNetLoss, Config
from baselines_test import GAPClassifier, AttentionPoolClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = datasets.ImageFolder("dummy_data/train", transform=train_tf)
val_ds = datasets.ImageFolder("dummy_data/val", transform=eval_tf)
test_ds = datasets.ImageFolder("dummy_data/test", transform=eval_tf)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
num_classes = len(train_ds.classes)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec,
            "macro_f1": f1, "balanced_accuracy": bal_acc, "mcc": mcc}


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs["logits"].argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return compute_metrics(all_labels, all_preds)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels, model)
        loss.backward()
        optimizer.step()
        if hasattr(model, "renormalize_prototypes"):
            model.renormalize_prototypes()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad_(False)


def unfreeze_last_stages(model, stage_indices=(2, 3)):
    for idx in stage_indices:
        for p in model.backbone.stages[idx].parameters():
            p.requires_grad_(True)


def get_optimizer_stage1(model, lr_new=1e-3, wd=1e-4):
    freeze_backbone(model)
    new_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.AdamW(new_params, lr=lr_new, weight_decay=wd)


def get_optimizer_stage2(model, lr_backbone=1e-5, lr_new=1e-4, wd=1e-4):
    freeze_backbone(model)
    unfreeze_last_stages(model)
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    new_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr_backbone}, {"params": new_params, "lr": lr_new}],
        weight_decay=wd,
    )


def run_two_stage_training(model, model_name, loss_fn, stage1_epochs, stage2_epochs,
                            patience=3, ckpt_dir="ckpts"):
    os.makedirs(ckpt_dir, exist_ok=True)
    best_f1, best_state, no_improve = -1, None, 0
    history = []

    def run_stage(optimizer, n_epochs, stage_name):
        nonlocal best_f1, best_state, no_improve
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1), eta_min=1e-6)
        for epoch in range(n_epochs):
            tl = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            vm = evaluate(model, val_loader, DEVICE)
            sched.step()
            history.append({"stage": stage_name, "epoch": epoch, "train_loss": tl, **vm})
            improved = vm["macro_f1"] > best_f1
            if improved:
                best_f1, best_state, no_improve = vm["macro_f1"], copy.deepcopy(model.state_dict()), 0
                torch.save(best_state, os.path.join(ckpt_dir, f"{model_name}_best.pth"))
            else:
                no_improve += 1
            if no_improve >= patience:
                return True
        return False

    stopped = run_stage(get_optimizer_stage1(model), stage1_epochs, "stage1")
    if not stopped:
        run_stage(get_optimizer_stage2(model), stage2_epochs, "stage2")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---- CDS-PatchNet ----
cfg = Config()
cds_model = CDSPatchNet(num_classes, cfg, pretrained=False).to(DEVICE)
cds_loss_module = CDSPatchNetLoss(cfg)


def cds_loss_fn(outputs, labels, model):
    total, _ = cds_loss_module(outputs, labels, model.prototypes)
    return total


hist_cds = run_two_stage_training(cds_model, "cds_patchnet", cds_loss_fn, stage1_epochs=2, stage2_epochs=2, patience=5)
test_metrics_cds = evaluate(cds_model, test_loader, DEVICE)
print("CDS-PatchNet test metrics:", test_metrics_cds)

# ---- Baselines ----
ce = nn.CrossEntropyLoss()


def ce_loss_fn(outputs, labels, model):
    return ce(outputs["logits"], labels)


gap_model = GAPClassifier(num_classes, pretrained=False).to(DEVICE)
run_two_stage_training(gap_model, "gap_baseline", ce_loss_fn, stage1_epochs=2, stage2_epochs=2, patience=5)
test_metrics_gap = evaluate(gap_model, test_loader, DEVICE)
print("GAP baseline test metrics:", test_metrics_gap)

attn_model = AttentionPoolClassifier(num_classes, pretrained=False).to(DEVICE)
run_two_stage_training(attn_model, "attn_baseline", ce_loss_fn, stage1_epochs=2, stage2_epochs=2, patience=5)
test_metrics_attn = evaluate(attn_model, test_loader, DEVICE)
print("Attention-pool baseline test metrics:", test_metrics_attn)

df = pd.DataFrame({
    "CDS-PatchNet": test_metrics_cds,
    "GAP Baseline": test_metrics_gap,
    "Attention-Pool Baseline": test_metrics_attn,
}).T
print(df)

print("checkpoint files:", os.listdir("ckpts"))
print("ALL-MODELS PIPELINE TEST PASSED")
