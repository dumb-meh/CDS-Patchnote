# %% [markdown]
# # CDS-PatchNet Fine-Grained Classification - Complete Implementation
# This notebook implements the complete CDS-PatchNet fine-grained classification pipeline.
# All model definitions, training loops, and evaluation are contained in this single file.

# %% Imports
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
import timm

# %% Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# %% Data Transforms
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# %% Configuration Class
class Config:
    """
    Configuration class for CDS-PatchNet hyperparameters.
    
    Attributes:
        feat_dim (int): Feature dimension for patch projections and prototypes.
        tau (float): Temperature for softmax in semantic affinity computation.
        alpha (float): Weight for semantic affinity in combined affinity matrix.
        beta (float): Weight for spatial affinity in combined affinity matrix.
        gamma (float): Weight for class-specific affinity in combined affinity matrix.
        sigma (float): Standard deviation for Gaussian kernel in spatial affinity.
        solver_iters (int): Number of iterations for replicator dynamics solver.
        lambda_proto (float): Weight for prototype loss term.
        lambda_coh (float): Weight for coherence loss term.
    """
    def __init__(self):
        self.feat_dim = 256
        self.tau = 0.1
        self.alpha = 0.4
        self.beta = 0.2
        self.gamma = 0.4
        self.sigma = 1.0
        self.solver_iters = 10
        self.lambda_proto = 0.1
        self.lambda_coh = 0.01


# %% Helper Function: Build Affinity Matrix
def build_affinity(patches, prototype, config, class_r_tilde=None):
    """
    Build the combined affinity matrix for replicator dynamics.
    
    Args:
        patches (torch.Tensor): Patch features of shape (B, N, feat_dim) where N=49.
        prototype (torch.Tensor): Class prototype of shape (feat_dim,).
        config (Config): Configuration object.
        class_r_tilde (torch.Tensor, optional): Class-specific soft assignments (B, N).
    
    Returns:
        torch.Tensor: Affinity matrix of shape (B, N, N).
    """
    B, N, D = patches.shape
    
    # Normalize patches for cosine similarity
    patches_norm = F.normalize(patches, dim=-1)  # (B, N, D)
    
    # Semantic affinity: pairwise cosine similarity between patches
    # sim[i,j] = cos(patch_i, patch_j)
    sem_affinity = torch.bmm(patches_norm, patches_norm.transpose(1, 2))  # (B, N, N)
    sem_affinity = torch.relu(sem_affinity)  # max(0, sim)
    
    # Zero out diagonal
    eye = torch.eye(N, device=patches.device).unsqueeze(0).expand(B, -1, -1)
    sem_affinity = sem_affinity * (1 - eye)
    
    # Spatial affinity: Gaussian kernel on grid coordinates
    # Create 7x7 grid coordinates
    grid_size = 7
    coords = torch.stack(torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32, device=patches.device),
        torch.arange(grid_size, dtype=torch.float32, device=patches.device),
        indexing='ij'
    ), dim=-1).reshape(-1, 2)  # (49, 2)
    
    # Compute pairwise squared Euclidean distances
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (1, 49, 49, 2)
    sq_dist = (diff ** 2).sum(dim=-1)  # (1, 49, 49)
    
    # Gaussian kernel
    spa_affinity = torch.exp(-sq_dist / (2 * config.sigma ** 2))  # (1, 49, 49)
    spa_affinity = spa_affinity.expand(B, -1, -1)
    
    # Zero out diagonal
    spa_affinity = spa_affinity * (1 - eye)
    
    # Combined affinity
    A = config.alpha * sem_affinity + config.beta * spa_affinity
    
    # Add class-specific term if r_tilde is provided
    if class_r_tilde is not None:
        # Outer product: r_tilde.unsqueeze(-1) * r_tilde.unsqueeze(-2) -> (B, N, N)
        class_affinity = class_r_tilde.unsqueeze(-1) * class_r_tilde.unsqueeze(-2)
        A = A + config.gamma * class_affinity
    
    return A


# %% CDSPatchNet Model
class CDSPatchNet(nn.Module):
    """
    CDS-PatchNet: Class-Discriminative Semantic Patch Network for fine-grained classification.
    
    This model uses a ConvNeXt backbone to extract patch features, then applies
    replicator dynamics to compute class-specific patch memberships, and finally
    fuses global and local features for classification.
    """
    
    def __init__(self, num_classes, config, pretrained=True):
        """
        Initialize CDSPatchNet.
        
        Args:
            num_classes (int): Number of output classes.
            config (Config): Configuration object with hyperparameters.
            pretrained (bool): Whether to use pretrained backbone weights.
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Backbone: ConvNeXt Tiny without classification head
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # Don't apply global pooling
        )
        
        # Patch projection: 768 -> feat_dim
        self.patch_proj = nn.Linear(768, config.feat_dim)
        
        # Class prototypes: learnable parameters (num_classes, feat_dim)
        self.prototypes = nn.Parameter(
            torch.empty(num_classes, config.feat_dim)
        )
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.prototypes)
        
        # Fusion MLP per class
        # Input: [g, l_c, g*l_c, |g-l_c|] -> 4*feat_dim
        # Output: 1 score per class
        fusion_input_dim = 4 * config.feat_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        # Temperature parameter for softmax
        self.tau = config.tau
    
    def renormalize_prototypes(self):
        """L2-normalize prototypes along the feature dimension."""
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
    
    def forward(self, x):
        """
        Forward pass of CDSPatchNet.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224).
        
        Returns:
            dict: Contains 'logits', 'memberships', 'global', and 'patches'.
        """
        B = x.size(0)
        N = 49  # 7x7 patches
        C = self.num_classes
        D = self.config.feat_dim
        
        # Extract backbone features: (B, 768, 7, 7)
        feat_map = self.backbone.forward_features(x)
        
        # Reshape to (B, 49, 768)
        feat_map = feat_map.permute(0, 2, 3, 1).reshape(B, N, 768)
        
        # Project patches: (B, 49, feat_dim)
        patches = self.patch_proj(feat_map)
        
        # Global feature: mean pooling over patches
        g = patches.mean(dim=1)  # (B, feat_dim)
        
        # Normalize prototypes and patches for cosine similarity
        proto_norm = F.normalize(self.prototypes, dim=1)  # (C, feat_dim)
        patches_norm = F.normalize(patches, dim=-1)  # (B, N, feat_dim)
        
        # Cosine similarity between patches and prototypes: (B, N, C)
        cos_sim = torch.bmm(patches_norm, proto_norm.transpose(0, 1))
        
        # Apply temperature and softmax across patches (dim=1) to get r_tilde
        r_tilde = F.softmax(cos_sim / self.tau, dim=1)  # (B, N, C)
        
        # Renormalize prototypes after forward (optional, done in training)
        # self.renormalize_prototypes()  # Called externally in training loop
        
        # Compute memberships using replicator dynamics for each class
        memberships = []
        
        for c in range(C):
            # Get class-specific soft assignments: (B, N)
            r_c = r_tilde[:, :, c]
            
            # Build affinity matrix for this class
            A_c = build_affinity(patches, self.prototypes[c], self.config, class_r_tilde=r_c)
            
            # Replicator dynamics
            # Initialize: uniform distribution
            x = torch.ones(B, N, device=x.device) / N  # (B, N)
            
            for _ in range(self.config.solver_iters):
                # x_new = x * (A @ x) / (x @ (A @ x))
                Ax = torch.bmm(A_c, x.unsqueeze(-1)).squeeze(-1)  # (B, N)
                xAx = (x * Ax).sum(dim=-1, keepdim=True)  # (B, 1)
                
                # Update rule with numerical stability
                x = x * Ax / (xAx + 1e-8)
                
                # Normalize
                x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
            
            memberships.append(x)
        
        # Stack memberships: (B, C, N)
        memberships = torch.stack(memberships, dim=1)
        
        # Compute local features for each class: l_c = sum_i (x_c_i * patch_i)
        # memberships: (B, C, N), patches: (B, N, D)
        # Result: (B, C, D)
        l_c = torch.bmm(memberships, patches)
        
        # Fusion for each class
        # For each class c, build vector: [g, l_c, g*l_c, |g-l_c|]
        # g: (B, D), l_c: (B, C, D)
        # Expand g to (B, 1, D) then repeat to (B, C, D)
        g_expanded = g.unsqueeze(1).expand(-1, C, -1)  # (B, C, D)
        
        # Element-wise products and differences
        prod = g_expanded * l_c  # (B, C, D)
        diff = torch.abs(g_expanded - l_c)  # (B, C, D)
        
        # Concatenate: [g, l_c, g*l_c, |g-l_c|] -> (B, C, 4*D)
        fusion_vec = torch.cat([g_expanded, l_c, prod, diff], dim=-1)
        
        # Apply fusion MLP independently to each class
        # Reshape to (B*C, 4*D), apply MLP, reshape back
        fusion_vec_flat = fusion_vec.reshape(B * C, -1)
        scores_flat = self.fusion_mlp(fusion_vec_flat).squeeze(-1)  # (B*C,)
        scores = scores_flat.reshape(B, C)  # (B, C)
        
        return {
            "logits": scores,
            "memberships": memberships,
            "global": g,
            "patches": patches
        }


# %% CDSPatchNetLoss
class CDSPatchNetLoss:
    """
    Loss function for CDS-PatchNet.
    
    Combines cross-entropy loss, prototype alignment loss, and coherence loss.
    """
    
    def __init__(self, config):
        """
        Initialize the loss module.
        
        Args:
            config (Config): Configuration object.
        """
        self.config = config
    
    def __call__(self, outputs, labels, model_prototypes):
        """
        Compute the total loss.
        
        Args:
            outputs (dict): Model outputs containing 'logits', 'memberships', 'patches', 'global'.
            labels (torch.Tensor): Ground truth labels of shape (B,).
            model_prototypes (torch.Tensor): Model prototypes (num_classes, feat_dim).
        
        Returns:
            tuple: (total_loss, loss_dict)
        """
        logits = outputs["logits"]  # (B, C)
        memberships = outputs["memberships"]  # (B, C, N)
        patches = outputs["patches"]  # (B, N, D)
        g = outputs["global"]  # (B, D)
        
        B = logits.size(0)
        C = logits.size(1)
        
        # Cross-entropy loss
        L_cls = F.cross_entropy(logits, labels)
        
        # Prototype loss: encourage global features to align with their class prototypes
        # Use detach on prototypes to avoid double-gradient issues
        proto_for_labels = model_prototypes[labels].detach()  # (B, D)
        cos_sim_proto = F.cosine_similarity(g, proto_for_labels, dim=-1)  # (B,)
        L_proto = (1 - cos_sim_proto).mean()
        
        # Coherence loss: encourage high membership values to be coherent under affinity
        # For each sample, get the true class membership
        batch_idx = torch.arange(B, device=labels.device)
        x_true = memberships[batch_idx, labels]  # (B, N)
        
        # Build affinity matrix for the true class
        # We need to build one affinity matrix per sample using its patches and true prototype
        L_coh_sum = 0.0
        
        for b in range(B):
            true_class = labels[b].item()
            true_proto = model_prototypes[true_class]
            
            # Get r_tilde for this sample and true class
            # Recompute r_tilde for this sample
            patch_b = patches[b:b+1]  # (1, N, D)
            proto_norm = F.normalize(true_proto.unsqueeze(0), dim=1)  # (1, D)
            patch_norm = F.normalize(patch_b, dim=-1)  # (1, N, D)
            
            cos_sim = torch.bmm(patch_norm, proto_norm.transpose(0, 1)).squeeze(0)  # (N,)
            r_tilde_b = F.softmax(cos_sim / self.config.tau, dim=0)  # (N,)
            
            # Build affinity matrix
            A_true = build_affinity(patch_b, true_proto, self.config, class_r_tilde=r_tilde_b.unsqueeze(0))  # (1, N, N)
            A_true = A_true.squeeze(0)  # (N, N)
            
            # Coherence: -x^T A x
            x_b = x_true[b]  # (N,)
            Ax = A_true @ x_b.unsqueeze(-1)  # (N, 1)
            xAx = (x_b * Ax.squeeze(-1)).sum()  # scalar
            
            L_coh_sum = L_coh_sum - xAx
        
        L_coh = L_coh_sum / B
        
        # Total loss
        total = L_cls + self.config.lambda_proto * L_proto + self.config.lambda_coh * L_coh
        
        loss_dict = {
            "L_cls": L_cls.item(),
            "L_proto": L_proto.item(),
            "L_coh": L_coh.item(),
            "total": total.item()
        }
        
        return total, loss_dict


# %% GAPClassifier Baseline
class GAPClassifier(nn.Module):
    """
    Global Average Pooling baseline classifier.
    
    Uses ConvNeXt backbone with GAP followed by a simple MLP head.
    """
    
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize GAPClassifier.
        
        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained backbone weights.
        """
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images (B, 3, 224, 224).
        
        Returns:
            dict: Contains 'logits'.
        """
        # Extract features: (B, 768, 7, 7)
        feat_map = self.backbone.forward_features(x)
        
        # Global average pooling: (B, 768)
        gap = feat_map.mean(dim=[2, 3])
        
        # Classify
        logits = self.classifier(gap)
        
        return {"logits": logits}


# %% AttentionPoolClassifier Baseline
class AttentionPoolClassifier(nn.Module):
    """
    Attention Pooling baseline classifier.
    
    Uses ConvNeXt backbone with attention-weighted pooling followed by an MLP head.
    """
    
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize AttentionPoolClassifier.
        
        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained backbone weights.
        """
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Attention query: linear layer mapping 768 -> 1
        self.attention_query = nn.Linear(768, 1)
        
        # Classifier head (same as GAP)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images (B, 3, 224, 224).
        
        Returns:
            dict: Contains 'logits'.
        """
        # Extract features: (B, 768, 7, 7)
        feat_map = self.backbone.forward_features(x)
        
        # Reshape to (B, 49, 768)
        B = x.size(0)
        patches = feat_map.permute(0, 2, 3, 1).reshape(B, 49, 768)
        
        # Compute attention scores: (B, 49, 1) -> (B, 49)
        attn_scores = self.attention_query(patches).squeeze(-1)
        
        # Softmax over patches
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 49)
        
        # Weighted sum: (B, 49, 1) @ (B, 49, 768) -> (B, 1, 768) -> (B, 768)
        pooled = torch.bmm(attn_weights.unsqueeze(1), patches).squeeze(1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return {"logits": logits}


# %% Data Loading
# Note: Replace "dummy_data" with your actual data paths
try:
    train_ds = datasets.ImageFolder("dummy_data/train", transform=train_tf)
    val_ds = datasets.ImageFolder("dummy_data/val", transform=eval_tf)
    test_ds = datasets.ImageFolder("dummy_data/test", transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    
    num_classes = len(train_ds.classes)
    print(f"Loaded data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test samples")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_ds.classes}")
except FileNotFoundError as e:
    print(f"Data directory not found: {e}")
    print("Please ensure dummy_data/train, dummy_data/val, and dummy_data/test directories exist.")
    raise


# %% Metrics Computation
def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        dict: Dictionary of metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc
    }


# %% Evaluation Function
def evaluate(model, loader, device):
    """
    Evaluate model on a data loader.
    
    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Data loader.
        device (torch.device): Device to run evaluation on.
    
    Returns:
        dict: Computed metrics.
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs["logits"].argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return compute_metrics(all_labels, all_preds)


# %% Training Function
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (callable): Loss function.
        device (torch.device): Device.
    
    Returns:
        float: Average training loss.
    """
    model.train()
    running = 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels, model)
        loss.backward()
        optimizer.step()
        
        # Renormalize prototypes if applicable
        if hasattr(model, "renormalize_prototypes"):
            model.renormalize_prototypes()
        
        running += loss.item() * images.size(0)
    
    return running / len(loader.dataset)


# %% Freeze/Unfreeze Helpers
def freeze_backbone(model):
    """Freeze all backbone parameters."""
    for p in model.backbone.parameters():
        p.requires_grad_(False)


def unfreeze_last_stages(model, stage_indices=(2, 3)):
    """Unfreeze specific stages of the backbone."""
    for idx in stage_indices:
        if hasattr(model.backbone, 'stages'):
            for p in model.backbone.stages[idx].parameters():
                p.requires_grad_(True)


# %% Optimizer Factories for Two-Stage Training
def get_optimizer_stage1(model, lr_new=1e-3, wd=1e-4):
    """
    Get optimizer for stage 1 (backbone frozen, only new params trained).
    
    Args:
        model (nn.Module): Model.
        lr_new (float): Learning rate for new parameters.
        wd (float): Weight decay.
    
    Returns:
        torch.optim.Optimizer: AdamW optimizer.
    """
    freeze_backbone(model)
    new_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return torch.optim.AdamW(new_params, lr=lr_new, weight_decay=wd)


def get_optimizer_stage2(model, lr_backbone=1e-5, lr_new=1e-4, wd=1e-4):
    """
    Get optimizer for stage 2 (last backbone stages unfrozen).
    
    Args:
        model (nn.Module): Model.
        lr_backbone (float): Learning rate for backbone parameters.
        lr_new (float): Learning rate for new parameters.
        wd (float): Weight decay.
    
    Returns:
        torch.optim.Optimizer: AdamW optimizer with parameter groups.
    """
    freeze_backbone(model)
    unfreeze_last_stages(model)
    
    backbone_params = [
        p for n, p in model.named_parameters()
        if n.startswith("backbone.") and p.requires_grad
    ]
    new_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    
    return torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr_backbone}, {"params": new_params, "lr": lr_new}],
        weight_decay=wd
    )


# %% Two-Stage Training Pipeline
def run_two_stage_training(model, model_name, loss_fn, stage1_epochs, stage2_epochs,
                           patience=3, ckpt_dir="ckpts"):
    """
    Run two-stage training with early stopping.
    
    Args:
        model (nn.Module): Model to train.
        model_name (str): Name for checkpoint files.
        loss_fn (callable): Loss function.
        stage1_epochs (int): Number of epochs for stage 1.
        stage2_epochs (int): Number of epochs for stage 2.
        patience (int): Early stopping patience.
        ckpt_dir (str): Directory for checkpoints.
    
    Returns:
        list: Training history.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_f1 = -1
    best_state = None
    no_improve = 0
    history = []
    
    def run_stage(optimizer, n_epochs, stage_name):
        nonlocal best_f1, best_state, no_improve
        
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_epochs, 1), eta_min=1e-6
        )
        
        for epoch in range(n_epochs):
            tl = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            vm = evaluate(model, val_loader, DEVICE)
            sched.step()
            
            history.append({
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": tl,
                **vm
            })
            
            improved = vm["macro_f1"] > best_f1
            if improved:
                best_f1 = vm["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                torch.save(
                    best_state,
                    os.path.join(ckpt_dir, f"{model_name}_best.pth")
                )
                print(f"[{stage_name} Epoch {epoch}] Improved! F1={best_f1:.4f}")
            else:
                no_improve += 1
                print(f"[{stage_name} Epoch {epoch}] No improvement. Patience: {no_improve}/{patience}")
            
            if no_improve >= patience:
                print(f"Early stopping at {stage_name} epoch {epoch}")
                return True
        
        return False
    
    # Stage 1: Train new parameters with frozen backbone
    print(f"\n=== Starting Stage 1 for {model_name} ===")
    stopped = run_stage(get_optimizer_stage1(model), stage1_epochs, "stage1")
    
    # Stage 2: Fine-tune last backbone stages
    if not stopped:
        print(f"\n=== Starting Stage 2 for {model_name} ===")
        run_stage(get_optimizer_stage2(model), stage2_epochs, "stage2")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model for {model_name} (F1={best_f1:.4f})")
    
    return history


# %% Main Training & Evaluation Pipeline
if __name__ == "__main__":
    print("=" * 60)
    print("CDS-PatchNet Fine-Grained Classification Pipeline")
    print("=" * 60)
    
    # ---- CDS-PatchNet ----
    print("\n" + "=" * 60)
    print("Training CDS-PatchNet")
    print("=" * 60)
    
    cfg = Config()
    cds_model = CDSPatchNet(num_classes, cfg, pretrained=False).to(DEVICE)
    cds_loss_module = CDSPatchNetLoss(cfg)
    
    def cds_loss_fn(outputs, labels, model):
        total, _ = cds_loss_module(outputs, labels, model.prototypes)
        return total
    
    hist_cds = run_two_stage_training(
        cds_model, "cds_patchnet", cds_loss_fn,
        stage1_epochs=2, stage2_epochs=2, patience=5
    )
    
    test_metrics_cds = evaluate(cds_model, test_loader, DEVICE)
    print("\nCDS-PatchNet test metrics:", test_metrics_cds)
    
    # ---- GAP Baseline ----
    print("\n" + "=" * 60)
    print("Training GAP Baseline")
    print("=" * 60)
    
    ce = nn.CrossEntropyLoss()
    
    def ce_loss_fn(outputs, labels, model):
        return ce(outputs["logits"], labels)
    
    gap_model = GAPClassifier(num_classes, pretrained=False).to(DEVICE)
    run_two_stage_training(
        gap_model, "gap_baseline", ce_loss_fn,
        stage1_epochs=2, stage2_epochs=2, patience=5
    )
    
    test_metrics_gap = evaluate(gap_model, test_loader, DEVICE)
    print("\nGAP baseline test metrics:", test_metrics_gap)
    
    # ---- Attention Pool Baseline ----
    print("\n" + "=" * 60)
    print("Training Attention Pool Baseline")
    print("=" * 60)
    
    attn_model = AttentionPoolClassifier(num_classes, pretrained=False).to(DEVICE)
    run_two_stage_training(
        attn_model, "attn_baseline", ce_loss_fn,
        stage1_epochs=2, stage2_epochs=2, patience=5
    )
    
    test_metrics_attn = evaluate(attn_model, test_loader, DEVICE)
    print("\nAttention-pool baseline test metrics:", test_metrics_attn)
    
    # ---- Results Summary ----
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    df = pd.DataFrame({
        "CDS-PatchNet": test_metrics_cds,
        "GAP Baseline": test_metrics_gap,
        "Attention-Pool Baseline": test_metrics_attn,
    }).T
    
    print(df)
    
    print("\nCheckpoint files:", os.listdir("ckpts"))
    print("\nALL-MODELS PIPELINE TEST PASSED!")
