# ============================================================
# train_target_DeiT.py  (ResNet18-style version)
#
# ✅ Mimic your simple ResNet18 train_target_model() style:
#   - load CIFAR10/100
#   - random_split 80/20
#   - train target on 80%
#   - train shadow on 20%
#   - print epoch loss/acc
#   - return target_model, shadow_model, train_set, rest_set
#
# ✅ Still includes critical CIFAR-ViT fixes:
#   1) build_deit defined (fix NameError)
#   2) save_split_dir supported (fix unexpected keyword)
#   3) patch4_32 variant preferred for CIFAR32 (fix token mismatch)
# ============================================================

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    import timm
except ImportError as e:
    raise ImportError("This trainer needs timm. Install it via: pip install timm") from e


# ============================================================
# 0) Repro
# ============================================================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) DeiT/ViT builder (CIFAR32 patch4 fix)
# ============================================================
def build_deit(
    num_classes: int,
    img_size: int = 32,
    model_name: str = "deit_tiny_patch16_224",
    patch_size_fix: int = 4,
):
    """
    ✅ CIFAR32 critical fix:
    Prefer patch4_32 variants to avoid pos_embed/token mismatch.
    """
    candidates = []

    if img_size == 32 and patch_size_fix == 4:
        candidates += [
            "vit_tiny_patch4_32",
            "deit_tiny_patch4_32",
            "vit_small_patch4_32",
            "deit_small_patch4_32",
        ]

    candidates.append(model_name)

    last_err = None
    for name in candidates:
        try:
            m = timm.create_model(
                name,
                pretrained=False,
                num_classes=num_classes,
                img_size=img_size,
            )
            print(f"✅ build_deit: using timm model = {name}")
            return m
        except TypeError:
            # older timm might not accept img_size kwarg
            try:
                m = timm.create_model(
                    name,
                    pretrained=False,
                    num_classes=num_classes,
                )
                print(f"✅ build_deit: using timm model = {name} (no img_size kwarg)")
                return m
            except Exception as e2:
                last_err = e2
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"[build_deit] Failed. Tried={candidates}. Last error={repr(last_err)}"
    )


# ============================================================
# 2) Train loop (epoch loss/acc like your style)
# ============================================================
def train_loop(model, loader, device="cuda", num_epochs=50, lr=3e-4):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Loss={running_loss/len(loader):.4f} "
            f"Acc={correct/total:.4f}"
        )

    return model


# ============================================================
# 3) Main API (ResNet18-style)
# ============================================================
def train_target_model(
    batch_size=128,
    num_epochs=50,
    device="cuda",
    dataset_name="cifar10",
    seed=0,
    train_frac=0.8,
    data_root="./data",
    download=True,
    save_split_dir=None,   # ✅ compatibility with your big script
    deit_name="deit_tiny_patch16_224",
    img_size=32,
    patch_size_fix=4,
):
    """
    Returns:
        target_model, shadow_model, train_set, rest_set
    """
    set_seed(seed)

    # ---------- Transform (mimic your ResNet18 version) ----------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ---------- Dataset ----------
    if dataset_name.lower() == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=True, download=download, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=True, download=download, transform=transform)
        num_classes = 100
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'.")

    # ---------- Split (random_split, deterministic like your style) ----------
    N = len(ds)
    n_train = int(train_frac * N)
    n_rest = N - n_train

    g = torch.Generator()
    g.manual_seed(seed)

    train_set, rest_set = random_split(ds, [n_train, n_rest], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(rest_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # ✅ optional: save split indices for reuse
    if save_split_dir is not None:
        os.makedirs(save_split_dir, exist_ok=True)

        # random_split gives Subset objects with .indices
        train_idx = np.array(train_set.indices, dtype=np.int64)
        rest_idx = np.array(rest_set.indices, dtype=np.int64)

        np.savez(
            os.path.join(save_split_dir, f"split_indices_{dataset_name}_seed{seed}.npz"),
            train_idx=train_idx,
            query_idx=rest_idx
        )
        print(f"[Split Saved] -> {save_split_dir}/split_indices_{dataset_name}_seed{seed}.npz")

    # ---------- Build target model ----------
    target_model = build_deit(
        num_classes=num_classes,
        img_size=img_size,
        model_name=deit_name,
        patch_size_fix=patch_size_fix,
    )

    # ---------- Train target ----------
    print("🚀 Training target DeiT/ViT …")
    target_model = train_loop(
        model=target_model,
        loader=train_loader,
        device=device,
        num_epochs=num_epochs,
        lr=3e-4,
    )

    # ---------- Build shadow model ----------
    shadow_model = build_deit(
        num_classes=num_classes,
        img_size=img_size,
        model_name=deit_name,
        patch_size_fix=patch_size_fix,
    )

    # ---------- Train shadow ----------
    print("🚀 Training shadow DeiT/ViT …")
    shadow_model = train_loop(
        model=shadow_model,
        loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        lr=3e-4,
    )

    return target_model, shadow_model, train_set, rest_set


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, shadow, train_set, rest_set = train_target_model(
        batch_size=128,
        num_epochs=2,
        device=device,
        dataset_name="cifar10",
        seed=1,
        train_frac=0.8,
        save_split_dir="./checkpoints",
    )
    print("Done. train:", len(train_set), "rest:", len(rest_set))
