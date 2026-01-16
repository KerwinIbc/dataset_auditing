# ============================================================
# change_main_methodB_lambda_budget_shadow_then_target_CIFAR100.py
#
# CIFAR-100 version (100 classes)
#
# ✅ Matches the paper-figure logic:
#   Outer (dual) update:
#     λ_{k+1} = max(0, λ_k + η (mean_KL(λ_k) - C))
#
#   Inner updates (fixed λ_k during inner sampling):
#     - z: SGLD update (your existing SGLD form)
#     - θ: SAME as your previous update (Adam on generator, using NES dL/dx)
#
# 🔧 Key fix vs your pasted version:
#   - During burn-in, we DO NOT set λ = 0.
#     Burn-in only controls: step size schedule + whether we collect samples.
#     This aligns with "inner layer update uses λ_k" in the figure.
# ============================================================

import os
import copy
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from matplotlib.patches import Ellipse


# ---- your project modules ----
from train_ResNet18 import train_target_model
from train_cVAE import train_cVAE
from compute_information_plane import (
    compute_MI_full_dataset,
    compute_dataset_MI_balanced_with_MIcal
)

# ============================================================
# 0) Repro utilities
# ============================================================
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ============================================================
# 1) Estimate per-class prior r_y(t) (tau-aligned)
#    r_y(t) = mean_{(x,y)=y} softmax(logits(x)/tau)
# ============================================================
@torch.no_grad()
def estimate_per_class_prior(model, dataset, num_classes=10, batch_size=512, device="cuda", tau=1.0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    r_sum = torch.zeros(num_classes, num_classes, device=device)
    cnt = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits / tau, dim=1)

        # aggregate per class
        for c in range(num_classes):
            m = (y == c)
            if m.any():
                r_sum[c] += probs[m].sum(dim=0)
                cnt[c] += m.sum()

    r = r_sum / (cnt.unsqueeze(1) + 1e-8)
    r = torch.clamp(r, 1e-8, 1.0)
    r = r / r.sum(dim=1, keepdim=True)
    return r

# ============================================================
# 2) Budget C estimation on shadow data
#    C = E KL(p_sh(t|x) || r_y(t))
# ============================================================
@torch.no_grad()
def estimate_budget_C(
    shadow_model,
    dataset,
    r_per_class,
    device="cuda",
    tau=0.7,
    batch_size=512,
    conf_filter=True,
    conf_th=0.90
):
    shadow_model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_kl = 0.0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = shadow_model(x)
        probs = torch.softmax(logits / tau, dim=1)
        conf, pred = probs.max(dim=1)

        prior = r_per_class[y]  # [B,C]
        kl = torch.sum(probs * (torch.log(probs + 1e-8) - torch.log(prior + 1e-8)), dim=1)

        if conf_filter:
            keep = (pred == y) & (conf >= conf_th)
            if keep.any():
                total_kl += float(kl[keep].sum().item())
                total_n += int(keep.sum().item())
        else:
            total_kl += float(kl.sum().item())
            total_n += int(y.numel())

    if total_n == 0:
        return float("inf")

    return total_kl / total_n

# ============================================================
# 3) NES gradients (black-box)
# ============================================================
def estimate_nes_gradient_x(
    queried_model,
    imgs,            # [B,C,H,W]
    target_labels,   # [B]
    sigma=0.01,
    q_directions=128,
    lambda_comp=0.01,
    r_per_class=None,
    tau=0.7
):
    """
    Returns an estimate of d/dx [ Lacc(x) + lambda_comp * Lcomp(x) ].
    """
    B, C, H, W = imgs.shape
    device = imgs.device

    u = torch.randn(q_directions, B, C, H, W, device=device)
    imgs_exp = imgs.unsqueeze(0).expand(q_directions, -1, -1, -1, -1)

    imgs_pos = (imgs_exp + sigma * u).reshape(-1, C, H, W)
    imgs_neg = (imgs_exp - sigma * u).reshape(-1, C, H, W)
    batch_query = torch.cat([imgs_pos, imgs_neg], dim=0)

    labels_query = target_labels.view(1, B).expand(2 * q_directions, B).reshape(-1)

    with torch.no_grad():
        logits = queried_model(batch_query)
        probs = torch.softmax(logits / tau, dim=1)
        log_probs = torch.log(probs + 1e-8)

        loss_acc = F.nll_loss(log_probs, labels_query, reduction='none')

        if r_per_class is None:
            prior = torch.full_like(probs, 1.0 / probs.size(1))
        else:
            prior = r_per_class[labels_query]

        loss_comp = torch.sum(
            probs * (torch.log(probs + 1e-8) - torch.log(prior + 1e-8)), dim=1
        )

        loss_total = loss_acc + lambda_comp * loss_comp

        loss_pos = loss_total[:q_directions * B].view(q_directions, B)
        loss_neg = loss_total[q_directions * B:].view(q_directions, B)

        diff = (loss_pos - loss_neg).view(q_directions, B, 1, 1, 1)
        grad_x = torch.mean(diff * u, dim=0) / (2 * sigma)

    return grad_x

def estimate_nes_gradient_z(
    queried_model,
    generator,
    z,                 # [Bz,z_dim]
    labels,            # [Bz]
    sigma=0.01,
    q_directions=64,
    lambda_comp=0.01,
    r_per_class=None,
    tau=0.7,
    device="cuda"
):
    """
    Returns an estimate of d/dz [ Lacc(G(z)) + lambda_comp * Lcomp(G(z)) ].
    """
    Bz, z_dim = z.shape
    u = torch.randn(q_directions, Bz, z_dim, device=device)

    z_exp = z.unsqueeze(0).expand(q_directions, -1, -1)
    z_pos = z_exp + sigma * u
    z_neg = z_exp - sigma * u

    batch_z = torch.cat([z_pos, z_neg], dim=0).reshape(-1, z_dim)
    labels_q = labels.view(1, Bz).expand(2 * q_directions, Bz).reshape(-1)

    with torch.no_grad():
        try:
            imgs = generator.decode(batch_z, labels_q)
        except Exception:
            imgs = generator(batch_z, labels_q)

        logits = queried_model(imgs)
        probs = torch.softmax(logits / tau, dim=1)
        log_probs = torch.log(probs + 1e-8)

        loss_acc = F.nll_loss(log_probs, labels_q, reduction='none')

        if r_per_class is None:
            prior = torch.full_like(probs, 1.0 / probs.size(1))
        else:
            prior = r_per_class[labels_q]

        loss_comp = torch.sum(
            probs * (torch.log(probs + 1e-8) - torch.log(prior + 1e-8)), dim=1
        )

        loss_total = loss_acc + lambda_comp * loss_comp

    loss_pos = loss_total[:q_directions * Bz].view(q_directions, Bz)
    loss_neg = loss_total[q_directions * Bz:].view(q_directions, Bz)
    diff = (loss_pos - loss_neg).view(q_directions, Bz, 1)

    grad_z = torch.mean(diff * u, dim=0) / (2 * sigma)
    return grad_z

# ============================================================
# 4) Inversion core (z = SGLD, theta = "previous update" i.e., Adam)
# ============================================================
def ib_sgld_blackbox_nes_multichain(
    queried_model,
    generator,
    target_class_idx,
    r_per_class=None,
    device="cuda",
    z_dim=128,
    num_samples=300,
    total_steps=1200,
    burn_in=250,
    num_chains=32,
    sample_every=10,
    conf_th=0.90,
    tau=0.7,
    epsilon_hi=0.08,
    epsilon_lo=0.02,
    sigma=0.01,
    q_directions=32,
    q_img_grad=64,
    lambda_prior=0.001,
    update_generator=True,
    lr_g=1e-4,
    lambda_policy="fixed",   # "fixed" or "jitter"
    lambda_fixed=0.01,
    lambda_center=0.01,
    lambda_delta=0.005,
    refresh_every=200,
    refresh_frac=0.125
):
    """
    Inner-loop (fixed lambda during this run):
      - z: SGLD (your existing discretization)
      - theta: Adam (previous update), using NES dL/dx then backprop to theta
    """
    generator.train() if update_generator else generator.eval()
    queried_model.eval()

    opt_g = None
    if update_generator:
        opt_g = optim.Adam(generator.parameters(), lr=lr_g)

    Bz = num_chains
    z = torch.randn(Bz, z_dim, device=device)
    labels = torch.full((Bz,), target_class_idx, device=device, dtype=torch.long)

    collected = []

    for t in range(total_steps):
        # --------------------------------------------------------
        # IMPORTANT FIX:
        #   λ is NOT forced to 0 during burn-in.
        #   Burn-in only affects eps schedule & sample collection.
        # --------------------------------------------------------
        if lambda_policy == "fixed":
            lam_t = float(lambda_fixed)
        else:
            lo = max(0.0, float(lambda_center - lambda_delta))
            hi = float(lambda_center + lambda_delta)
            lam_t = float(np.random.uniform(lo, hi))

        # step size schedule for SGLD in z-space
        eps_t = epsilon_hi if t < burn_in else epsilon_lo

        # ---- NES gradient in z-space (black-box) ----
        grad_z = estimate_nes_gradient_z(
            queried_model=queried_model,
            generator=generator,
            z=z,
            labels=labels,
            sigma=sigma,
            q_directions=q_directions,
            lambda_comp=lam_t,
            r_per_class=r_per_class,
            tau=tau,
            device=device
        )

        # ---- SGLD update for z (your original form) ----
        with torch.no_grad():
            noise = torch.randn_like(z) * math.sqrt(eps_t)
            z.copy_(z - 0.5 * eps_t * (grad_z + lambda_prior * z) + noise)

            # optional chain refresh (helps mixing after burn-in)
            if refresh_every > 0 and (t % refresh_every == 0) and (t > 0):
                k = max(1, int(Bz * refresh_frac))
                idx = torch.randperm(Bz, device=device)[:k]
                z[idx] = torch.randn_like(z[idx])

        # ---- theta update (keep your previous update = Adam) ----
        if update_generator:
            opt_g.zero_grad(set_to_none=True)

            try:
                img_main = generator.decode(z, labels)
            except Exception:
                img_main = generator(z, labels)

            # NES estimate for d/dx [Lacc + lam_t * Lcomp]
            grad_img = estimate_nes_gradient_x(
                queried_model=queried_model,
                imgs=img_main,
                target_labels=labels,
                sigma=sigma,
                q_directions=q_img_grad,
                lambda_comp=lam_t,
                r_per_class=r_per_class,
                tau=tau
            )

            # Backprop grad_img through generator to get grads on theta, then Adam step
            img_main.backward(grad_img)
            opt_g.step()
        else:
            with torch.no_grad():
                try:
                    img_main = generator.decode(z, labels)
                except Exception:
                    img_main = generator(z, labels)

        # ---- only collect AFTER burn-in ----
        if t > burn_in and (t % sample_every == 0):
            with torch.no_grad():
                logits = queried_model(img_main)
                probs = torch.softmax(logits / tau, dim=1)
                conf, pred = probs.max(dim=1)
                keep = (pred == labels) & (conf >= conf_th)

            if keep.any():
                collected.append(img_main[keep].detach().cpu())

            n_collected = sum(x.size(0) for x in collected)
            if n_collected >= num_samples:
                break

    if len(collected) == 0:
        return torch.empty(0)

    return torch.cat(collected, dim=0)[:num_samples]

# ============================================================
# 5) Evaluate mean KL + mean conf
# ============================================================
@torch.no_grad()
def eval_conf_and_kl(model, imgs, labels, r_per_class, tau=0.7, device="cuda"):
    if imgs.numel() == 0:
        return dict(mean_conf=0.0, mean_kl=float("inf"), n=0)

    model.eval()
    imgs = imgs.to(device)
    labels = labels.to(device)

    logits = model(imgs)
    probs = torch.softmax(logits / tau, dim=1)
    conf, pred = probs.max(dim=1)

    prior = r_per_class[labels]
    kl = torch.sum(probs * (torch.log(probs + 1e-8) - torch.log(prior + 1e-8)), dim=1)

    return dict(
        mean_conf=float(conf.mean().item()),
        mean_kl=float(kl.mean().item()),
        n=int(labels.numel())
    )

# ============================================================
# 6) Dual-ascent calibration of lambda on SHADOW
# ============================================================
def calibrate_lambda_dual_on_shadow(
    shadow_model,
    generator,
    r_per_class_shadow,
    C_budget,
    device="cuda",
    tau=0.7,
    conf_th=0.90,
    lambda_init=0.01,
    eta=0.5,
    max_iters=12,
    tol_rel=0.10,
    classes=10,
    eval_classes=None,              # None => use all; else a subset list/tuple
    eval_samples_per_class=80,
    eval_total_steps=800,
    eval_burn_in=160,
    num_chains=24,
    z_dim=128,
    sigma=0.01,
    q_directions=32,
    q_img_grad=64,
    epsilon_hi=0.08,
    epsilon_lo=0.02,
    lambda_prior=0.001,
    update_generator=False,         # usually keep False during calibration for stability
    lr_g=1e-4,
):
    if eval_classes is None:
        eval_classes = list(range(classes))
    else:
        eval_classes = list(eval_classes)

    lam = float(lambda_init)
    history = []

    print("\n🧭 [Method-B] Dual-ascent calibrating λ on SHADOW ...")
    print(f"   Budget C = {C_budget:.6f} | tau={tau} | conf_th={conf_th}")
    print(f"   classes={classes} | eval_classes={len(eval_classes)} | update_generator={update_generator}")

    for it in range(max_iters):
        conf_list, kl_list, n_list = [], [], []

        # deep copy generator per outer iteration (keeps runs comparable)
        gen_tmp = copy.deepcopy(generator).to(device)

        for cls in eval_classes:
            imgs = ib_sgld_blackbox_nes_multichain(
                queried_model=shadow_model,
                generator=gen_tmp,
                target_class_idx=cls,
                r_per_class=r_per_class_shadow,
                device=device,
                z_dim=z_dim,
                num_samples=eval_samples_per_class,
                total_steps=eval_total_steps,
                burn_in=eval_burn_in,
                num_chains=num_chains,
                sample_every=10,
                conf_th=conf_th,
                tau=tau,
                epsilon_hi=epsilon_hi,
                epsilon_lo=epsilon_lo,
                sigma=sigma,
                q_directions=q_directions,
                q_img_grad=q_img_grad,
                lambda_prior=lambda_prior,
                update_generator=update_generator,
                lr_g=lr_g,
                lambda_policy="fixed",
                lambda_fixed=lam
            )

            if imgs.numel() == 0:
                continue

            labels = torch.full((imgs.size(0),), cls, dtype=torch.long)
            m = eval_conf_and_kl(
                model=shadow_model,
                imgs=imgs,
                labels=labels,
                r_per_class=r_per_class_shadow,
                tau=tau,
                device=device
            )

            conf_list.append(m["mean_conf"])
            kl_list.append(m["mean_kl"])
            n_list.append(m["n"])

        if len(kl_list) == 0:
            rec = dict(iter=it, lambda_val=lam, mean_conf=0.0, mean_kl=float("inf"), rel_err=float("inf"))
            history.append(rec)
            print(f"  iter={it:02d} | λ={lam:.6f} | collected=0 | (try lower conf_th or more steps)")
            lam = max(0.0, lam * 0.5)
            continue

        mean_conf = float(np.mean(conf_list))
        mean_kl = float(np.mean(kl_list))
        rel_err = abs(mean_kl - C_budget) / (abs(C_budget) + 1e-12)

        rec = dict(iter=it, lambda_val=lam, mean_conf=mean_conf, mean_kl=mean_kl, rel_err=rel_err)
        history.append(rec)

        print(f"  iter={it:02d} | λ={lam:.6f} | mean_conf={mean_conf:.4f} | "
              f"mean_KL={mean_kl:.6f} | rel_err={rel_err:.3f}")

        if rel_err <= tol_rel:
            print(f"\n✅ Converged: |KL-C|/C <= {tol_rel}. λ*={lam:.6f}")
            break

        # Dual ascent update (projected)
        lam = max(0.0, lam + eta * (mean_kl - C_budget))
        lam = float(min(lam, 5.0))  # clamp for safety

    return float(lam), history

# ============================================================
# 7) Runner helper
# ============================================================
def run_inversion_experiment(
    method_name,
    queried_model,
    generator,
    r_per_class,
    classes=10,
    samples_per_class=2000,
    device="cuda",
    save_dir=None,
    **kwargs
):
    print(f"\n🚀 Running: {method_name}")
    if save_dir is not None:
        ensure_dir(save_dir)

    all_imgs, all_labels = [], []
    for cls in range(classes):
        imgs = ib_sgld_blackbox_nes_multichain(
            queried_model=queried_model,
            generator=generator,
            target_class_idx=cls,
            r_per_class=r_per_class,
            device=device,
            num_samples=samples_per_class,
            **kwargs
        )

        if imgs.numel() == 0:
            print(f"  class {cls}: got 0 samples")
            continue

        labels = torch.full((imgs.size(0),), cls, dtype=torch.long)
        all_imgs.append(imgs)
        all_labels.append(labels)

        print(f"  class {cls}: collected {imgs.size(0)}")

        if save_dir is not None:
            cdir = os.path.join(save_dir, f"class_{cls}")
            ensure_dir(cdir)
            take = min(10, imgs.size(0))
            idx = torch.randperm(imgs.size(0))[:take]
            imgs_to_save = imgs[idx]
            imgs_to_save = torch.clamp(imgs_to_save * 0.5 + 0.5, 0, 1)
            for k in range(take):
                save_image(imgs_to_save[k], os.path.join(cdir, f"sample_{k}.png"))

    if len(all_imgs) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    return torch.cat(all_imgs, dim=0), torch.cat(all_labels, dim=0)

# ============================================================
# 8) Main
# ============================================================
def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    TAU = 0.7
    CLASSES = 10

    BUDGET_CONF_FILTER = True
    BUDGET_CONF_TH = 0.90

    LAMBDA_DELTA = 0.005

    FINAL_SAMPLES_PER_CLASS = 2000

    # -------------------------
    # 1) Train models + split data (CIFAR-100)
    # -------------------------
    target_model, shadow_model, train_set, query_set = train_target_model(device=device)
    target_model.eval()
    shadow_model.eval()

    # -------------------------
    # 2) Train generator on query_set (CIFAR-100)
    # -------------------------
    generator = train_cVAE(query_set, device=device)

    # -------------------------
    # 3) Estimate shadow prior r_y(t)
    # -------------------------
    print("\n📌 Estimating r_y(t) from SHADOW on query_set (tau-aligned) ...")
    r_per_class_shadow = estimate_per_class_prior(
        model=shadow_model,
        dataset=query_set,
        num_classes=CLASSES,
        batch_size=512,
        device=device,
        tau=TAU
    )
    print("   r_per_class_shadow:", tuple(r_per_class_shadow.shape))

    # -------------------------
    # 4) (B1) Estimate budget C
    # -------------------------
    print("\n🧮 (B1) Estimating budget C from shadow_model + query_set ...")
    C_budget = estimate_budget_C(
        shadow_model=shadow_model,
        dataset=query_set,
        r_per_class=r_per_class_shadow,
        device=device,
        tau=TAU,
        batch_size=512,
        conf_filter=BUDGET_CONF_FILTER,
        conf_th=BUDGET_CONF_TH
    )
    print(f"   Budget C = {C_budget:.6f}  (conf_filter={BUDGET_CONF_FILTER}, conf_th={BUDGET_CONF_TH})")

    # -------------------------
    # 5) (B2) Calibrate lambda* (subset classes for speed)
    # -------------------------
    eval_subset = list(range(10))  # 0..9

    lambda_star, hist = calibrate_lambda_dual_on_shadow(
        shadow_model=shadow_model,
        generator=generator,
        r_per_class_shadow=r_per_class_shadow,
        C_budget=C_budget,
        device=device,
        tau=TAU,
        conf_th=0.90,
        lambda_init=0.01,
        eta=0.6,
        max_iters=60,
        tol_rel=0.10,
        classes=CLASSES,
        eval_classes=eval_subset,
        eval_samples_per_class=80,
        eval_total_steps=800,
        eval_burn_in=160,
        num_chains=24,
        update_generator=False  # keep generator fixed during calibration
    )

    ensure_dir("results")
    with open("results/methodB_lambda_dual_history_cifar100.txt", "w") as f:
        for r in hist:
            f.write(str(r) + "\n")

    print(f"\n✅ (B2) Calibrated λ* = {lambda_star:.6f}")
    print(f"   Will use λ in [{max(0.0, lambda_star-LAMBDA_DELTA):.6f}, {lambda_star+LAMBDA_DELTA:.6f}] on target.")

    # -------------------------
    # 6) (B3) Final inversion on TARGET
    # -------------------------
    gen_final = copy.deepcopy(generator).to(device)

    imgs_gen, labels_gen = run_inversion_experiment(
        method_name="Method-B Final (CIFAR-100)",
        queried_model=target_model,
        generator=gen_final,
        r_per_class=r_per_class_shadow,
        classes=CLASSES,
        samples_per_class=FINAL_SAMPLES_PER_CLASS,
        device=device,
        save_dir="results/blackbox_methodB_cifar100",
        z_dim=128,
        total_steps=3500,
        burn_in=800,
        num_chains=64,
        sample_every=10,
        conf_th=0.90,
        tau=TAU,
        epsilon_hi=0.08,
        epsilon_lo=0.02,
        sigma=0.01,
        q_directions=64,
        q_img_grad=128,
        lambda_prior=0.001,
        update_generator=True,
        lr_g=1e-4,
        lambda_policy="jitter",
        lambda_center=lambda_star,
        lambda_delta=LAMBDA_DELTA,
        refresh_every=200,
        refresh_frac=0.125
    )

    # -------------------------
    # 7) Information Plane
    # -------------------------
    print("\n📊 Computing TRAIN MI ...")
    train_XT, train_TY = compute_MI_full_dataset(target_model, train_set, device=device)
    train_XT, train_TY = np.array(train_XT), np.array(train_TY)

    print("\n📊 Computing QUERY/TEST MI ...")
    test_XT, test_TY = compute_MI_full_dataset(target_model, query_set, device=device)
    test_XT, test_TY = np.array(test_XT), np.array(test_TY)

    print("\n📊 Computing GENERATED MI ...")
    ds_gen = TensorDataset(imgs_gen, labels_gen)

    gen_XT, gen_TY = compute_dataset_MI_balanced_with_MIcal(
        target_model, ds_gen, device=device,
        samples_per_class=1500,
        num_points=40
    )
    gen_XT, gen_TY = np.array(gen_XT), np.array(gen_TY)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(train_XT, train_TY, s=120, label="Train (target)")
    ax.scatter(test_XT, test_TY, s=120, label="Query/Test")
    ax.scatter(gen_XT, gen_TY, s=60, marker="^", label="Gen")

    gen_xy = np.stack([gen_XT, gen_TY], axis=1)
    cx, cy = gen_xy.mean(axis=0)

    # ---- axis-aligned ellipse: a=max|x-cx|, b=max|y-cy| ----
    dx = np.abs(gen_xy[:, 0] - cx)
    dy = np.abs(gen_xy[:, 1] - cy)
    a = float(dx.max() * 1.02)  # x half-axis
    b = float(dy.max() * 1.02)  # y half-axis

    ell = Ellipse(
        xy=(cx, cy),
        width=2 * a,
        height=2 * b,
        angle=0.0,  # no rotation
        fill=False,
        linewidth=2,
        linestyle="--"
    )
    ax.add_patch(ell)
    ax.scatter([cx], [cy], marker="x", s=60, label=f"Gen mean center (a={a:.4f}, b={b:.4f})")

    # expand axes to show ellipse fully
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(min(xmin, cx - a), max(xmax, cx + a))
    ax.set_ylim(min(ymin, cy - b), max(ymax, cy + b))

    ax.set_xlabel("I(X;T)")
    ax.set_ylabel("I(T;Y)")
    ax.set_title("Information Plane: CIFAR-10 (Train vs Query vs Generated)")
    ax.grid(True)
    ax.legend()

    out_png = "results/info_plane_methodB_cifar10_with_gen_circle.png"
    fig.savefig(out_png, dpi=200)
    print("\nSaved:", out_png)


if __name__ == "__main__":
    main()
