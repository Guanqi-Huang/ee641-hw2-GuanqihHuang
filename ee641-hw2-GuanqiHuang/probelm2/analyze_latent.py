from __future__ import annotations
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalVAE


def ensure_dirs(*paths: str | Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def upsample_pattern_grid(pats: torch.Tensor, scale: int = 16) -> torch.Tensor:
    b, H, W = pats.shape
    imgs = F.interpolate(pats.view(b,1,H,W), scale_factor=scale, mode="nearest")
    return imgs

def save_grid_bw(patterns: torch.Tensor, path: str, nrow: int = 10, scale: int = 16):
    ensure_dirs(Path(path).parent)
    imgs = upsample_pattern_grid(patterns.float(), scale=scale)
    B, C, H, W = imgs.shape
    rows = (B + nrow - 1) // nrow
    canvas = torch.zeros(1, rows*H, nrow*W, device=imgs.device)
    k = 0
    for r in range(rows):
        for c in range(nrow):
            if k >= B:
                break
            canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = imgs[k]
            k += 1
    arr = (canvas.squeeze(0).clamp(0,1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path) 

def sigmoid_logits_to_binary(logits: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thresh).float()

def compute_metrics(patterns: torch.Tensor) -> Dict[str, np.ndarray]:
    """
      density_total: mean hits per step (0..1)
      syncopation: off-beat hits ratio (off-beat = steps not divisible by 4)
      per-instrument activity: inst0..inst8 mean activations
    """
    X = patterns.float()  
    N = X.size(0)
    density_total = X.mean(dim=(1,2))                             # [N]
    onbeats = torch.zeros(16, dtype=torch.float32)
    onbeats[::4] = 1.0
    off_mask = (1.0 - onbeats).view(1,16,1)
    on_mask  = onbeats.view(1,16,1)
    off_hits = (X * off_mask).sum(dim=(1,2))
    on_hits  = (X * on_mask).sum(dim=(1,2))
    syncopation = off_hits / (off_hits + on_hits + 1e-8)          # [N]
    inst_means = X.mean(dim=1)                                    # [N,9]
    metrics = {
        "density_total": to_numpy(density_total),
        "syncopation": to_numpy(syncopation),
    }
    for j in range(9):
        metrics[f"inst{j}_mean"] = to_numpy(inst_means[:, j])
    return metrics

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float(np.clip((x*y).mean(), -1.0, 1.0))

def correlations_per_dim(latents: np.ndarray, metrics: Dict[str, np.ndarray]) -> Dict[str, List[Tuple[int,float]]]:
    """
    latents: [N,D]; returns dict
    """
    N, D = latents.shape
    out: Dict[str, List[Tuple[int,float]]] = {}
    for mname, mvals in metrics.items():
        cs = []
        for d in range(D):
            c = pearson(latents[:, d], mvals)
            cs.append((d, abs(c)))
        cs.sort(key=lambda t: t[1], reverse=True)
        out[mname] = cs
    return out

def try_tsne(Z: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, str]:
    try:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca",
                      random_state=641).fit_transform(Z)
        return coords, "tsne"
    except Exception:
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        coords = U[:, :2] * S[:2]
        return coords, "pca"

@torch.no_grad()
def encode_dataset(model: HierarchicalVAE, loader, device):
    mu_hs, mu_ls, labels, xs = [], [], [], []
    for x, _, s_idx in loader:
        x = x.to(device)
        z_h, mu_h, logv_h, z_l, mu_l, logv_l = model.enc(x)
        mu_hs.append(mu_h)
        mu_ls.append(mu_l)
        labels.append(s_idx.to(device))  # avoid torch.tensor(tensor) warning
        xs.append(x)
    mu_hs = torch.cat(mu_hs, 0).cpu()
    mu_ls = torch.cat(mu_ls, 0).cpu()
    labels = torch.cat(labels, 0).cpu()
    xs = torch.cat(xs, 0).cpu()
    return mu_hs, mu_ls, labels, xs

def style_centroids(mu_h: torch.Tensor, labels: torch.Tensor, num_styles: int) -> torch.Tensor:
    D = mu_h.shape[1]
    cents = torch.zeros(num_styles, D)
    for s in range(num_styles):
        mask = (labels == s)
        if mask.any():
            cents[s] = mu_h[mask].mean(dim=0)
    return cents

@torch.no_grad()
def generate_per_style(model: HierarchicalVAE, cents_h: torch.Tensor, n_per_style: int,
                       out_dir: Path, device: torch.device):
    S, Dh = cents_h.shape
    Dl = model.enc.mu_l.out_features if hasattr(model.enc, "mu_l") else 16
    for s in range(S):
        z_h = cents_h[s].to(device).unsqueeze(0).repeat(n_per_style, 1)
        z_l = torch.randn(n_per_style, Dl, device=device)
        logits = model.dec(z_h, z_l)           # [B,16,9]
        samp = sigmoid_logits_to_binary(logits)
        torch.save(samp.cpu(), out_dir / f"style_{s:02d}_samples.pt")
        save_grid_bw(samp.cpu(), out_dir / f"style_{s:02d}_samples.png", nrow=10)

@torch.no_grad()
def interpolation_sequences(model: HierarchicalVAE, cents_h: torch.Tensor, out_dir: Path,
                            device: torch.device, steps: int = 10):
    S, Dh = cents_h.shape
    Dl = model.enc.mu_l.out_features if hasattr(model.enc, "mu_l") else 16

    # Between two style centroids in z_h (fix z_l=0)
    if S >= 2:
        a, b = np.random.choice(S, 2, replace=False)
        zA, zB = cents_h[a].to(device), cents_h[b].to(device)
        label = f"style_{a}_to_{b}"
    else:
        zA, zB = torch.randn(Dh, device=device), torch.randn(Dh, device=device)
        label = "style_fallback_random"

    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z_h = (1 - alphas) * zA.unsqueeze(0) + alphas * zB.unsqueeze(0)
    z_l = torch.zeros(steps, Dl, device=device)
    seq_logits = model.dec(z_h, z_l)
    seq = sigmoid_logits_to_binary(seq_logits)
    torch.save(seq.cpu(), out_dir / f"interp_{label}.pt")
    save_grid_bw(seq.cpu(), out_dir / f"interp_{label}.png", nrow=steps)

    # Random z_h/z_l endpoints
    z0_h, z1_h = torch.randn(1, Dh, device=device), torch.randn(1, Dh, device=device)
    z0_l, z1_l = torch.randn(1, Dl, device=device), torch.randn(1, Dl, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1,1)
    z_h2 = (1 - alphas) * z0_h + alphas * z1_h
    z_l2 = (1 - alphas) * z0_l + alphas * z1_l
    seq_logits2 = model.dec(z_h2, z_l2)
    seq2 = sigmoid_logits_to_binary(seq_logits2)
    torch.save(seq2.cpu(), out_dir / "interp_random_latents.pt")
    save_grid_bw(seq2.cpu(), out_dir / "interp_random_latents.png", nrow=steps)

@torch.no_grad()
def style_transfer_examples(model: HierarchicalVAE, loader, cents_h: torch.Tensor,
                            out_dir: Path, device: torch.device, examples: int = 10):
    Dh = cents_h.shape[1]
    Dl = model.enc.mu_l.out_features if hasattr(model.enc, "mu_l") else 16
    origs, trans = [], []
    picked = 0
    for x, _, s_idx in loader:
        x = x.to(device)
        z_h, mu_h, logv_h, z_l, mu_l, logv_l = model.enc(x)
        for i in range(x.size(0)):
            src_style = int(s_idx[i])
            # pick a different target style
            targets = [t for t in range(cents_h.size(0)) if t != src_style]
            if len(targets) == 0:
                z_h_tgt = torch.randn(1, Dh, device=device)
            else:
                tgt_style = random.choice(targets)
                z_h_tgt = cents_h[tgt_style].to(device).unsqueeze(0)
            z_l_src = mu_l[i].unsqueeze(0)                               # [1,Dl] (use mean for stability)
            logits_tr = model.dec(z_h_tgt, z_l_src)                      # [1,16,9]
            patt_tr = sigmoid_logits_to_binary(logits_tr)
            origs.append((x[i].cpu() > 0.5).float().unsqueeze(0))        # binarize original
            trans.append(patt_tr.cpu())
            picked += 1
            if picked >= examples:
                break
        if picked >= examples:
            break
    if picked > 0:
        origs = torch.cat(origs, dim=0)   
        trans = torch.cat(trans, dim=0)  
        torch.save({"originals": origs, "transferred": trans}, out_dir / "style_transfer_examples.pt")
        both = torch.cat([origs, trans], dim=0)
        save_grid_bw(both, out_dir / "style_transfer_examples.png", nrow=10)

def scatter_plot(coords: np.ndarray, labels: np.ndarray, save_path: Path, title: str):
    plt.figure(figsize=(6.5,5.2))
    S = int(labels.max()) + 1 if labels.size else 0
    for s in range(S):
        m = labels == s
        plt.scatter(coords[m,0], coords[m,1], s=12, alpha=0.75, label=f"style {s}")
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    plt.title(title)
    if S <= 10:
        plt.legend(frameon=False, fontsize=9)
    ensure_dirs(save_path.parent)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170)
    plt.close()

def traversals(model: HierarchicalVAE, which: str, dims: List[int], out_dir: Path,
               device: torch.device, steps: int = 10, span: float = 2.5):
    Dh = model.enc.mu_h.out_features
    Dl = model.enc.mu_l.out_features
    base_h = torch.zeros(1, Dh, device=device)
    base_l = torch.zeros(1, Dl, device=device)
    alphas = torch.linspace(-span, span, steps, device=device).view(-1,1)

    for d in dims:
        if which == "z_h":
            z_h = base_h.repeat(steps,1); z_h[:, d] = alphas.squeeze(1)
            z_l = base_l.repeat(steps,1)
        else:
            z_h = base_h.repeat(steps,1)
            z_l = base_l.repeat(steps,1); z_l[:, d] = alphas.squeeze(1)
        logits = model.dec(z_h, z_l)
        pats = sigmoid_logits_to_binary(logits)
        save_grid_bw(pats.cpu(), out_dir / f"{which}_dim{d:02d}.png", nrow=steps)

def summarize_traversal_effects(model: HierarchicalVAE, which: str, dims: List[int],
                                device: torch.device, steps: int = 9, span: float = 2.5) -> Dict:
    Dh = model.enc.mu_h.out_features
    Dl = model.enc.mu_l.out_features
    base_h = torch.zeros(1, Dh, device=device)
    base_l = torch.zeros(1, Dl, device=device)
    vals = torch.linspace(-span, span, steps, device=device).view(-1,1)
    summary = {}
    for d in dims:
        if which == "z_h":
            z_h = base_h.repeat(steps,1); z_h[:, d] = vals.squeeze(1)
            z_l = base_l.repeat(steps,1)
        else:
            z_h = base_h.repeat(steps,1)
            z_l = base_l.repeat(steps,1); z_l[:, d] = vals.squeeze(1)
        with torch.no_grad():
            pats = sigmoid_logits_to_binary(model.dec(z_h, z_l)).cpu()    # [S,16,9]
        mets = compute_metrics(pats)
        # record metric ranges across traversal
        metric_ranges = {k: float(np.max(v)-np.min(v)) for k,v in mets.items()}
        # top-3 most-affected metrics
        top3 = sorted(metric_ranges.items(), key=lambda kv: kv[1], reverse=True)[:3]
        summary[f"{which}_dim{d}"] = {"top_metrics": top3, "span": span, "steps": steps}
    return summary

# ----------------------- main -----------------------

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
    res_dir = Path("results")
    gen_dir = res_dir / "generated_patterns"
    lat_dir = res_dir / "latent_analysis"
    ensure_dirs(gen_dir, lat_dir, lat_dir / "traversals")

    # data and loaders
    ds_val = DrumPatternDataset("../data/drums", split="val")
    ds_train = DrumPatternDataset("../data/drums", split="train")
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=256, shuffle=False)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=256, shuffle=False)

    # model
    ckpt = res_dir / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}. Train first with: python train.py")
    model = HierarchicalVAE()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    # encode val set to latent clouds & style centroids 
    mu_h, mu_l, y, x_val = encode_dataset(model, val_loader, device)  # [N,Dh], [N,Dl], [N], [N,16,9]
    S = int(y.max().item()) + 1 if y.numel() > 0 else 5
    cents_h = style_centroids(mu_h, y, S)                              # [S,Dh]
    torch.save({"mu_h": mu_h, "mu_l": mu_l, "labels": y}, lat_dir / "encodings_val.pt")

    # 10 samples per style
    generate_per_style(model, cents_h, n_per_style=10, out_dir=gen_dir, device=device)

    # interpolation sequences
    interpolation_sequences(model, cents_h, gen_dir, device, steps=10)

    # style transfer examples
    style_transfer_examples(model, val_loader, cents_h, gen_dir, device, examples=10)

    # t-SNE (or PCA) over mu_h 
    coords, method = try_tsne(to_numpy(mu_h), to_numpy(y))
    # save plot 
    scatter_plot(coords, to_numpy(y), lat_dir / f"tsne_latent_{method}.png",
                 title=f"Latent space ({method.upper()}) on μ_h")

    # Disentanglement analysis 
    # metrics from original val patterns
    mets = compute_metrics(x_val)                          
    # correlations for z_h and z_l
    corr_h = correlations_per_dim(to_numpy(mu_h), mets)    
    corr_l = correlations_per_dim(to_numpy(mu_l), mets)
    dis = {
        "method": "pearson_abs",
        "top_by_metric_z_h": {m: corr_h[m][:5] for m in corr_h},  # top-5 dims per metric
        "top_by_metric_z_l": {m: corr_l[m][:5] for m in corr_l},
    }
    with open(lat_dir / "disentanglement.json", "w") as f:
        json.dump(dis, f, indent=2)

    # Dimension interpretation
    # pick first few dims for each level (customize if you want)
    Dh = mu_h.shape[1]; Dl = mu_l.shape[1]
    dims_h = list(range(min(3, Dh)))
    dims_l = list(range(min(3, Dl)))
    traversals(model, "z_h", dims_h, lat_dir / "traversals", device, steps=10, span=2.5)
    traversals(model, "z_l", dims_l, lat_dir / "traversals", device, steps=10, span=2.5)
    summ = {}
    summ.update(summarize_traversal_effects(model, "z_h", dims_h, device))
    summ.update(summarize_traversal_effects(model, "z_l", dims_l, device))
    with open(lat_dir / "dimension_interpretation.json", "w") as f:
        json.dump(summ, f, indent=2)

if __name__ == "__main__":
    main()
