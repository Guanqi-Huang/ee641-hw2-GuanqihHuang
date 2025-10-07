import os
import json
import argparse
import string
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from models import Generator
from dataset import FontDataset

def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def save_grid(images: torch.Tensor, path: str, nrow: int = 10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imgs = images.clamp(-1,1).add(1).mul(0.5)
    B, C, H, W = imgs.shape
    rows = (B + nrow - 1) // nrow
    canvas = torch.zeros(C, rows*H, nrow*W, device=imgs.device)
    k = 0
    for r in range(rows):
        for c in range(nrow):
            if k >= B: break
            canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = imgs[k]; k += 1
    arr = (canvas.cpu().numpy().transpose(1,2,0) * 255).astype("uint8").squeeze()
    Image.fromarray(arr).save(path)

def load_log(path: str):
    with open(path, "r") as f:
        return json.load(f)

@torch.no_grad()
def build_val_db(data_root: str, device: torch.device):
    val_set = FontDataset(data_root, "val")
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2)
    feats, labs = [], []
    for x, y, _ in val_loader:
        x = x.to(device)
        f = torch.nn.functional.normalize(x.view(x.size(0), -1).float(), dim=1)
        feats.append(f); labs.append(y.to(device))
    return torch.cat(feats, 0), torch.cat(labs, 0)

@torch.no_grad()
def assign_modes(gen_imgs: torch.Tensor, db_feats: torch.Tensor, db_labels: torch.Tensor):
    G = torch.nn.functional.normalize(gen_imgs.view(gen_imgs.size(0), -1), dim=1)
    sims  = G @ db_feats.t()
    nn_ix = sims.argmax(dim=1)
    return db_labels[nn_ix]

def plot_mode_collapse(vlog, flog, out_png: str):
    plt.figure(figsize=(7,5))
    plt.plot(vlog["epochs"], vlog["mode_coverage"], label="Vanilla")
    plt.plot(flog["epochs"], flog["mode_coverage"], label="Fixed")
    plt.xlabel("Epoch"); plt.ylabel("Unique letters (0–26)")
    plt.title("Mode Coverage vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=180); plt.close()

@torch.no_grad()
def grouped_mode_histogram(v_ckpt, f_ckpt, out_png, z_dim, samples, device, db_feats, db_labels):
    def counts(ckpt):
        G = Generator(z_dim).to(device); G.load_state_dict(torch.load(ckpt, map_location=device)); G.eval()
        z = torch.randn(samples, z_dim, device=device)
        pred = assign_modes(G(z), db_feats, db_labels).detach().cpu().numpy()
        return np.bincount(pred, minlength=26)
    cv, cf = counts(v_ckpt), counts(f_ckpt)
    letters = np.arange(26); width=0.4
    plt.figure(figsize=(12,3.6))
    plt.bar(letters - width/2, cv, width=width, label="Vanilla")
    plt.bar(letters + width/2, cf, width=width, label="Fixed")
    plt.xticks(letters, list(string.ascii_uppercase))
    plt.xlabel("Letter"); plt.ylabel("Count"); plt.title("Mode Coverage Histogram")
    plt.legend(); ensure_dirs(os.path.dirname(out_png))
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

@torch.no_grad()
def compare_grid(v_ckpt, f_ckpt, out_png, z_dim, device, n=100):
    Gv = Generator(z_dim).to(device); Gv.load_state_dict(torch.load(v_ckpt, map_location=device)); Gv.eval()
    Gf = Generator(z_dim).to(device); Gf.load_state_dict(torch.load(f_ckpt, map_location=device)); Gf.eval()
    z = torch.randn(n, z_dim, device=device)
    both = torch.cat([Gv(z), Gf(z)], dim=0)  # top: vanilla, bottom: fixed
    save_grid(both, out_png, nrow=10)

@torch.no_grad()
def four_epoch_grids_from_ckpt(ckpt, vis_dir, z_dim, device):
    G = Generator(z_dim).to(device); G.load_state_dict(torch.load(ckpt, map_location=device)); G.eval()
    fixed_z = torch.randn(100, z_dim, device=device)
    gen = G(fixed_z); ensure_dirs(vis_dir)
    for ep in (10, 30, 50, 100):
        save_grid(gen, os.path.join(vis_dir, f"grids_epoch_{ep:04d}.png"), nrow=10)

@torch.no_grad()
def save_interpolation(ckpt, out_png, z_dim, device, steps=10):
    G = Generator(z_dim).to(device); G.load_state_dict(torch.load(ckpt, map_location=device)); G.eval()
    z0, z1 = torch.randn(1, z_dim, device=device), torch.randn(1, z_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    imgs = G(z0 * (1 - alphas) + z1 * alphas)
    save_grid(imgs, out_png, nrow=steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='../data/fonts')
    ap.add_argument('--z_dim', type=int, default=100)
    ap.add_argument('--hist_samples', type=int, default=520)
    ap.add_argument('--interp_steps', type=int, default=10)
    ap.add_argument('--grids_from', type=str, default='fixed')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dirs('result', 'result/visualizations')

    # File paths
    v_log_path = 'result/training_log_vanilla'
    f_log_path = 'result/training_log_fixed'
    v_ckpt = 'result/best_generator_vanilla.pth'
    f_ckpt = 'result/best_generator_fixed.pth'

    # Load logs and draw combined curve
    vlog = load_log(v_log_path)
    flog = load_log(f_log_path)
    plot_mode_collapse(vlog, flog, out_png='result/mode_collapse_analysis.png')
    print("Saved result/mode_collapse_analysis.png")

    # Build DB for histogram labeling
    db_feats, db_labels = build_val_db(args.data_root, device)

    # Mode coverage histogram
    grouped_mode_histogram(v_ckpt, f_ckpt, out_png='result/visualizations/mode_histogram.png',
                           z_dim=args.z_dim, samples=args.hist_samples, device=device,
                           db_feats=db_feats, db_labels=db_labels)
    print("Saved result/visualizations/mode_histogram.png")

    # Vanilla vs Fixed comparison grid
    compare_grid(v_ckpt, f_ckpt, out_png='result/visualizations/compare_vanilla_fixed.png',
                 z_dim=args.z_dim, device=device, n=100)
    print("Saved result/visualizations/compare_vanilla_fixed.png")

    # Four epoch grids + Interpolation
    src_ckpt = v_ckpt if args.grids_from == 'vanilla' else f_ckpt
    four_epoch_grids_from_ckpt(src_ckpt, 'result/visualizations', args.z_dim, device)
    print("Saved 4 epoch grids into result/visualizations/")

    save_interpolation(src_ckpt, 'result/visualizations/interpolation.png',
                       args.z_dim, device, steps=args.interp_steps)
    print("Saved result/visualizations/interpolation.png")

if __name__ == "__main__":
    main()
