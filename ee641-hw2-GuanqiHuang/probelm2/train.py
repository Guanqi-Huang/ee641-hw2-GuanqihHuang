from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalVAE
from training_utils import (
    kl_annealing_schedule, temperature_annealing_schedule, save_json
)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results")
    (out_dir / "generated_patterns").mkdir(parents=True, exist_ok=True)
    (out_dir / "latent_analysis").mkdir(parents=True, exist_ok=True)

    # Config
    batch_size = 64
    num_epochs = 100
    lr = 1e-3
    z_high_dim, z_low_dim = 8, 16

    # Data
    train_ds = DrumPatternDataset("../data/drums", split="train")
    val_ds   = DrumPatternDataset("../data/drums", split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model and Optimization
    model = HierarchicalVAE(z_high_dim=z_high_dim, z_low_dim=z_low_dim, hidden=256).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    log: Dict[str, Any] = {"train": [], "val": []}
    best_val = float("inf")
    best_ckpt = out_dir / "best_model.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        beta = kl_annealing_schedule(epoch, method="cyclical", total_epochs=num_epochs, cycles=4)
        temp = temperature_annealing_schedule(epoch, total_epochs=num_epochs)

        tot_loss = tot_recon = tot_kll = tot_klh = 0.0
        for x, _, _ in train_loader:
            x = x.to(device)
            logits, (mu_l, logv_l, mu_h, logv_h) = model(x)
            loss, recon, kl_l, kl_h = model.elbo(x, logits, mu_l, logv_l, mu_h, logv_h, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            tot_loss += float(loss.item())
            tot_recon += float(recon.item())
            tot_kll += float(kl_l.item())
            tot_klh += float(kl_h.item())

        n_batches = len(train_loader)
        tr_row = {
            "epoch": epoch,
            "loss": tot_loss / n_batches,
            "recon": tot_recon / n_batches,
            "kl_low": tot_kll / n_batches,
            "kl_high": tot_klh / n_batches,
            "beta": beta,
            "temp": temp,
        }
        log["train"].append(tr_row)
        print(f"[Train {epoch:03d}] loss={tr_row['loss']:.4f} recon={tr_row['recon']:.4f} "
              f"kl_low={tr_row['kl_low']:.4f} kl_high={tr_row['kl_high']:.4f} beta={beta:.3f}")

        # Validation
        model.eval()
        with torch.no_grad():
            v_tot = v_rec = v_kll = v_klh = 0.0
            for x, _, _ in val_loader:
                x = x.to(device)
                logits, (mu_l, logv_l, mu_h, logv_h) = model(x)
                v_loss, v_recon, v_kl_l, v_kl_h = model.elbo(x, logits, mu_l, logv_l, mu_h, logv_h, beta=beta)
                v_tot += float(v_loss.item())
                v_rec += float(v_recon.item())
                v_kll += float(v_kl_l.item())
                v_klh += float(v_kl_h.item())
            v_batches = len(val_loader)
            val_row = {
                "epoch": epoch,
                "loss": v_tot / v_batches,
                "recon": v_rec / v_batches,
                "kl_low": v_kll / v_batches,
                "kl_high": v_klh / v_batches,
                "beta": beta
            }
            log["val"].append(val_row)
            print(f"[ Val  {epoch:03d}] loss={val_row['loss']:.4f} recon={val_row['recon']:.4f} "
                  f"kl_low={val_row['kl_low']:.4f} kl_high={val_row['kl_high']:.4f}")

            # Save best
            if val_row["loss"] < best_val:
                best_val = val_row["loss"]
                torch.save({"epoch": epoch, "state_dict": model.state_dict()}, best_ckpt)

        # Save running log each epoch
        save_json(log, out_dir / "training_log.json")

    print(f"Best model saved to: {best_ckpt}")

if __name__ == "__main__":
    train()
