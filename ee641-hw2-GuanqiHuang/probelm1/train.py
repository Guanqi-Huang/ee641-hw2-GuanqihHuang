import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import make_letter_classifier_embed, assign_modes
from fixes import FeatureMatchingLoss, DiscriminatorWithFeatures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='../data/fonts')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--z_dim', type=int, default=100)
    ap.add_argument('--use_fix', type=str, default='none', choices=['none','feature_matching'])
    ap.add_argument('--run_name', type=str, required=True, choices=['vanilla','fixed'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_set = FontDataset(args.data_root, 'train')
    val_set   = FontDataset(args.data_root, 'val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Models
    G = Generator(args.z_dim).to(device)
    D = Discriminator().to(device)
    fm_loss = None
    if args.use_fix == 'feature_matching':
        D = DiscriminatorWithFeatures(D).to(device)
        fm_loss = FeatureMatchingLoss().to(device)

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce  = nn.BCEWithLogitsLoss()

    # Output
    out_root = 'result'
    os.makedirs(out_root, exist_ok=True)
    log_path  = os.path.join(out_root, f"training_log_{args.run_name}")              # no extension, per your request
    ckpt_path = os.path.join(out_root, f"best_generator_{args.run_name}.pth")

    # Build reference DB for mode coverage
    db_feats, db_labels = make_letter_classifier_embed(val_loader, device)

    log = {"epochs": [], "G_loss": [], "D_loss": [], "mode_coverage": []}

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        g_acc, d_acc = 0.0, 0.0

        for x, _, _ in train_loader:
            x = x.to(device) * 2 - 1
            B = x.size(0)

            # Train D
            optD.zero_grad(set_to_none=True)
            if fm_loss is not None:
                real_logits, _ = D(x)
            else:
                real_logits = D(x)
            with torch.no_grad():
                z = torch.randn(B, args.z_dim, device=device)
                fake = G(z)
            if fm_loss is not None:
                fake_logits, _ = D(fake)
            else:
                fake_logits = D(fake)

            d_real = bce(real_logits.squeeze(), torch.ones(B, device=device))
            d_fake = bce(fake_logits.squeeze(), torch.zeros(B, device=device))
            d_loss = d_real + d_fake
            d_loss.backward(); optD.step()

            # Train G
            optG.zero_grad(set_to_none=True)
            z = torch.randn(B, args.z_dim, device=device)
            fake = G(z)
            if fm_loss is not None:
                fake_logits, fake_feats = D(fake)
                g_adv = bce(fake_logits.squeeze(), torch.ones(B, device=device))
                with torch.no_grad():
                    _, real_feats = D(x)
                g_fm = fm_loss(real_feats.detach(), fake_feats)
                g_loss = g_adv + 10.0 * g_fm
            else:
                fake_logits = D(fake)
                g_loss = bce(fake_logits.squeeze(), torch.ones(B, device=device))
            g_loss.backward(); optG.step()

            g_acc += float(g_loss.item())
            d_acc += float(d_loss.item())

        # compute mode coverage on a fixed sample size
        G.eval()
        with torch.no_grad():
            z = torch.randn(260, args.z_dim, device=device)
            gen_small = G(z)
            pred = assign_modes(gen_small, db_feats, db_labels)
            cov = torch.unique(pred, sorted=True).numel()

        log["epochs"].append(epoch)
        log["G_loss"].append(round(g_acc / len(train_loader), 4))
        log["D_loss"].append(round(d_acc / len(train_loader), 4))
        log["mode_coverage"].append(int(cov))

        # Save best generator
        if cov == max(log["mode_coverage"]):
            torch.save(G.state_dict(), ckpt_path)

        # Persist log every epoch
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        print(f"[{args.run_name}] Epoch {epoch:03d} | G: {log['G_loss'][-1]:.4f} | D: {log['D_loss'][-1]:.4f} | Modes: {cov}")

if __name__ == "__main__":
    main()
