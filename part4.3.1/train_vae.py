# part4_vae/train_vae.py
import os, argparse, torch, torchvision.utils as vutils
import torch.utils.data as td
from dataset_oasis import Oasis2DSlices
from model_vae import VAE, vae_loss
from torch import amp  # PyTorch ≥ 2.2

def get_loader(bs, split, data_root=None):
    if data_root is None:
        data_root = os.environ.get("OASIS_ROOT", "/home/groups/comp3710/OASIS")
    nw  = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    pin = torch.cuda.is_available()
    ds  = Oasis2DSlices(root=data_root, split=split, take_every=4, size=128)
    return td.DataLoader(ds, batch_size=bs, shuffle=(split=="train"),
                         num_workers=nw, pin_memory=pin, persistent_workers=(nw>0))

def save_recon(x, xhat, path, n=8):
    b = min(n, x.size(0))
    grid = torch.cat([x[:b], xhat[:b]], dim=0)
    vutils.save_image(grid, path, nrow=b, normalize=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs",     type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--z",      type=int, default=16)
    ap.add_argument("--amp",    action="store_true")
    ap.add_argument("--data-root", type=str,
                    default=os.environ.get("OASIS_ROOT", "/home/groups/comp3710/OASIS"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using data root:", args.data_root)
    tr_ld = get_loader(args.bs, "train", args.data_root)
    te_ld = get_loader(args.bs, "test",  args.data_root)

    model = VAE(z_dim=args.z).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (dev == "cuda" and args.amp)
    scaler  = amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs("artifacts", exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        for x,_ in tr_ld:
            x = x.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=use_amp):
                xhat, mu, lv = model(x)
            with amp.autocast("cuda", enabled=False):
                loss, rec, kl = vae_loss(x, xhat, mu, lv)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

        # 验证 + 存图
        model.eval()
        with torch.no_grad():
            x,_ = next(iter(te_ld))
            x = x.to(dev)
            with amp.autocast("cuda", enabled=use_amp):
                xhat, mu, lv = model(x)
            with amp.autocast("cuda", enabled=False):
                val, rec, kl = vae_loss(x, xhat, mu, lv)
            save_recon(x.cpu(), xhat.cpu(), f"artifacts/recon_ep{ep:02d}.png", n=8)
        print(f"Epoch {ep:02d} | val={val.item():.3f} rec={rec.item():.3f} kl={kl.item():.3f}")

    torch.save(model.state_dict(), "artifacts/vae_oasis.pth")
    print("Saved to artifacts/vae_oasis.pth")

if __name__ == "__main__":
    main()
