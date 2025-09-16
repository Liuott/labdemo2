# part4_vae/train_vae.py
import os, argparse, torch, torchvision.utils as vutils
import torch.utils.data as td
from dataset_oasis import Oasis2DSlices
from model_vae import VAE, vae_loss
from torch import amp

def get_loader(bs, split):
    ds = Oasis2DSlices(split=split)
    pin = torch.cuda.is_available()
    return td.DataLoader(ds, batch_size=bs, shuffle=(split=="train"),
                         num_workers=4, pin_memory=pin)

def save_recon(x, xhat, path, n=8):
    # 把前 n 张输入/重建交替拼在一起，方便肉眼比较
    b = min(n, x.size(0))
    grid = torch.cat([x[:b], xhat[:b]], dim=0)
    vutils.save_image(grid, path, nrow=b, normalize=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--z", type=int, default=16)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, te_ld = get_loader(args.bs, "train"), get_loader(args.bs, "test")
    model = VAE(z_dim=args.z).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = (dev=="cuda" and args.amp)
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs("artifacts", exist_ok=True)
    for ep in range(1, args.epochs+1):
        model.train()
        for x,_ in tr_ld:
            x = x.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=use_amp):
                xhat, mu, lv = model(x)
                loss, rec, kl = vae_loss(x, xhat, mu, lv)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

        # 简易验证 + 存图
        model.eval()
        with torch.no_grad():
            x,_ = next(iter(te_ld))
            x = x.to(dev)
            xhat, mu, lv = model(x)
            val, rec, kl = vae_loss(x, xhat, mu, lv)
            save_recon(x.cpu(), xhat.cpu(), f"artifacts/recon_ep{ep:02d}.png", n=8)
        print(f"Epoch {ep:02d} | val={val:.3f} rec={rec:.3f} kl={kl:.3f}")

    torch.save(model.state_dict(), "artifacts/vae_oasis.pth")
    print("Saved to artifacts/vae_oasis.pth")

if __name__ == "__main__":
    main()
