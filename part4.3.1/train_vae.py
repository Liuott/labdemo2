import torch, torch.utils.data as td, argparse, os
from dataset_oasis import Oasis2DSlices
from model_vae import VAE, vae_loss

def get_loader(bs, split):
    ds = Oasis2DSlices(split=split)
    return td.DataLoader(ds, batch_size=bs, shuffle=(split=="train"),
                         num_workers=4, pin_memory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--z", type=int, default=16)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_ld, te_ld = get_loader(args.bs, "train"), get_loader(args.bs, "test")
    model = VAE(z_dim=args.z).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for ep in range(1, args.epochs+1):
        model.train()
        for x,_ in tr_ld:
            x = x.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                xhat, mu, lv = model(x)
                loss, rec, kl = vae_loss(x, xhat, mu, lv)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        # 简易验证
        model.eval()
        with torch.no_grad():
            x,_ = next(iter(te_ld))
            x = x.to(dev)
            xhat, mu, lv = model(x)
            val, rec, kl = vae_loss(x, xhat, mu, lv)
        print(f"Epoch {ep:02d} | val={val:.3f} rec={rec:.3f} kl={kl:.3f}")
    os.makedirs("ckpt", exist_ok=True)
    torch.save(model.state_dict(), "ckpt/vae_oasis.pth")

if __name__ == "__main__":
    main()
