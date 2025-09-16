# part4_vae/infer_vae.py
import os, argparse, torch, numpy as np
import torchvision.utils as vutils
from dataset_oasis import Oasis2DSlices
from model_vae import VAE
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=16)
    ap.add_argument("--ckpt", type=str, default="artifacts/vae_oasis.pth")
    ap.add_argument("--points", type=int, default=2000)
    ap.add_argument("--data-root", type=str,
                    default=os.environ.get("OASIS_ROOT", "/home/groups/comp3710/OASIS"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 传入 root，workers 跟随 Slurm，pin_memory 仅在 cuda 开
    nw = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")), 2)
    pin = torch.cuda.is_available()
    ds = Oasis2DSlices(root=args.data_root, split="test", take_every=4, size=128)
    ld = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True,
                                     num_workers=nw, pin_memory=pin, persistent_workers=False)

    model = VAE(z_dim=args.z).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    # 1) 重建
    x,_ = next(iter(ld))
    x = x.to(dev)
    with torch.no_grad():
        xhat, mu, lv = model(x)
    os.makedirs("artifacts", exist_ok=True)
    vutils.save_image(torch.cat([x[:8].cpu(), xhat[:8].cpu()],0),
                      "artifacts/recon_eval.png", nrow=8, normalize=True)

    # 2) 潜空间可视化
    zs = []
    with torch.no_grad():
        cnt = 0
        for xb,_ in ld:
            xb = xb.to(dev)
            try:
                mu,_ = model.encode(xb)   
            except AttributeError:
                _, mu, _ = model(xb)      # 退化方案：用 forward 输出的 mu
            zs.append(mu.cpu().numpy())
            cnt += xb.size(0)
            if cnt >= args.points: break
    Z = np.concatenate(zs,0)

    try:
        import umap
        U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0).fit_transform(Z)
        title = "VAE latent (UMAP)"
    except Exception:
        from sklearn.decomposition import PCA
        U = PCA(2).fit_transform(Z)
        title = "VAE latent (PCA)"

    plt.figure(figsize=(5,5))
    plt.scatter(U[:,0], U[:,1], s=5, alpha=0.5)
    plt.title(title); plt.tight_layout()
    plt.savefig("artifacts/latent_2d.png", dpi=180)
    print("Saved artifacts/recon_eval.png and artifacts/latent_2d.png")

if __name__ == "__main__":
    main()
