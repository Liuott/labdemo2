# part4_vae/dataset_oasis.py
import os, glob, numpy as np, torch
import torch.nn.functional as F
try:
    import nibabel as nib  # 读 NIfTI 用
except Exception:
    nib = None

def _load_volume(path):
    if path.endswith(".npy"):
        vol = np.load(path)
    else:
        assert nib is not None, "This dataset contains NIfTI; please `pip install nibabel`."
        vol = nib.load(path).get_fdata()
    vol = np.asarray(vol, dtype=np.float32)
    # 简单归一化到 [0,1]（去掉极值更稳）
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
    return vol

def _resize_tensor(img, size=128):
    # img: [1,H,W] torch.float32
    img = F.interpolate(img.unsqueeze(0), size=(size,size), mode="bilinear", align_corners=False)
    return img.squeeze(0)

class Oasis2DSlices(torch.utils.data.Dataset):
    """
    从 root 递归收集 .npy / .nii / .nii.gz，按 80/20 划分 train/test。
    3D 体数据按 take_every 采样轴向切片，统一 resize 到 128x128。
    """
    def __init__(self, root="/home/groups/comp3710/oasis_preproc", split="train",
                 take_every=4, size=128):
        self.size = size
        paths = []
        for ext in ("*.npy","*.nii","*.nii.gz"):
            paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
        assert paths, f"No volumes found under {root}. Check the path."
        paths.sort()
        n_train = int(0.8 * len(paths))
        self.vol_paths = paths[:n_train] if split=="train" else paths[n_train:]
        self.take_every = take_every

        # 索引 (path, z)；z=None 表示 2D 文件
        self.index = []
        for p in self.vol_paths:
            vol = _load_volume(p)
            if vol.ndim == 2:
                self.index.append((p, None))
            else:
                for z in range(0, vol.shape[0], self.take_every):
                    self.index.append((p, z))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        p, z = self.index[i]
        vol = _load_volume(p)
        img = vol if z is None else vol[z]
        img = torch.from_numpy(img)[None, ...]          # [1,H,W]
        # 居中裁剪成正方形，再 resize 到 128
        H, W = img.shape[-2:]
        s = min(H, W)
        top, left = (H - s)//2, (W - s)//2
        img = img[..., top:top+s, left:left+s]
        img = _resize_tensor(img, self.size)            # [1,128,128]
        img = img.clamp(0,1).float()
        return img, img  # 自编码器：输入=目标
