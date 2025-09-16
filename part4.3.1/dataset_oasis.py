import os, glob, numpy as np, torch
from torch.utils.data import Dataset
import cv2

try:
    import nibabel as nib  # 如果是 .nii.gz 需要
except Exception:
    nib = None

def _load_volume(path):
    if path.endswith(".npy"):
        vol = np.load(path)  # [D,H,W] or [H,W]
    else:  # .nii or .nii.gz
        assert nib is not None, "Please pip install nibabel"
        vol = nib.load(path).get_fdata()  # float64
    vol = np.asarray(vol, dtype=np.float32)
    # 归一化到 [0,1]
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    return vol  # [D,H,W] or [H,W]

def _center_crop(img, size=128):
    h, w = img.shape
    s = min(h, w)
    y0 = (h - s)//2; x0 = (w - s)//2
    img = img[y0:y0+s, x0:x0+s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

class Oasis2DSlices(Dataset):
    def __init__(self, root="/home/groups/comp3710/oasis_preproc", split="train", take_every=4):
        # 根目录你可以用 ls 看真实子目录名，下面的 glob 兼容多种放法
        cand = []
        for ext in ("*.npy","*.nii","*.nii.gz"):
            cand += glob.glob(os.path.join(root, "**", ext), recursive=True)
        assert len(cand)>0, f"No volumes found under {root}"
        # 简单划分：前 80% 训练，后 20% 测试
        cand.sort()
        n = int(len(cand)*0.8)
        self.paths = cand[:n] if split=="train" else cand[n:]
        self.take_every = take_every

        # 预索引所有切片路径（轻量）
        self.index = []
        for p in self.paths:
            vol = _load_volume(p)
            if vol.ndim==2:  # 单片
                self.index.append((p, None))
            else:
                for z in range(0, vol.shape[0], self.take_every):
                    self.index.append((p, z))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        p, z = self.index[i]
        vol = _load_volume(p)
        img = vol if z is None else vol[z]
        img = _center_crop(img, 128).astype(np.float32)  # [128,128]
        x = torch.from_numpy(img[None, ...])             # [1,128,128]
        return x, x  # 自编码器：输入=目标
