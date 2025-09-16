
import os, glob, numpy as np, torch
import torch.nn.functional as F


try:
    from PIL import Image   # 读 PNG/JPG 用
except Exception:
    Image = None

def _load_volume_any(path):
    """Return ndarray float32 in [0,1], shape [H,W] or [D,H,W]."""
    pl = path.lower()
    if pl.endswith(".npy"):
        arr = np.load(path)
    elif pl.endswith(".npz"):
        z = np.load(path)
        # 兼容多种 key
        if "arr_0" in z.files:
            arr = z["arr_0"]
        else:
            # 拿第一个数组型条目
            for k in z.files:
                if isinstance(z[k], np.ndarray):
                    arr = z[k]; break
            else:
                raise ValueError(f"npz has no ndarray: {path}")

    elif pl.endswith(".png") or pl.endswith(".jpg") or pl.endswith(".jpeg") or pl.endswith(".tif") or pl.endswith(".tiff"):
        assert Image is not None, "Please `pip install pillow` to read images"
        img = Image.open(path).convert("L")  # 灰度
        arr = np.array(img, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    # 归一化到 [0,1]（鲁棒一些：1%~99% 百分位）
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0, 1)
    return arr

def _resize_tensor(img, size=128):
    # img: [1,H,W] float32
    img = F.interpolate(img.unsqueeze(0), size=(size,size), mode="bilinear", align_corners=False)
    return img.squeeze(0)

class Oasis2DSlices(torch.utils.data.Dataset):
    """
    从 root 递归收集下列后缀：
      .npy .npz .nii .nii.gz .hdr/.img .mgz .mgh .png .jpg .jpeg .tif .tiff
    3D 体：每 take_every 张取一张轴向切片；统一中心裁剪为正方形并 resize 到 128。
    """
    def __init__(self, root, split="train", take_every=4, size=128):
        assert os.path.isdir(root), f"root not a directory: {root}"
        self.size = size
        # 收集所有支持的文件
        pats = ["*.npy","*.npz","*.nii","*.nii.gz","*.hdr","*.img","*.mgz","*.mgh","*.png","*.jpg","*.jpeg","*.tif","*.tiff"]
        paths = []
        for pat in pats:
            paths += glob.glob(os.path.join(root, "**", pat), recursive=True)

        # 过滤 Analyze 成对文件：只保留 .hdr（避免 .img/.hdr 重复）
        filt = []
        seen_hdr = set()
        for p in paths:
            pl = p.lower()
            if pl.endswith(".img"):
                # 如果有同名 .hdr，就跳过 .img
                hdr = p[:-4] + ".hdr"
                if os.path.exists(hdr):
                    continue
            if pl.endswith(".hdr"):
                if p in seen_hdr: 
                    continue
                seen_hdr.add(p)
            filt.append(p)
        paths = sorted(set(filt))

        print(f"[Oasis2DSlices] root={root} total files matched={len(paths)}")
        if len(paths) == 0:
            raise AssertionError(f"No volumes/images found under {root}. Try different root or add format support.")

        # 80/20 划分
        n_train = int(0.8 * len(paths))
        self.vol_paths = paths[:n_train] if split=="train" else paths[n_train:]
        self.take_every = take_every

        # 建立 (path, z) 索引
        self.index = []
        for p in self.vol_paths:
            try:
                vol = _load_volume_any(p)
            except Exception:

                continue
            if vol.ndim == 2:
                self.index.append((p, None))
            elif vol.ndim == 3:
                D = vol.shape[0]
                for z in range(0, D, self.take_every):
                    self.index.append((p, z))
        if len(self.index) == 0:
            raise AssertionError(f"No usable 2D slices indexed from files under {root}.")

        print(f"[Oasis2DSlices] split={split} indexed slices={len(self.index)}")

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        p, z = self.index[i]
        vol = _load_volume_any(p)
        img = vol if z is None else vol[z]
        if img.ndim != 2:
            # 万一读到多通道（H,W,C），取灰度
            img = np.mean(img, axis=-1).astype(np.float32)
        img = torch.from_numpy(img)[None, ...]  # [1,H,W]
        # 中心裁剪成正方形并 resize
        H, W = img.shape[-2:]
        s = min(H, W)
        top, left = (H - s)//2, (W - s)//2
        img = img[..., top:top+s, left:left+s]
        img = _resize_tensor(img, self.size).clamp(0,1).float()
        return img, img

