import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class FontDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train"):
        assert split in {"train","val"}
        self.root = os.path.join(data_root, split)
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Split dir not found: {self.root}")
        self.paths: List[str] = sorted(
            [os.path.join(self.root, f) for f in os.listdir(self.root) if f.lower().endswith(".png")]
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No PNG images in {self.root}")

    def _path_to_label(self, p: str) -> int:
        base = os.path.basename(p)
        for ch in base:
            if "A" <= ch <= "Z":
                return ord(ch) - ord("A")
        return 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        im = Image.open(p).convert("L")           
        arr = np.array(im, dtype=np.uint8)        
        x = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
        y = self._path_to_label(p)
        return x, y, p
