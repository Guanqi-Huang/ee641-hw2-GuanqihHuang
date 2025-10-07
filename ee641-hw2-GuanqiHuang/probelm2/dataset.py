from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Any
from torch.utils.data import Dataset

def _try_npz_arrays(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Return the patterns array from common keys:
      - 'patterns' (preferred)
      - 'X'
      - first array (arr_0) as a fallback
    """
    for key in ["patterns", "X"]:
        if key in npz:
            arr = np.array(npz[key])
            break
    else:
        # fallback to first item
        first_key = list(npz.files)[0]
        arr = np.array(npz[first_key])
    if arr.ndim != 3 or arr.shape[1:] not in [(16, 9), (9, 16)]:
        raise ValueError(f"Unexpected patterns shape {arr.shape}; expected [N,16,9] or [N,9,16].")
    # If stored as [N,9,16], transpose to [N,16,9]
    if arr.shape[1:] == (9, 16):
        arr = np.transpose(arr, (0, 2, 1))
    arr = (arr > 0.5).astype(np.float32)
    return arr

def _parse_styles_json(meta: Dict[str, Any], N: int) -> Tuple[List[str], List[int]]:
    styles = None
    labels = None
    if isinstance(meta, dict):
        # nested meta
        if "meta" in meta and isinstance(meta["meta"], dict) and "styles" in meta["meta"]:
            styles = meta["meta"]["styles"]

        if "styles" in meta and isinstance(meta["styles"], list):
            styles = meta["styles"]

        if "labels" in meta:
            labels = meta["labels"]

            if isinstance(labels, list):
                if len(labels) != N:
                    raise ValueError(f"labels length {len(labels)} != N {N}")
                if all(isinstance(v, int) for v in labels):
                    if styles is None:
                        # try  "classes" or "style_names"
                        styles = meta.get("classes") or meta.get("style_names")
                        if styles is None:
                            # default 5 classes if not given
                            styles = ["rock", "jazz", "hiphop", "electronic", "latin"]
                    return list(styles), list(labels)
                else:
                    # labels are names -> map to indices
                    if styles is None:
                        # preserve first-occurrence order
                        seen = []
                        for name in labels:
                            if name not in seen:
                                seen.append(name)
                        styles = seen
                    name2idx = {n: i for i, n in enumerate(styles)}
                    y = [name2idx[str(n)] for n in labels]
                    return list(styles), y

        # items with style names
        if "items" in meta and isinstance(meta["items"], list):
            items = meta["items"]
            if len(items) != N:
                raise ValueError(f"items length {len(items)} != N {N}")
            names = [it.get("style") for it in items]
            if styles is None:
                seen = []
                for n in names:
                    if n not in seen:
                        seen.append(n)
                styles = seen
            name2idx = {n: i for i, n in enumerate(styles)}
            y = [name2idx[str(n)] for n in names]
            return list(styles), y

    # meta is a list of dicts with 'style'
    if isinstance(meta, list) and len(meta) == N and all(isinstance(d, dict) and "style" in d for d in meta):
        names = [d["style"] for d in meta]
        seen = []
        for n in names:
            if n not in seen:
                seen.append(n)
        name2idx = {n: i for i, n in enumerate(seen)}
        y = [name2idx[str(n)] for n in names]
        return seen, y

    default_styles = ["rock", "jazz", "hiphop", "electronic", "latin"]
    return default_styles, [0] * N

class DrumPatternDataset(Dataset):

    def __init__(self, root: str | Path = "../data/drums", split: str = "train", val_ratio: float = 0.1, seed: int = 641):
        root = Path(root)
        if not root.exists():
            alt = Path(__file__).resolve().parent.parent / "data" / "drums"
            if alt.exists():
                root = alt
        self.root = root

        pat_split = root / f"patterns_{split}.pt"
        sty_split = root / f"styles_{split}.pt"
        if pat_split.exists() and sty_split.exists():
            X = torch.load(pat_split).float()  
            y = torch.load(sty_split).long()   
            self.patterns = X
            self.labels = y
            self.style_names = ["rock", "jazz", "hiphop", "electronic", "latin"]
            self.N = X.shape[0]
            return

        # Legacy single .pt pair
        pat_pt = root / "patterns.pt"
        sty_pt = root / "styles.pt"
        if pat_pt.exists() and sty_pt.exists():
            self.patterns = torch.load(pat_pt).float()
            self.labels = torch.load(sty_pt).long()
            self.N = self.patterns.shape[0]
            self.style_names = ["rock", "jazz", "hiphop", "electronic", "latin"]
            return
        npz_path = root / "patterns.npz"
        json_path = root / "patterns.json"
        if not npz_path.exists() or not json_path.exists():
            raise FileNotFoundError(
                f"Could not find required files at {root}. Expected patterns.npz and patterns.json."
            )

        npz = np.load(npz_path, allow_pickle=True)
        X_np = _try_npz_arrays(npz)             
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # styles parsing
        style_names, y_idx = _parse_styles_json(meta, N=X_np.shape[0])

        self.patterns = torch.from_numpy(X_np.astype(np.float32))   
        self.labels = torch.tensor(y_idx, dtype=torch.long)         
        self.N = self.patterns.shape[0]
        self.style_names = list(style_names)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x = self.patterns[idx]                      
        s_idx = int(self.labels[idx].item())
        S = torch.zeros(len(self.style_names), dtype=torch.float32)
        if 0 <= s_idx < len(self.style_names):
            S[s_idx] = 1.0
        return x, S, s_idx
