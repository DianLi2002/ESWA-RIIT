from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class BJTU_CacheDataset(Dataset):
    """
    Dataset for cached BJTU data.

    Expected cache format:

    train:
        scene_train.npy: (N, C, 256, 256)
        gt_train.npy:    (N, C, 256, 256)

    val/test:
        scene_val.npy:   (N, H, W, C)
        gt_val.npy:      (N, H, W, C)
        scene_test.npy:  (N, H, W, C)
        gt_test.npy:     (N, H, W, C)

    names:
        names_train.txt
        names_val.txt
        names_test.txt
    """

    def __init__(self, opt, split: str):
        super().__init__()

        assert split in ["train", "val", "test"], \
            f"split must be train/val/test, got {split}"

        self.opt = opt
        self.split = split
        self.root = Path(opt.merged_root)

        self.scene_path = self.root / f"scene_{split}.npy"
        self.gt_path = self.root / f"gt_{split}.npy"
        self.names_path = self.root / f"names_{split}.txt"

        if not self.scene_path.exists():
            raise FileNotFoundError(f"Missing scene cache: {self.scene_path}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Missing gt cache: {self.gt_path}")
        if not self.names_path.exists():
            raise FileNotFoundError(f"Missing names file: {self.names_path}")

        self.scene_data = np.load(self.scene_path, mmap_mode="r")
        self.gt_data = np.load(self.gt_path, mmap_mode="r")

        with open(self.names_path, "r") as f:
            self.names = [line.strip() for line in f.readlines()]

        if len(self.scene_data) != len(self.gt_data):
            raise ValueError(
                f"scene and gt length mismatch: "
                f"{len(self.scene_data)} vs {len(self.gt_data)}"
            )

        if len(self.names) != len(self.scene_data):
            raise ValueError(
                f"names and data length mismatch: "
                f"{len(self.names)} vs {len(self.scene_data)}"
            )

        self.expected_channels = getattr(opt, "in_channels", None)

        self._check_shape()

        print(f"[BJTU_CacheDataset] split={split}")
        print(f"  scene: {self.scene_data.shape}, dtype={self.scene_data.dtype}")
        print(f"  gt:    {self.gt_data.shape}, dtype={self.gt_data.dtype}")
        print(f"  names: {len(self.names)}")
        print(f"  root:  {self.root}")

    def _check_shape(self):
        if self.scene_data.ndim != 4:
            raise ValueError(
                f"Expected 4D scene cache, got shape={self.scene_data.shape}"
            )

        if self.gt_data.ndim != 4:
            raise ValueError(
                f"Expected 4D gt cache, got shape={self.gt_data.shape}"
            )

        if self.scene_data.shape != self.gt_data.shape:
            raise ValueError(
                f"scene and gt shape mismatch: "
                f"{self.scene_data.shape} vs {self.gt_data.shape}"
            )

        if self.expected_channels is None:
            return

        shape = self.scene_data.shape

        if self.split == "train":
            # train cache should be NCHW
            channels = shape[1]
        else:
            # val/test cache should be NHWC
            channels = shape[-1]

        if channels != self.expected_channels:
            raise ValueError(
                f"Channel mismatch for split={self.split}: "
                f"cache channels={channels}, opt.in_channels={self.expected_channels}. "
                f"Please check whether you are using BJTU 31-band or 204-band cache."
            )

    def __len__(self):
        return len(self.names)

    def _to_chw_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """
        Convert cached array to CHW torch tensor.

        train cache: already CHW
        val/test cache: HWC -> CHW
        """
        if self.split == "train":
            # arr: C,H,W
            arr = np.ascontiguousarray(arr)
        else:
            # arr: H,W,C -> C,H,W
            arr = np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))

        return torch.from_numpy(arr)

    def __getitem__(self, index):
        img_A = self.scene_data[index]
        img_B = self.gt_data[index]

        img_A = self._to_chw_tensor(img_A)
        img_B = self._to_chw_tensor(img_B)

        return img_A, img_B, self.names[index]