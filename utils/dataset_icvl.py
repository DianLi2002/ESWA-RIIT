import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
np.random.seed(42)
    
class ICVLCalibrationDataset(Dataset):
    def __init__(self, opt, input_dir: str, target_dir: str, file_list: List[str], crop=False):
        self.opt = opt
        self.crop = crop
        self.file_list = file_list

        # ==============================================
        # 🔥 核心优化：一次性预加载所有数据到内存
        # 训练时不再读取硬盘，CPU 零 IO 开销
        # ==============================================
        self.data_cache = []
        print(f"[ICVLCalibrationDataset] 预加载 {len(file_list)} 对数据到内存...")

        for fname in file_list:
            # 读取原始数据（仅这一次访问硬盘）
            inp = np.load(os.path.join(input_dir, fname))
            tgt = np.load(os.path.join(target_dir, fname))

            # 归一化
            inp = inp / 4096.0
            tgt = tgt / 4096.0

            # 转换为 Tensor 并转置为 (C, H, W)
            inp_t = torch.from_numpy(inp.astype(np.float32)).permute(2, 0, 1).contiguous()
            tgt_t = torch.from_numpy(tgt.astype(np.float32)).permute(2, 0, 1).contiguous()

            self.data_cache.append((inp_t, tgt_t, fname))

        print(f"[ICVLCalibrationDataset] 预加载完成，共 {len(self.data_cache)} 对。训练时零磁盘读取。")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        inp_t, tgt_t, fname = self.data_cache[idx]   # (C, H, W)
        C, H, W = inp_t.shape

        if self.crop:   # 训练模式：随机裁剪
            rand_h = random.randint(0, H - self.opt.crop_size)
            rand_w = random.randint(0, W - self.opt.crop_size)
            inp_crop = inp_t[:, rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
            tgt_crop = tgt_t[:, rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
            return inp_crop, tgt_crop, fname
        else:           # 验证/测试：返回全图
            return inp_t, tgt_t, fname


def build_datasets(
    opt = None,
    root_dir = None,
    num_val = 100,
    num_test = 100,
    seed = 42,
    flag = 'train'
) -> Tuple[Dataset, Dataset, Dataset]:

    input_dir = os.path.join(root_dir, "input")
    target_dir = os.path.join(root_dir, "target")

    assert os.path.isdir(input_dir)
    assert os.path.isdir(target_dir)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".npy")
    ])

    assert len(files) > num_val + num_test, "数据量不足"

    # 固定随机种子，保证可复现
    random.seed(seed)
    random.shuffle(files)

    test_files = files[:num_test]
    val_files = files[num_test:num_test + num_val]
    train_files = files[num_test + num_val:]

    if flag == 'train':
        train_dataset = ICVLCalibrationDataset(opt, input_dir, target_dir, train_files, crop = True)
        return train_dataset
    elif flag == "val":
        val_dataset = ICVLCalibrationDataset(opt, input_dir, target_dir, val_files, crop = False)
        return val_dataset
    else:    
        test_dataset = ICVLCalibrationDataset(opt, input_dir, target_dir, test_files, crop = False)

        return test_dataset