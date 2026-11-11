import json
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 配置 =================
SCENE_DIR = Path('/opt/data/private/ld/data/BJTU/rw')
GT_DIR    = Path('/opt/data/private/ld/data/BJTU/gt')
OUT_DIR   = Path('/opt/data/private/ld/data/BJTU/merged')
OUT_DIR.mkdir(exist_ok=True)
JSON_PATH = '/opt/data/private/ld/Automatic-spectral-calibration-of-HSI/real_light_split.json'
NUM_WORKERS = min(cpu_count(), 8)  # 根据 CPU 和磁盘性能调整，SSD 可设 8-12

# ============== 全局变量（供 worker 使用） ==============
# 将不变化的数据放在全局，避免每次重复传递
SCENE_DIR_G = SCENE_DIR
GT_DIR_G = GT_DIR

def init_worker(scene_dir, gt_dir):
    """初始化 worker 进程的全局路径"""
    global SCENE_DIR_G, GT_DIR_G
    SCENE_DIR_G = scene_dir
    GT_DIR_G = gt_dir

def process_file(args):
    """处理一个文件，返回 (idx, scene_array, gt_array) 或直接写入临时变量？改为返回数组"""
    idx, fname = args
    # 1. 获取 .npy 路径
    if fname.endswith('.mat'):
        fname_npy = fname[:-4] + '.npy'
    elif fname.endswith('.npy'):
        fname_npy = fname
    else:
        fname_npy = fname + '.npy'

    # 2. 提取数字 ID
    base_name = fname_npy[:-4]
    match = re.search(r'\d+', base_name)
    if not match:
        raise ValueError(f"无法提取数字ID: {fname}")
    img_id = match.group()
    gt_name = 'gtRef_' + img_id + '.npy'

    # 3. 加载场景数据 (H,W,C) -> 归一化 -> 转置 (C,H,W)
    scene_path = SCENE_DIR_G / fname_npy
    scene = np.load(scene_path)
    scene = (scene / 4096.0).astype(np.float32)
    if scene.ndim == 3:
        scene = np.transpose(scene, (2, 0, 1))

    # 4. 加载 GT 数据
    gt_path = GT_DIR_G / gt_name
    gt = np.load(gt_path).astype(np.float32)
    if gt.ndim == 3:
        gt = np.transpose(gt, (2, 0, 1))

    return idx, scene, gt

def merge_split(split):
    """对单个 split 执行多进程合并"""
    with open(JSON_PATH) as f:
        splits = json.load(f)
    file_list = splits[split]
    N = len(file_list)
    if N == 0:
        print(f"  {split} 无文件，跳过")
        return

    # 获取单张图像的 shape（假设所有图像尺寸相同）
    sample_fname = file_list[0]
    if sample_fname.endswith('.mat'):
        sample_npy = sample_fname[:-4] + '.npy'
    else:
        sample_npy = sample_fname + '.npy' if not sample_fname.endswith('.npy') else sample_fname
    sample_path = SCENE_DIR / sample_npy
    sample_img = np.load(sample_path)
    if sample_img.ndim == 2:
        H, W = sample_img.shape
        C = 1
    else:
        H, W, C = sample_img.shape
    # 确保 C 是通道数（最后一个维度）
    # 如果原始数据已经是 (C,H,W)，需要判断（此处假设为 (H,W,C)）
    # 可通过原代码确认，这里保守假设 (H,W,C)
    print(f"  图像尺寸: (H={H}, W={W}, C={C})")

    # 预创建 memmap 文件
    # 注意：train 缓存使用 CHW (N, C, H, W)，val/test 缓存使用 HWC (N, H, W, C)，
    # 以与 utils/dataset_bjtu.py / dataset_bjtu_e.py 中的读取约定保持一致。
    scene_path = OUT_DIR / f'scene_{split}.npy'
    gt_path = OUT_DIR / f'gt_{split}.npy'
    shape = (N, C, H, W) if split == 'train' else (N, H, W, C)

    # 先创建并关闭，确保文件存在
    sm = np.memmap(scene_path, dtype='float32', mode='w+', shape=shape)
    gm = np.memmap(gt_path, dtype='float32', mode='w+', shape=shape)
    del sm, gm

    # 准备任务
    tasks = [(i, fname) for i, fname in enumerate(file_list)]

    # 多进程处理并写入
    with Pool(NUM_WORKERS, initializer=init_worker, initargs=(SCENE_DIR, GT_DIR)) as pool:
        # 使用 imap_unordered 获取结果，按序写入 memmap
        with tqdm(total=N, desc=f"Merging {split}") as pbar:
            for result in pool.imap_unordered(process_file, tasks, chunksize=4):
                idx, scene_arr, gt_arr = result
                # 写入 memmap（需要重新打开，因为 memmap 对象不能在进程间共享）
                # 这里直接在主进程中写入，打开一次 memmap
                # 注意：为避免频繁打开，可以在循环外打开，但写入必须在主进程
                # 我们开启两个 memmap，持续写入
                # 由于是主进程顺序接收结果并写入，安全
                # 打开 memmap（如果尚未打开）
                if 'scene_mmap' not in locals():
                    import gc
                    # 首次打开
                    scene_mmap = np.memmap(scene_path, dtype='float32', mode='r+', shape=shape)
                    gt_mmap = np.memmap(gt_path, dtype='float32', mode='r+', shape=shape)
                scene_mmap[idx] = scene_arr if split == 'train' else np.transpose(scene_arr, (1, 2, 0))
                gt_mmap[idx] = gt_arr if split == 'train' else np.transpose(gt_arr, (1, 2, 0))
                pbar.update(1)

    # 确保刷新并关闭
    try:
        scene_mmap.flush()
        gt_mmap.flush()
    except:
        pass
    del scene_mmap, gt_mmap

    # 保存名称列表
    names_path = OUT_DIR / f'names_{split}.txt'
    with open(names_path, 'w') as f:
        for name in file_list:
            f.write(name + '\n')
    print(f"  {split} 完成，文件保存至 {OUT_DIR}")


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        merge_split(split)
    print("所有合并完成！")