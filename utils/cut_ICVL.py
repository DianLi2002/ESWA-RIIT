import os
import h5py
import numpy as np

# ========================
# 路径设置
# ========================
mat_dir = "/opt/data/private/yonghu/data_ld/ICVL"          # 存放 .mat 文件的文件夹
out_dir = "/opt/data/private/yonghu/data_ld/ICVL_sub"      # 输出 .npy 文件的文件夹
os.makedirs(out_dir, exist_ok=True)

# ========================
# 子图大小
# ========================
crop_size = 512

# ========================
# 遍历所有 .mat 文件
# ========================
for fname in os.listdir(mat_dir):
    if not fname.endswith(".mat"):
        continue

    mat_path = os.path.join(mat_dir, fname)
    base_name = os.path.splitext(fname)[0]

    print(f"Processing {fname} ...")

    # ========================
    # 读取 Matlab v7.3 文件
    # ========================
    with h5py.File(mat_path, "r") as f:
        # rad 在 h5py 中通常是倒序存储的，需要转置
        rad = f["rad"][:]           # 原始 shape 可能是 (31, 1392, 1300)
        rad = np.transpose(rad, (2, 1, 0))  # -> (1300, 1392, 31)

    H, W, C = rad.shape
    assert C == 31, "光谱通道数不是 31"

    # ========================
    # 四个顶角裁剪
    # ========================
    crops = {
        "tl": rad[0:crop_size, 0:crop_size, :],                          # 左上
        "tr": rad[0:crop_size, W-crop_size:W, :],                        # 右上
        "bl": rad[H-crop_size:H, 0:crop_size, :],                        # 左下
        "br": rad[H-crop_size:H, W-crop_size:W, :]                       # 右下
    }

    # ========================
    # 保存为 .npy 文件
    # ========================
    for k, crop in crops.items():
        out_name = f"{base_name}_{k}.npy"
        out_path = os.path.join(out_dir, out_name)
        np.save(out_path, crop)

print("All files processed successfully.")
