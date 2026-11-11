import os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# =========================
# 路径配置
# =========================
SRC_DIR = "/opt/data/private/ld/data/ICVL_sub"
OUT_DIR = "/opt/data/private/ld/data/ICVL_degraded_hard"

INPUT_DIR = os.path.join(OUT_DIR, "input")
TARGET_DIR = os.path.join(OUT_DIR, "target")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

def spatial_spectral_miscalibration(msi, gain_std=0.1, spatial_smooth=32):
    H, W, C = msi.shape

    # 空间低频扰动场
    gx = np.random.normal(0, gain_std, size=(H//spatial_smooth, W//spatial_smooth, C))
    gx = np.repeat(np.repeat(gx, spatial_smooth, axis=0), spatial_smooth, axis=1)
    gx = gx[:H, :W, :]

    gain = 1.0 + gx
    gain = np.clip(gain, 0.6, 1.4)

    return msi * gain

def spectral_blur(msi, kernel_size=3, strength=0.15):
    C = msi.shape[-1]

    kernel = np.ones(kernel_size)
    kernel = kernel / kernel.sum()

    msi_blur = np.zeros_like(msi)
    for c in range(C):
        for k in range(kernel_size):
            idx = np.clip(c + k - kernel_size//2, 0, C-1)
            msi_blur[..., c] += kernel[k] * msi[..., idx]

    return (1 - strength) * msi + strength * msi_blur

def coupled_spatial_spectral_illumination(msi, strength=0.3):
    H, W, C = msi.shape
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)

    spatial = 1 + strength * (xx + yy)
    spectral = np.linspace(0.8, 1.2, C)

    illum = spatial[..., None] * spectral[None, None, :]
    illum = np.clip(illum, 0.3, 1.5)

    return msi * illum

def low_light_noise(msi, shot_noise_ratio=0.02, read_noise_std=0.005):
    shot_noise = np.random.normal(
        0,
        shot_noise_ratio * np.sqrt(np.maximum(msi, 0)),
        msi.shape
    )
    read_noise = np.random.normal(
        0,
        read_noise_std * np.mean(msi),
        msi.shape
    )
    return msi + shot_noise + read_noise

def sensor_clipping_and_quant(msi, bit_depth=12):
    max_val = 2 ** bit_depth - 1
    msi_q = np.clip(msi, 0, max_val)
    msi_q = np.round(msi_q)
    return msi_q

def hard_degradation(msi):
    msi = msi.astype(np.float32)

    msi *= np.random.uniform(0.1, 0.6)                 # 强低光
    msi = coupled_spatial_spectral_illumination(msi)
    msi = spatial_spectral_miscalibration(msi)
    msi = spectral_blur(msi)
    msi = low_light_noise(msi, 0.04, 0.01)
    msi = sensor_clipping_and_quant(msi)

    return msi

# =========================
# 单文件处理函数（供多进程调用）
# =========================
def process_one_file(fname):
    try:
        src_path = os.path.join(SRC_DIR, fname)

        msi = np.load(src_path)              # GT
        msi_deg = hard_degradation(msi)

        np.save(os.path.join(INPUT_DIR, fname), msi_deg)
        np.save(os.path.join(TARGET_DIR, fname), msi)

        return True
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return False


# =========================
# 主函数（并行）
# =========================
def main():
    files = [f for f in os.listdir(SRC_DIR) if f.endswith(".npy")]
    print(f"Found {len(files)} patches.")

    num_workers = 10
    print(f"Using {num_workers} processes.")

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(process_one_file, files),
            total=len(files)
        ))

    print("Parallel degradation finished.")


if __name__ == "__main__":
    main()