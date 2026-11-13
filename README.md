<div align="center">

# RIIT

### Reflectance–Illumination Interaction Transformer for Learning-Based Automatic Spectral Calibration of Hyperspectral Images

**Dian Li** · **Siyuan Hao** · **Cheng Fang** · **Yuanxin Ye**

*Expert Systems with Applications*, 2027

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-b31b1b.svg)](https://www.sciencedirect.com/science/article/pii/S095741742603722X)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?logo=pytorch&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/License-MIT-44cc11.svg)](LICENSE)

Official PyTorch implementation of **RIIT**, a reflectance–illumination interaction transformer for automatic spectral calibration of hyperspectral images (HSI).

</div>

---

## 📖 Overview

Hyperspectral images acquired in the wild are contaminated by **illumination**, which mixes multiplicatively with surface reflectance and corrupts the measured spectral signature. *Automatic spectral calibration* (ASC) aims to recover the intrinsic reflectance cube directly from a single uncalibrated HSI, without any white-reference panel or extra prior acquisition.

RIIT casts ASC as a **reflectance–illumination interaction** problem instead of a plain image-to-image translation:

- An **illumination stream** operates on a low-frequency observation (discrete Gaussian filtering) and produces multi-scale illumination priors at full, ½ and ¼ resolution.
- A **reflectance stream** (U-Net style transformer) restores the reflectance cube and is repeatedly *injected* with those priors through **Spectral Calibration Cross-Attention (SCCA)**.
- Each block is completed by **Spatial–Spectral Attention (SSA)** and a multi-scale feed-forward network, and the whole network is trained with a **physics-informed objective** (Charbonnier + spectral angle) that constrains spectral fidelity.

<div align="center">
<img src="images/framework.png" width="100%">
<br>
<sub><b>Figure 1.</b> Overview of RIIT. Top: the asymmetric dual-stream architecture; the illumination stream (lower branch) guides the reflectance stream at three scales via SCCA injection. Bottom: details of SCCA and SSA.</sub>
</div>

---

## ✨ Highlights

| Feature | What it gives you |
|:--|:--|
| **Interaction, not fusion** | Illumination features provide the keys/values that reflectance features attend to, so calibration happens *inside* the attention rather than by naive concatenation. |
| **Two complementary attentions** | SCCA couples a spectral-axis cross-attention (channel-wise calibration) with a linear-complexity spatial cross-attention, fused by a learnable scalar α. |
| **Illumination-aware SSA** | A 5×5 depth-wise branch captures local spatial context while spectral multi-head self-attention models global band correlations, fused by a learnable scalar β. |
| **Physics-informed loss** | Charbonnier reconstruction + spectral angle loss (λ = 1.0) directly penalise spectral distortion, which is what ASC actually cares about. |
| **Lightweight and fast** | **6.42 M** parameters and **88.26 G** FLOPs for a 31 × 256 × 256 input. Against SIT: −9.6 % inference latency, −47.2 % peak inference memory, and +58.9 % training throughput. |
| **Single-image inference** | No reference panel, no paired acquisition, no test-time optimisation — one forward pass per HSI cube. |

---

## 🧠 Method

### Components

| Module | Location | Role |
|:--|:--|:--|
| `GaussianBlurLayer` | `model/RIIT.py` | Fixed 9×9 depth-wise Gaussian kernel (σ = 3), extracts the low-frequency illumination observation. |
| `IlluminationStream` | `model/RIIT.py` | 3-stage CNN encoder (InstanceNorm + LeakyReLU) producing priors at full, ½ and ¼ resolution. |
| `SpectralCrossAttention` | `model/RIIT.py` | Spectral cross-attention over feature channels: queries from reflectance, keys/values from illumination. |
| `SpatialCrossAttention` | `model/RIIT.py` | Linear-complexity (ELU+1) spatial cross-attention for spatially varying illumination. |
| `SpectralAttention` | `model/RIIT.py` | Channel-wise multi-head self-attention (attention map computed over spectral channels) for global band correlation. |
| `FeedForward` | `model/RIIT.py` | 3×3×3 depth-wise conv + two dilated depth-wise convs (dilation 2 and 3), gated by GELU. |
| `PhysicsInformedLoss` | `loss/char_sam_loss.py` | `Charbonnier(pred, gt) + λ · SAM(pred, gt)`. |

> **Naming.** In the paper, **SCCA** denotes the whole cross-stream update, implemented as `SpectralCrossAttention` + `SpatialCrossAttention` combined by the learnable scalar `alpha`; **SSA** denotes `SpectralAttention` + a 5×5 depth-wise spatial branch combined by `beta`.

### Default configuration

| Setting | Value |
|:--|:--|
| Input / output bands | 31 (400–700 nm, 10 nm step) |
| Base width `dim` | 64 |
| Transformer blocks per level | `[2, 2, 2]` |
| Attention heads | `[2, 4, 8]` |
| FFN expansion factor | 2.66 |
| Patch size during training | 256 × 256 |
| Global residual | `out = conv(x) + input` |

---

## 📊 Results

### Comparison with state-of-the-art methods

RIIT is compared against eight baselines spanning task-specific ASC (SIT), colour-constancy / low-light enhancement (Gray-World, DivIll, LPTN, IGDFormer) and generic HSI restoration (SERT, HCANet). All models are trained and tested separately on each dataset.

| Method | Params (M) | FLOPs (G) | BJTU-UVA PSNR↑ | SAM↓ | SSIM↑ | BJTU-UVA-E PSNR↑ | SAM↓ | SSIM↑ | ICVL PSNR↑ | SAM↓ | SSIM↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Gray-World | – | – | 21.55 | 10.08 | 0.601 | 18.89 | 9.33 | 0.425 | 19.95 | 13.38 | 0.580 |
| DivIll (ACM MM'21) | 0.813 | 53.41 | 23.90 | 4.05 | 0.764 | 37.76 | 4.84 | 0.967 | 30.19 | 3.09 | 0.952 |
| LPTN (TGRS'22) | 0.651 | 686.81 | 27.74 | 6.65 | 0.883 | 42.40 | 3.03 | 0.985 | 38.37 | 2.19 | 0.987 |
| SERT (CVPR'23) | 1.878 | 119.71 | 26.36 | 4.48 | 0.890 | 41.74 | 3.27 | 0.985 | 38.95 | 1.53 | 0.991 |
| HCANet (GRSL'24) | 6.158 | 122.39 | 28.61 | 4.95 | 0.905 | 43.17 | 2.31 | 0.987 | 35.02 | 1.79 | 0.986 |
| IGDFormer (PR'25) | 2.080 | 26.25 | 28.94 | 3.97 | 0.903 | 42.62 | 3.10 | 0.981 | 38.76 | 2.30 | 0.971 |
| SIT (CVPR'25) | 12.531 | 151.18 | 28.86 | 4.07 | 0.905 | 44.30 | 2.05 | 0.988 | 33.76 | 1.94 | 0.982 |
| **RIIT (ours)** | **6.423** | **88.26** | **29.32** | **3.13** | **0.911** | **44.95** | **1.91** | **0.988** | **41.03** | **0.92** | **0.995** |

> Numbers as reported in the paper. Parameter and FLOP counts follow the `thop` convention used there; counted with `torch`, the model holds 6.79 M parameters (the two strided down-sampling convolutions are not attributed by `thop`).

The BJTU-UVA row is reproducible with the released checkpoint: `test_results/riit/test_result_riit_dim64.txt` records `psnr 29.3243 / sam 3.1384 / ssim 0.9114`, plus RMSE 0.0471 and ERGAS 11.61 computed by the same script.

### Ablation study

Removing SCCA also removes cross-stream illumination guidance, so it doubles as an ablation of the guidance pathway itself.

| Setting | BJTU-UVA PSNR↑ | SAM↓ | SSIM↑ | ICVL PSNR↑ | SAM↓ | SSIM↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| w/o SCCA & SSA (backbone only) | 23.15 | 4.05 | 0.864 | 36.20 | 1.63 | 0.991 |
| w/o SCCA (SSA only) | 25.07 | 3.47 | 0.897 | 38.46 | 1.18 | 0.994 |
| w/o SSA (SCCA only) | 25.76 | 3.25 | 0.907 | 36.82 | 1.20 | 0.993 |
| **Full model** | **29.32** | **3.13** | **0.911** | **41.03** | **0.92** | **0.995** |

Two further analyses support the default configuration:

- **Kernel function** in spatial linear cross-attention (BJTU-UVA): `ELU(·)+1` 29.32 / 3.13 / 0.911 vs. `ReLU(·)` 28.60 / 3.57 / 0.909.
- **Hyper-parameters** (BJTU-UVA): `dim = 64` beats 16 / 32 / 96 / 128; `σ = 3` beats 1 / 5 / 7 / 9 and a learnable scale (28.53 / 3.53); `λ_SAM = 1.0` beats 0.1 / 0.5 / 0.8 / 1.5.

### Efficiency

Measured on a single RTX 4090, batch size 1; inference at 512 × 512, training at 256 × 256.

| Method | Inference latency (ms) | Inference memory (MB) | Training latency (ms) | Training memory (MB) |
|:--|:--:|:--:|:--:|:--:|
| SIT | 316.60 | 4123.3 | 272.79 | 8801.6 |
| **RIIT** | **286.21** | **2178.5** | **171.58** | **6520.1** |

### Spectral fidelity

The recovered spectra track the reference curve, while prior methods show either a global bias (Gray-World), a spectral tilt (DivIll, LPTN) or an over-smoothed profile (SERT).

<div align="center">
<img src="images/spectral_curves.png" width="92%">
<br>
<sub><b>Figure 2.</b> Predicted vs. reference spectra at pixels (128, 128), (384, 256) and (400, 300) on BJTU-UVA, BJTU-UVA-E and ICVL. RIIT (red) stays closest to the ground truth (black) across the whole 400–700 nm range.</sub>
</div>

### Qualitative comparison

<div align="center">
<img src="images/qualitative.png" width="100%">
<br>
<sub><b>Figure 3.</b> Absolute-error heatmaps (dark blue ≈ 0, red ≈ 1). Rows (a)–(b), (c)–(d) and (e)–(f) are samples from BJTU-UVA, BJTU-UVA-E and ICVL, respectively.</sub>
</div>

---

## 📁 Repository structure

```text
RIIT/
├── model/
│   └── RIIT.py                 # RIIT network: illumination stream, SCCA, SSA, dual-stream blocks
├── loss/
│   └── char_sam_loss.py        # Charbonnier + spectral angle (physics-informed) loss
├── utils/
│   ├── dataset_bjtu.py         # Cached BJTU-UVA loader
│   ├── dataset_bjtu_e.py       # Cached BJTU-UVA-E loader
│   ├── dataset_icvl.py         # ICVL patch loader + deterministic split builder
│   ├── resample.py             # 204 → 31 bands spectral resampling (400–700 nm)
│   ├── split_data.py           # Materialise train/val/test folders from the split JSON
│   ├── merge.py                # Pack image pairs into memmap caches (scene_*.npy / gt_*.npy)
│   ├── get_BJTU_E.py           # Illumination simulation → BJTU-UVA-E
│   ├── create_BJTU_E_json.py   # Build the BJTU-UVA-E split JSON
│   ├── cut_ICVL.py             # ICVL .mat (v7.3) → 512 × 512 corner patches
│   └── ICVL_deg_hard.py        # Hard degradation model for ICVL
├── images/                     # Figures used in this README
├── checkpoints/                # Released pretrained weights
├── train.py                    # Training entry point (argparse)
├── trainer.py                  # Training loop
├── test.py                     # Evaluation entry point (argparse)
├── tester.py                   # Evaluation loop
├── model_ini.py                # Model / loss factory
├── metrics_torch.py            # PSNR, RMSE, SAM, ERGAS, SSIM (GPU, batched)
├── ill_sim.pkl                 # 10 measured illuminant spectra (204 bands)
├── real_light_split.json       # BJTU-UVA split: 535 / 114 / 116
├── real_light_split_bjtu_e.json # BJTU-UVA-E split: 5350 / 1140 / 1160
├── requirements.txt
├── LICENSE                     # MIT
└── README.md
```

---

<a id="installation"></a>

## ⚙️ Installation

```bash
conda create -n riit python=3.10 -y
conda activate riit

# PyTorch matching your CUDA version (the paper uses 2.5.1 / CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Requirements: **Python 3.10** and **PyTorch ≥ 2.0** (paper: 3.10.16 / 2.5.1). All experiments were run on a single **NVIDIA RTX 4090 (24 GB)**; training needs a CUDA GPU, while evaluation also runs on CPU. Multi-GPU training is available via `DataParallel` (`--gpu_ids 0,1`).

---

## 🗂️ Data preparation

> ⚠️ **All scripts in `utils/` carry absolute paths at the top of the file** (e.g. `/opt/data/private/ld/data/...`). Edit them to your own directory layout before running anything.

### 1. BJTU-UVA — real-world HSI (`--dataset_name BJTU`)

765 pairs of real-world HSIs acquired with a Specim IQ platform, each paired with a calibration reference measured shortly after acquisition with the camera kept in place. The cubes are spectrally resampled to 31 bands over 400–700 nm at 512 × 512 pixels and split into **535 / 114 / 116** pairs for training / validation / testing.

Raw files: uncalibrated cubes `rw/rw_XXXX.npy` (204 bands, 12-bit) and reference reflectance `gt/gtRef_XXXX.npy`.

```bash
# (a) 204 → 31 bands (400–700 nm, 10 nm step), applied to both gt/ and rw/.
#     Edit input_gt / input_rw / output_* inside resample.py:main()
python utils/resample.py

# (b) (optional) materialise per-split folders for inspection
python utils/split_data.py

# (c) pack image pairs into memmap caches (edit SCENE_DIR / GT_DIR / OUT_DIR / JSON_PATH)
python utils/merge.py
```

`merge.py` produces the cache directory expected by `--merged_root`:

```text
<BJTU_ROOT>/merged_opt/
├── scene_train.npy     # (N, C, 256, 256) float32, in [0, 1]
├── gt_train.npy        # (N, C, 256, 256) float32
├── names_train.txt
├── scene_val.npy       # (N, H, W, C)  ← val/test caches are HWC
├── gt_val.npy
├── names_val.txt
├── scene_test.npy      # (N, H, W, C)
├── gt_test.npy
└── names_test.txt
```

The training split is stored as **CHW** at the 256 × 256 training-patch size, while the val/test splits are stored as **HWC** at full 512 × 512 resolution so that whole scenes can be evaluated without cropping. `merge.py` writes this layout automatically; the loaders verify it and raise a descriptive error on mismatch.

> If your merged training cache is still at 512 × 512, crop it to 256 × 256 (the paper trains on random 256 × 256 patches) or extend `merge.py` with a crop step.

Split sizes (from `real_light_split.json`): **535 / 114 / 116** for train / val / test.

### 2. BJTU-UVA-E — illumination-robust extension (`--dataset_name BJTU_E`)

Every reference cube of BJTU-UVA is multiplied, wavelength by wavelength, by the 10 physically measured illuminant spectra in `ill_sim.pkl` — five natural conditions (`du` dusk, `sd` sunny, `cd` cloudy, `rd` rainy, `sh` shadow) and five colour-filtered ones (`ye`, `pu`, `re`, `bl`, `gr`) — giving **7,650** pairs from the original 765 scenes, split into **5,350 / 1,140 / 1,160**.

```bash
# synthesise illuminated cubes: gtRef_1287.npy → {du,sd,...}_rw_1287.npy under BJTU_E/rw/
python utils/get_BJTU_E.py

# build the matching split JSON (every file name replicated with the 10 prefixes)
python utils/create_BJTU_E_json.py

# then run resample.py and merge.py again with the BJTU-UVA-E paths
```

Each synthesised cube `{ill}_rw_{id}.npy` shares the reference `gtRef_{id}.npy`; the reference folder therefore holds the same ten prefixed copies (the renaming helper is kept at the bottom of `get_BJTU_E.py`).

> The multiplication is spatially uniform per band, so it varies the illuminant spectrum but does not create new cast-shadow boundaries or mixed spatial illuminants.

### 3. ICVL — simulated hard degradation (`--dataset_name ICVL`)

The BGU ICVL dataset (31 bands, 400–700 nm) is cropped into four non-overlapping 512 × 512 corner patches per scene, giving **800** samples split into **600 / 100 / 100** for training / validation / testing.

```bash
python utils/cut_ICVL.py        # .mat (v7.3) → 4 × 512×512 corner patches per scene
python utils/ICVL_deg_hard.py   # per-patch degradation (see below) → input/ + target/
```

Each degraded input is synthesised offline from its reference patch by (i) a per-patch attenuation ρ ~ U(0.1, 0.6), (ii) a coupled spatial–spectral illumination field, (iii) a 32 × 32 block-wise gain field, (iv) a 3-band spectral mixing step, (v) signal-dependent shot noise plus read noise, and (vi) clipping to 12 bits with rounding. See Eq. (5)–(7) of the paper for the exact formulation.

The ICVL loader expects plain patch pairs (no memmap cache):

```text
<ICVL_ROOT>/
├── input/     # degraded patches, (H, W, 31), 12-bit
└── target/    # clean reflectance patches, identical file names
```

Patches are normalised by `/4096` and transposed to CHW on the fly. The split is deterministic (`seed = 42`, `num_val = num_test = 100`), matching the 600 / 100 / 100 partition used in the paper.

---

## 🚀 Training

```bash
# ---------- BJTU-UVA (500 epochs) ----------
python train.py \
    --dataset_name BJTU \
    --merged_root /path/to/BJTU_resample/merged_opt \
    --method_name riit --ex_name _dim64 --dim 64 \
    --epochs 500 --batch_size 1 --lr 1e-4 \
    --save_by_epoch 10

# ---------- BJTU-UVA-E (50 epochs) ----------
python train.py \
    --dataset_name BJTU_E \
    --merged_root /path/to/BJTU_E_resample/merged_opt \
    --method_name riit --ex_name _dim64_e --dim 64 \
    --epochs 50 --batch_size 1 --lr 1e-4

# ---------- ICVL (500 epochs) ----------
python train.py \
    --dataset_name ICVL \
    --merged_root /path/to/ICVL_deg \
    --method_name riit --ex_name _dim64_icvl --dim 64 \
    --epochs 500 --batch_size 1 --lr 1e-4

# ---------- resume ----------
python train.py --dataset_name BJTU --merged_root /path/to/merged_opt \
    --resume_path checkpoints/riit_dim64/riit_epoch200.pth
```

> **Epoch budget.** The paper trains 500 epochs on BJTU-UVA and ICVL, and **50 epochs on BJTU-UVA-E**, because the extended set already contains 10× the samples.

**Key training settings** (defaults in `train.py`)

| Argument | Default | Meaning |
|:--|:--|:--|
| `--epochs` | 500 | Training epochs |
| `--batch_size` | 1 | Batch size **per GPU**; scaled automatically when several GPUs are given |
| `--lr` | 1e-4 | Initial Adam learning rate |
| `--lr_decrease_epoch` / `--lr_decrease_factor` | 200 / 0.5 | Halve the LR every 200 epochs |
| `--lambda_sam` | 1.0 | Weight of the spectral angle term |
| `--grad_clip` | 0 (off) | Gradient-clipping max norm; **disabled by default, as in the paper** |
| `--crop_size` | 256 | Random crop size (used by the ICVL loader; BJTU-UVA caches are pre-cropped) |
| `--save_by_epoch` | 1 | Write a checkpoint every *N* epochs |
| `--gpu_ids` | `0` | Comma-separated GPU ids, e.g. `0,1` |
| `--num_workers` | 0 | Dataloader workers |

**Outputs**

```text
checkpoints/riit_dim64/riit_epoch{N}.pth      # one checkpoint per --save_by_epoch epochs
train_loss/riit/train_loss_riit_dim64.txt     # per-epoch loss log
```

> 💡 Each checkpoint is ≈ 79 MB, so a 500-epoch run with `--save_by_epoch 1` writes ≈ 40 GB. Use `--save_by_epoch 10` (or larger) in practice.
>
> 💡 **Validation.** In the paper, validation runs every 10 epochs on the validation split and the best-performing checkpoint is used for testing. The `val()` routine is included in `trainer.py` but commented out of the epoch loop (the val loader is only constructed for BJTU-UVA); uncomment the call and add checkpoint selection to reproduce that protocol. It logs PSNR / RMSE / ERGAS / SAM / SSIM.

---

## 🔍 Evaluation

```bash
python test.py \
    --dataset_name BJTU \
    --merged_root /path/to/BJTU_resample/merged_opt \
    --model_path checkpoints/riit_dim64_bjtu.pth \
    --method_name riit --ex_name _dim64 --dim 64

# BJTU-UVA-E
python test.py --dataset_name BJTU_E --merged_root /path/to/BJTU_E_resample/merged_opt \
    --model_path checkpoints/riit_dim64_bjtu_e.pth --ex_name _dim64_e

# ICVL
python test.py --dataset_name ICVL --merged_root /path/to/ICVL_deg \
    --model_path checkpoints/riit_dim64_icvl.pth --ex_name _dim64_icvl
```

Results are appended to `test_results/riit/test_result_riit{ex_name}.txt`, e.g.

```text
rmse:0.0471	psnr:29.3243	ergas:11.6121	sam:3.1384	ssim:0.9114
```

---

## 🏋️ Pretrained checkpoints

| File | Dataset | Trained epochs | Notes |
|:--|:--|:--:|:--|
| `checkpoints/riit_dim64_bjtu.pth` | BJTU-UVA | 500 | Real-world HSI, 31 bands |
| `checkpoints/riit_dim64_bjtu_e.pth` | BJTU-UVA-E | 50 | 10 measured illuminant conditions |
| `checkpoints/riit_dim64_icvl.pth` | ICVL | 500 | Simulated hard degradation |

All checkpoints use `dim = 64` and `num_blocks = [2, 2, 2]`. They are stored with the full training state (`model_state_dict`, `optimizer_state_dict`, `epoch`, `lr`), so they can be used both for evaluation and for resuming training.

> If the repository becomes too heavy for a plain clone, track `checkpoints/` with [Git LFS](https://git-lfs.com/) or attach the weights to a GitHub Release.

---

## 📐 Metrics

The paper reports **PSNR**, **SAM** and **SSIM**; `metrics_torch.py` additionally computes RMSE and ERGAS. All metrics are evaluated on reflectance in `[0, 1]` (12-bit data divided by 4096), per image and then averaged over the test set.

| Metric | Definition | Note |
|:--|:--|:--|
| **PSNR** ↑ | `10·log10(1 / MSE)` | Peak value fixed to 1.0 — spatial accuracy |
| **SAM** ↓ | Mean angle between predicted and reference spectra, in degrees | Spectral fidelity; scale-invariant |
| **SSIM** ↑ | Band-wise SSIM, 11×11 Gaussian window (σ = 1.5) | Structural fidelity |
| **RMSE** ↓ | `sqrt(mean((R̂ − R)²))` | Extra, logged by the script |
| **ERGAS** ↓ | `100/r · sqrt(mean_c((RMSE_c/μ_c)²))` | Extra, `r = 4` |

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2027riit,
  title     = {Reflectance--illumination interaction transformer for learning-based
               automatic spectral calibration of hyperspectral images},
  author    = {Li, Dian and Hao, Siyuan and Fang, Cheng and Ye, Yuanxin},
  journal   = {Expert Systems with Applications},
  volume    = {334},
  pages     = {134818},
  year      = {2027},
  doi       = {10.1016/j.eswa.2026.134818},
  publisher = {Elsevier}
}
```

---

## 🙏 Acknowledgements

- The **BJTU-UVA / BJTU-UVA-E** benchmarks for real-world automatic spectral calibration.
- The **BGU ICVL** dataset for the hyperspectral image patches.
- The LayerNorm / gated feed-forward design follows [Restormer](https://github.com/swz30/Restormer); we thank the authors for releasing their code.

---

<a id="license"></a>

## ⚖️ License

This project is released under the [MIT License](LICENSE).

```text
Copyright (c) 2026 Dian Li, Siyuan Hao, Cheng Fang, Yuanxin Ye
```

You are free to use, modify and redistribute the code, including for commercial purposes, provided that the copyright notice and permission notice are retained. If you use this code in academic work, a citation to our paper is appreciated.

## 📬 Contact

**Dian Li** — [andorld@163.com](mailto:andorld@163.com) · GitHub: [@Andor-Li](https://github.com/Andor-Li)

For questions about the code, please open an issue. For questions about the paper, feel free to reach out by email.
