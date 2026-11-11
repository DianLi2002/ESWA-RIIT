import torch
import torch.nn.functional as F

def calc_psnr_torch(img1, img2, max_val=1.0):

    B = img1.size(0)

    mse = torch.mean((img1 - img2) ** 2, dim=(1,2,3))

    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-10))

    return psnr.mean()

def calc_rmse_torch(img1, img2):
    rmse = torch.sqrt(((img1 - img2) ** 2).mean(dim=(1,2,3)))
    return rmse.mean()

def calc_sam_torch(img1, img2):
    # img1, img2: [C, H, W]
    # 也就是在通道维(dim=0)上计算角度
    # 如果输入是 [B, C, H, W], 则 dim=1
    if img1.ndim == 3:
        dim = 0
    else:
        dim = 1
        
    dot_product = torch.sum(img1 * img2, dim=dim)
    norm_img1 = torch.norm(img1, dim=dim)
    norm_img2 = torch.norm(img2, dim=dim)
    
    # 加上 1e-8 防止除以 0
    denominator = norm_img1 * norm_img2 + 1e-8
    cosine = torch.clamp(dot_product / denominator, -1.0, 1.0)
    sam_map = torch.acos(cosine)
    
    # # 转换为度数并取平均
    # sam = torch.mean(sam_map) * (180.0 / 3.141592653589793)
    # return sam

    # new
    if img1.ndim == 4:
        sam = sam_map.mean(dim=(1,2))  # 每张图
        sam = sam.mean()
    else:
        sam = sam_map.mean()
    sam = sam * (180.0 / 3.141592653589793)
    return sam

def calc_ergas_torch(img_tgt, img_fus, ratio=4):
    # img_tgt: 原始高光谱 (Reference)
    # img_fus: 融合后图像
    # ratio: 空间分辨率缩放倍数 (通常是 4 或 8，根据你的数据集设定)
    
    # [C, H, W] -> [B, C, H, W] 以便统一处理
    if img_tgt.ndim == 3:
        img_tgt = img_tgt.unsqueeze(0)
        img_fus = img_fus.unsqueeze(0)

    # 按通道计算 RMSE
    # diff: [B, C, H, W]
    diff = img_tgt - img_fus
    # mse_per_band: [B, C] -> 平均掉 H, W
    mse_per_band = torch.mean(diff ** 2, dim=(2, 3))
    rmse2_per_band = mse_per_band 
    
    # 计算每个波段的均值
    mean_per_band = torch.mean(img_tgt, dim=(2, 3))
    
    # ERGAS 公式
    # sum( (RMSE_i / MEAN_i)^2 )
    sum_term = torch.sum(rmse2_per_band / (mean_per_band ** 2 + 1e-8), dim=1)
    
    ergas = 100 / ratio * torch.sqrt(sum_term / img_tgt.shape[1])
    
    # 如果是一个 batch，取平均
    return torch.mean(ergas)


def _gaussian_window_2d(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window


def calc_ssim_torch(img1, img2, max_val=1.0, window_size=11):
    """
    Multi-spectral SSIM (Band-wise SSIM)

    img1, img2: [C, H, W] or [B, C, H, W]
    return: scalar SSIM
    """

    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape
    device = img1.device
    dtype = img1.dtype

    window = _gaussian_window_2d(window_size, 1.5, device, dtype)
    window = window.view(1, 1, window_size, window_size)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_per_band = []

    for c in range(C):
        x = img1[:, c:c+1, :, :]
        y = img2[:, c:c+1, :, :]

        mu_x = F.conv2d(x, window, padding=window_size // 2)
        mu_y = F.conv2d(y, window, padding=window_size // 2)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=window_size // 2) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

        # ssim_per_band.append(ssim_per_image)
        ssim_per_image = ssim_map.mean(dim=(1,2,3)) # new
        ssim_per_band.append(ssim_per_image) # new

    # 波段平均 + batch 平均
    # ssim = torch.stack(ssim_per_band).mean()
    ssim_per_band = torch.stack(ssim_per_band, dim=1) # new
    ssim = ssim_per_band.mean(dim=1)  # 每张图 new
    ssim = ssim.mean()  # batch平均 new

    return ssim

