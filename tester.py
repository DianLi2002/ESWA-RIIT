import os
import torch
from torch.utils.data import DataLoader
import model_ini
from metrics_torch import *
from utils.dataset_bjtu import BJTU_CacheDataset
from utils.dataset_bjtu_e import BJTU_E_CacheDataset
from utils.dataset_icvl import ICVLCalibrationDataset, build_datasets

def Tester(opt):
    print(opt.method_name)
    print(opt.dataset_name)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu_id}')
        print(f"Using GPU IDs: {opt.gpu_id}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available, running on CPU.")
    generator = model_ini.create_generator(opt)
    generator = generator.to(device)

    if opt.dataset_name == 'BJTU':
        opt.merged_root = opt.merged_root or '/opt/data/private/ld/data/BJTU_resample/merged_opt'
        test_dataset = BJTU_CacheDataset(opt, 'test')

        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True)
        
    elif opt.dataset_name == 'BJTU_E':
        opt.merged_root = opt.merged_root or '/opt/data/private/ld/data/BJTU_E_resample/merged_opt'
        test_dataset = BJTU_E_CacheDataset(opt, 'test')

        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True)

        
    elif opt.dataset_name == 'ICVL':
        opt.merged_root = opt.merged_root or '/opt/data/private/ld/data/ICVL_deg'
        test_dataset = build_datasets(opt=opt, root_dir=opt.merged_root,
                                      num_val=100, num_test=100, seed=42, flag='test')

        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True)
        
    print(f"Test set size: {len(test_dataset)}")
    
    print(f"Loading model...")

    checkpoint = torch.load(f'{opt.model_path}', map_location=device)
    generator.load_state_dict(checkpoint["model_state_dict"])

    # 初始化累加器 (使用 Tensor 存储在 GPU 上)
    total_metrics = torch.zeros(5, device=device) # [psnr, rmse, ergas, sam, ssim]
    num_samples = 0
    
    print("Testing...")
    generator.eval()
    with torch.no_grad():
        for index, (img1, img_ori, path) in enumerate(test_loader):
            # 数据上 GPU
            img1 = img1.to(device)
            img_ori = img_ori.to(device)
            
            # 推理
            out = generator(img1)

            # 注意：calc_ergas_torch 需要你知道缩放比例 ratio，通常是 4
            dataset_ratio = 4 
            batch_size_current = img1.size(0)
            
            # 针对 Batch 计算平均值
            b_psnr = calc_psnr_torch(img_ori, out)
            b_rmse = calc_rmse_torch(img_ori, out)
            b_ergas = calc_ergas_torch(img_ori, out, ratio=dataset_ratio)
            b_sam = calc_sam_torch(img_ori, out)
            b_ssim = calc_ssim_torch(img_ori, out)

            total_metrics[0] += b_psnr * batch_size_current
            total_metrics[1] += b_rmse * batch_size_current
            total_metrics[2] += b_ergas * batch_size_current
            total_metrics[3] += b_sam * batch_size_current
            total_metrics[4] += b_ssim * batch_size_current
            
            num_samples += batch_size_current

        # 计算平均值
        avg_metrics = total_metrics / num_samples
        
        # 转回 CPU 仅用于打印
        psnr_res = avg_metrics[0].item()
        rmse_res = avg_metrics[1].item()
        ergas_res = avg_metrics[2].item()
        sam_res = avg_metrics[3].item()
        ssim_res = avg_metrics[4].item()

        # 打印和记录
        valstr = f"rmse:{rmse_res:.4f}\tpsnr:{psnr_res:.4f}\tergas:{ergas_res:.4f}\tsam:{sam_res:.4f}\tssim:{ssim_res:.4f}"

        print(valstr)
        write_result(valstr, opt)

def write_result(res, opt):
    model_dir = f"./test_results/{opt.method_name}"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"test_result_{opt.method_name}{opt.ex_name}.txt"), 'a+') as f:
        f.write(res + '\n')