import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import model_ini
from metrics_torch import *
from utils.dataset_bjtu import BJTU_CacheDataset
from utils.dataset_bjtu_e import BJTU_E_CacheDataset
from utils.dataset_icvl import ICVLCalibrationDataset, build_datasets

def Trainer(opt):
    print(opt.method_name)
    print(opt.dataset_name)
    original_batch_size = opt.batch_size
    # 1. 解析 GPU IDs
    if isinstance(opt.gpu_ids, str):
        opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(',') if x.strip().isdigit()]
    elif isinstance(opt.gpu_ids, int) and opt.gpu_ids >= 0:
        opt.gpu_ids = [opt.gpu_ids]
    elif isinstance(opt.gpu_ids, (list, tuple)):
        pass  # 已经是列表或元组
    else:
        opt.gpu_ids = []  # 默认为空列表 (CPU)

    gpu_num = len(opt.gpu_ids)

    
    # 2. 设置设备, multi_gpu 标志, 和扩展 batch size
    if gpu_num > 0 and torch.cuda.is_available():
        print(f"Using GPU IDs: {opt.gpu_ids}")
        print(f"PyTorch can see {torch.cuda.device_count()} devices.")
        device = torch.device(f'cuda:{opt.gpu_ids[0]}')

        if gpu_num > 1:
            opt.multi_gpu = True
            print(f"Using {gpu_num} GPUs for DataParallel training.")
        else:
            opt.multi_gpu = False  # 使用 1 个 GPU
            print("Using 1 GPU for training.")

        # 假设 opt.batch_size 是 *每个GPU* 的 batch size
        opt.batch_size *= gpu_num
    
    else:
        device = torch.device('cpu')
        opt.multi_gpu = False
        gpu_num = 0 # 确保 gpu_num 为 0
        print("Using CPU for training.")
        # MODIFIED: CPU 训练时不扩展 batch size (gpu_num 为 0)

    # Create folder for saving models
    model_ini.check_path(opt.save_path)
    cudnn.benchmark = opt.cudnn_benchmark

    # Dataset and DataLoader
    # NOTE: `--merged_root` (if given) overrides the default cache directory below.
    if opt.dataset_name == 'BJTU':
        opt.merged_root = opt.merged_root or '/opt/data/private/ld/data/BJTU_resample/merged_opt'
        train_dataset = BJTU_CacheDataset(opt, 'train')
        val_dataset = BJTU_CacheDataset(opt, 'val')

        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=original_batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True)
    
        
    elif opt.dataset_name == 'BJTU_E':
        opt.merged_root = opt.merged_root or '/opt/data/private/ld/data/BJTU_E_resample/merged_opt'
        train_dataset = BJTU_E_CacheDataset(opt, 'train')
        # val_dataset = BJTU_E_CacheDataset(opt, 'val')

        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers, pin_memory=True)
        # val_loader = DataLoader(val_dataset, batch_size=original_batch_size, shuffle=False,
        #                         num_workers=opt.num_workers, pin_memory=True)

    elif opt.dataset_name == 'ICVL':
        # ICVL uses per-patch .npy pairs: <root>/input/*.npy and <root>/target/*.npy
        opt.merged_root = opt.merged_root or '/opt/data/private/ld/data/ICVL_deg'
        train_dataset = build_datasets(opt=opt, root_dir=opt.merged_root,
                                       num_val=100, num_test=100, seed=42, flag='train')
        val_dataset = build_datasets(opt=opt, root_dir=opt.merged_root,
                                     num_val=100, num_test=100, seed=42, flag='val')

        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                num_workers=opt.num_workers, pin_memory=True)
        # val_loader = DataLoader(val_dataset, batch_size=original_batch_size, shuffle=False,
        #                         num_workers=opt.num_workers, pin_memory=True)

    print('Total training images:', len(train_dataset))
    print('dataloader ready')

    # Initialize generator
    # MODIFIED: 首先将生成器移动到主设备
    generator, loss = model_ini.create_generator(opt)
    generator = generator.to(device)
    loss = loss.to(device)
    
    # MODIFIED: 使用 opt.multi_gpu 标志来决定是否包装
    if opt.multi_gpu:
        generator = nn.DataParallel(generator, device_ids=opt.gpu_ids)
        print(f'DataParallel initialized with device_ids: {opt.gpu_ids}')

    # Optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                                     weight_decay=opt.weight_decay)

    # Learning rate adjustment function (无变化)
    def adjust_learning_rate(opt, epoch, optimizer):
        lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    start_epoch = 0

    if opt.resume_path and os.path.isfile(opt.resume_path):
        print(f"==> Loading checkpoint from {opt.resume_path}")
        checkpoint = torch.load(opt.resume_path, map_location=device)

        # Load model
        if opt.multi_gpu:
            generator.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            generator.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer
        optimizer_G.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load progress
        start_epoch = checkpoint["epoch"]

        print(f"==> Resumed from epoch {start_epoch}")
    else:
        print("==> No checkpoint found. Training from scratch.")


    prev_time = time.time()
    
    for epoch in range(start_epoch, opt.epochs):
        adjust_learning_rate(opt, epoch + 1, optimizer_G)
        for index, (img_A, img_B, _) in enumerate(train_loader):
            # MODIFIED: 将数据移动到主设备
            # DataParallel 将自动处理分发到其他 GPU
            img_A = img_A.to(device, non_blocking=True)
            hs_ori = img_B.to(device, non_blocking=True)          
            optimizer_G.zero_grad()
            out = generator(img_A)
            loss_total = loss(out, hs_ori)
         
            # MODIFIED: backward() 会自动处理梯度聚合
            loss_total.backward()
            # Gradient clipping is disabled by default (--grad_clip 0), matching the paper.
            if getattr(opt, 'grad_clip', 0) and opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=opt.grad_clip)
            optimizer_G.step()

            iters_done = epoch * len(train_loader) + index
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            print("\r[Epoch %d/%d] [Batch %d/%d] Time left: %s" %
                  (epoch + 1, opt.epochs, index, len(train_loader), time_left), end='')

        loss_total_detach = loss_total.detach().item()
        print(f"\n epoch:{epoch+1}, loss:{loss_total_detach:.4f}")

        valstr = "epoch %d\tloss:%.4f" % (epoch+1, loss_total_detach)
        write_result(valstr, opt, info="train_loss")


        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     val(opt, generator, epoch + 1, device, loss_total.item(), original_batch_size, val_loader)

        # Save one checkpoint every `save_by_epoch` epochs (default: every epoch)
        if opt.save_by_epoch and (epoch + 1) % opt.save_by_epoch == 0:
            save_model(opt, epoch + 1, generator, optimizer_G)

            
def val(opt, generator, epoch, device, loss, original_batch_size, val_loader):
    # 1. 切换到评估模式
    generator.eval()
    
    # 2. 获取单卡模型 (绕过 DataParallel 的 Scatter/Gather 开销)
    if isinstance(generator, nn.DataParallel):
        eval_model = generator.module
    else:
        eval_model = generator
    
    # 初始化累加器 (使用 Tensor 存储在 GPU 上)
    total_metrics = torch.zeros(5, device=device) # [psnr, rmse, ergas, sam]
    num_samples = 0
    
    print("Validating...", end='')
    
    with torch.no_grad():
        for index, (img1, img_ori, path) in enumerate(val_loader):
            # 数据上 GPU
            img1 = img1.to(device)
            img_ori = img_ori.to(device)
            
            # 推理
            out = eval_model(img1)
            # out, pred_L = eval_model(img1)

            # 注意：calc_ergas_torch 需要你知道缩放比例 ratio，通常是 4
            dataset_ratio = 4 
            
            # 我们累加整个 Batch 的指标总和
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
    print(f"\r[Epoch {epoch}] Validation Complete.")
    valstr = f"epoch {epoch}\trmse:{rmse_res:.4f}\tpsnr:{psnr_res:.4f}\tergas:{ergas_res:.4f}\tsam:{sam_res:.4f}\tssim:{ssim_res:.4f}\tloss:{loss:.4f}"
    print(valstr)
    write_result(valstr, opt, info="val_results")
    
    # 恢复训练模式 (注意要恢复原始的 generator)
    generator.train()

def write_result(res, opt, info = "train_loss"):
    model_dir = f"./{info}/{opt.method_name}"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"{info}_{opt.method_name}{opt.ex_name}.txt"), 'a+') as f:
        f.write(res + '\n')


def save_model(opt, epoch, generator, optimizer_G):
    gen_state = generator.module.state_dict() if opt.multi_gpu else generator.state_dict()
    state = {
        "epoch": epoch,
        "model_state_dict": gen_state,
        "optimizer_state_dict": optimizer_G.state_dict(),
        "lr": optimizer_G.param_groups[0]["lr"]
    }

    folder = os.path.join(opt.save_path, f'{opt.method_name}{opt.ex_name}')
    os.makedirs(folder, exist_ok=True)
    filename = f"{opt.method_name}_epoch{epoch}.pth"
    path = os.path.join(folder, filename)

    torch.save(state, path)
    print(f"Saved checkpoint: {path}")