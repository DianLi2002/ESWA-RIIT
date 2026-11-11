import os
import numpy as np
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

# 忽略插值时的警告
warnings.filterwarnings('ignore', message='.*bounds_error.*')

def hyper_resample(datacube):
    # 目标31通道波长（400-700nm等间隔）
    samp_waves = np.linspace(400, 700, 31)
    # 原始204通道对应的波长列表（与原函数保持一致）
    wavelens = np.array([
        397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29,
        426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
        455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
        484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
        513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
        542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
        572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
        601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
        631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
        660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
        690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
        720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
        750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
        780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
        810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
        841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
        871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
        902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
        932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
        963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
        994.31, 997.40, 1000.49, 1003.58
    ])

    # 验证输入通道数
    if datacube.shape[-1] != 204:
        raise ValueError(f"输入数据通道数应为204，实际为{datacube.shape[-1]}")

    # 初始化输出数据立方体
    output_cube = np.zeros((datacube.shape[0], datacube.shape[1], len(samp_waves)), dtype=datacube.dtype)

    # 逐像素重采样 - 优化：使用向量化操作
    # 创建插值函数（对所有像素使用相同的波长）
    # 我们将对每个像素进行循环，但使用预计算的插值权重可以加速
    for i in range(datacube.shape[0]):
        for j in range(datacube.shape[1]):
            spectrum = datacube[i, j, :]
            # 线性插值（超出范围时外推）
            interp_func = interp1d(
                wavelens,
                spectrum,
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            output_cube[i, j, :] = interp_func(samp_waves)

    return output_cube


def process_single_file(args):
    """处理单个文件的函数，供多进程使用"""
    input_path, output_path = args
    try:
        # 读取数据
        datacube = np.load(input_path)
        
        # 重采样
        resampled = hyper_resample(datacube)
        
        # 保存结果
        np.save(output_path, resampled)
        
        # 返回成功标志和文件名
        return True, os.path.basename(input_path), resampled.shape
    except Exception as e:
        # 返回失败信息
        return False, os.path.basename(input_path), str(e)


def batch_resample_parallel(input_dir, output_dir, num_workers=None):
    """并行批量处理目录下的.npy文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"警告: {input_dir} 中没有找到.npy文件")
        return
    
    # 创建任务列表：每个任务是一个(输入路径, 输出路径)的元组
    tasks = []
    for filename in npy_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        tasks.append((input_path, output_path))
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = min(cpu_count(), len(npy_files))
    
    print(f"开始处理 {len(npy_files)} 个文件，使用 {num_workers} 个进程...")
    
    # 用于统计成功和失败的文件
    success_count = 0
    fail_count = 0
    
    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap_unordered(process_single_file, tasks),
            total=len(tasks),
            desc=f"处理 {os.path.basename(input_dir)}",
            unit="file"
        ))
    
    # 统计结果
    for success, filename, info in results:
        if success:
            success_count += 1
            if success_count <= 3:  # 只显示前3个成功文件的信息
                print(f"✓ {filename}: 成功，输出形状 {info}")
        else:
            fail_count += 1
            print(f"✗ {filename}: 失败 - {info}")
    
    print(f"处理完成: {success_count} 个成功, {fail_count} 个失败")
    
    # 显示总结信息
    if fail_count == 0:
        print(f"✓ 所有文件处理成功！")
    elif fail_count > 0:
        print(f"⚠ 有 {fail_count} 个文件处理失败")


def batch_resample_vectorized(input_dir, output_dir, num_workers=None):
    """并行批量处理目录下的.npy文件（向量化版本）"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"警告: {input_dir} 中没有找到.npy文件")
        return
    
    # 创建任务列表：每个任务是一个(输入路径, 输出路径)的元组
    tasks = []
    for filename in npy_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        tasks.append((input_path, output_path))
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = min(cpu_count(), len(npy_files))
    
    print(f"开始处理 {len(npy_files)} 个文件，使用 {num_workers} 个进程...")
    
    # 使用进程池并行处理
    success_count = 0
    fail_count = 0
    
    with Pool(processes=num_workers) as pool:
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc=f"处理 {os.path.basename(input_dir)}", unit="file") as pbar:
            # 使用imap_unordered获取结果
            for success, filename, info in pool.imap_unordered(process_single_file, tasks):
                if success:
                    success_count += 1
                    if success_count <= 3:  # 只显示前3个成功文件的信息
                        pbar.set_postfix_str(f"✓ {filename}: {info[2]} channels")
                else:
                    fail_count += 1
                    pbar.set_postfix_str(f"✗ {filename}: {info}")
                pbar.update(1)
    
    print(f"处理完成: {success_count} 个成功, {fail_count} 个失败")


def optimized_hyper_resample(datacube):
    """优化版本的重采样函数，减少内存分配"""
    # 目标31通道波长（400-700nm等间隔）
    samp_waves = np.linspace(400, 700, 31)
    # 原始204通道对应的波长列表
    wavelens = np.array([
        397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29,
        426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
        455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
        484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
        513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
        542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
        572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
        601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
        631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
        660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
        690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
        720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
        750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
        780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
        810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
        841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
        871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
        902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
        932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
        963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
        994.31, 997.40, 1000.49, 1003.58
    ])

    # 验证输入通道数
    if datacube.shape[-1] != 204:
        raise ValueError(f"输入数据通道数应为204，实际为{datacube.shape[-1]}")

    # 初始化输出数据立方体
    output_cube = np.zeros((datacube.shape[0], datacube.shape[1], len(samp_waves)), dtype=datacube.dtype)
    
    # 预计算插值权重（更快的插值方法）
    # 对于线性插值，我们可以直接计算权重
    # 找到每个目标波长对应的两个最近原始波长的索引和权重
    
    # 为每个目标波长找到最近的原始波长索引
    indices = np.searchsorted(wavelens, samp_waves)
    
    # 确保索引在有效范围内
    indices = np.clip(indices, 1, len(wavelens) - 1)
    
    # 计算权重
    left_idx = indices - 1
    right_idx = indices
    
    left_wavelens = wavelens[left_idx]
    right_wavelens = wavelens[right_idx]
    
    # 避免除以零
    denom = right_wavelens - left_wavelens
    denom[denom == 0] = 1
    
    right_weight = (samp_waves - left_wavelens) / denom
    left_weight = 1 - right_weight
    
    # 向量化计算所有像素
    for i in range(datacube.shape[0]):
        for j in range(datacube.shape[1]):
            spectrum = datacube[i, j, :]
            # 使用预计算的权重进行线性插值
            output_cube[i, j, :] = left_weight * spectrum[left_idx] + right_weight * spectrum[right_idx]
    
    return output_cube


def process_single_file_optimized(args):
    """处理单个文件的函数（优化版本），供多进程使用"""
    input_path, output_path = args
    try:
        # 读取数据
        datacube = np.load(input_path)
        
        # 使用优化版本的重采样
        resampled = optimized_hyper_resample(datacube)
        
        # 保存结果
        np.save(output_path, resampled)
        
        # 返回成功标志和文件名
        return True, os.path.basename(input_path), resampled.shape
    except Exception as e:
        # 返回失败信息
        return False, os.path.basename(input_path), str(e)


def main():
    # 定义输入输出路径
    input_gt = "/opt/data/private/ld/data/BJTU_E/gt"
    input_rw = "/opt/data/private/ld/data/BJTU_E/rw"
    output_gt = "/opt/data/private/ld/data/BJTU_E_resample/gt"
    output_rw = "/opt/data/private/ld/data/BJTU_E_resample/rw"
    
    # 设置工作进程数（可以根据需要调整）
    num_workers = 10  # 使用所有CPU核心
    print(f"系统CPU核心数: {num_workers}")
    
    # 处理gt_files目录
    print("\n" + "="*50)
    print("开始处理gt_files目录...")
    print("="*50)
    batch_resample_parallel(input_gt, output_gt, num_workers=num_workers)
    
    # 处理rw目录
    print("\n" + "="*50)
    print("开始处理rw目录...")
    print("="*50)
    batch_resample_parallel(input_rw, output_rw, num_workers=num_workers)
    
    print("\n" + "="*50)
    print("所有文件处理完成!")
    print("="*50)


if __name__ == "__main__":
    main()