import pickle
import numpy as np
import os
import re
from multiprocessing import Pool
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SRC_DIR_GT = "/opt/data/private/ld/data/BJTU/gt"
OUT_DIR = "/opt/data/private/ld/data/BJTU_E"

TARGET_DIR = os.path.join(OUT_DIR, "rw")

os.makedirs(TARGET_DIR, exist_ok=True)

def load_illumination_data(pickle_path='ill_sim.pkl'):
    """加载照明数据，避免重复加载"""
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    logger.info(f"Loaded illumination data with keys: {list(data.keys())}")
    return data

def extract_number_from_filename(fname):
    """从 gtRef_1287.npy 中提取数字部分 1287，若失败则返回原文件名（不含扩展名）"""
    base = os.path.splitext(fname)[0]          # gtRef_1287
    # 匹配模式：gtRef_后跟数字
    match = re.search(r'gtRef_(\d+)', base)
    if match:
        return match.group(1)                  # '1287'
    else:
        logger.warning(f"Cannot extract number from {fname}, using full base name")
        return base

def process_one_file(args):
    """处理单个文件的通用函数"""
    fname, src_dir, out_dir, ill_data = args
    try:
        src_path = os.path.join(src_dir, fname)
        data_array = np.load(src_path)
        
        # 提取数字ID用于新文件名
        num_id = extract_number_from_filename(fname)
        
        for ill_name, ill_coeff in ill_data.items():
            # 验证照明系数形状
            if not isinstance(ill_coeff, (np.ndarray, list, int, float)):
                logger.warning(f"Invalid illumination coefficient type for {ill_name}")
                continue
                
            # 进行计算
            sc_data = data_array * ill_coeff
            
            # 生成新文件名：光照前缀_rw_数字ID.npy
            safe_ill_name = str(ill_name).replace('/', '_').replace('\\', '_')
            output_name = f"{safe_ill_name}_rw_{num_id}.npy"
            output_path = os.path.join(out_dir, output_name)
            
            np.save(output_path, sc_data.astype(np.float32))
        
        return True, fname
    
    except Exception as e:
        logger.error(f"[ERROR] {fname}: {e}")
        return False, fname

def main():
    # 加载照明数据（只加载一次）
    ill_data = load_illumination_data()
    
    # 获取文件列表
    files_gt = [f for f in os.listdir(SRC_DIR_GT) if f.endswith(".npy")]
    logger.info(f"Found {len(files_gt)} GT patches.")
    
    # 准备参数
    gt_args = [(f, SRC_DIR_GT, TARGET_DIR, ill_data) for f in files_gt]
    
    num_workers = min(10, os.cpu_count() or 1)  # 不超过CPU核心数
    logger.info(f"Using {num_workers} processes.")
    
    # 处理所有文件
    success_count = 0
    fail_count = 0
    
    with Pool(num_workers) as pool:
        results = tqdm(
            pool.imap_unordered(process_one_file, gt_args),
            total=len(gt_args),
            desc="Processing files"
        )
        
        for success, fname in results:
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    logger.info(f"Processing completed. Success: {success_count}, Failed: {fail_count}")
    
    # 验证输出文件数量
    expected_count = len(gt_args) * len(ill_data)
    actual_count = len(os.listdir(TARGET_DIR))
    logger.info(f"Expected {expected_count} files, actual {actual_count} files")

if __name__ == "__main__":
    main()

# import numpy as np
# import os
# import shutil
# from multiprocessing import Pool
# from tqdm import tqdm
# import logging

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# SRC_DIR_GT = "/opt/data/private/ld/data/BJTU/gt"
# OUT_DIR = "/opt/data/private/ld/data/BJTU_E"

# TARGET_DIR = os.path.join(OUT_DIR, "gt")

# os.makedirs(TARGET_DIR, exist_ok=True)

# # 10个前缀列表
# PREFIXES = ['du', 'sd', 'cd', 'rd', 'sh', 'ye', 'pu', 're', 'bl', 'gr']

# def process_one_file_fast(args):
#     """快速处理单个文件：只复制和重命名"""
#     fname, src_dir, out_dir = args
#     try:
#         src_path = os.path.join(src_dir, fname)
        
#         # 检查文件是否存在
#         if not os.path.exists(src_path):
#             logger.error(f"File not found: {src_path}")
#             return False, fname
        
#         # 直接复制文件，不加载数据（最快的方法）
#         for prefix in PREFIXES:
#             output_name = f"{prefix}_{fname}"
#             output_path = os.path.join(out_dir, output_name)
            
#             # 使用shutil.copy2保留元数据
#             shutil.copy2(src_path, output_path)
        
#         return True, fname
    
#     except Exception as e:
#         logger.error(f"[ERROR] {fname}: {e}")
#         return False, fname

# def process_one_file_with_validation(args):
#     """带验证的处理：加载数据确保正确性"""
#     fname, src_dir, out_dir = args
#     try:
#         src_path = os.path.join(src_dir, fname)
        
#         # 加载原始数据
#         data_array = np.load(src_path)
        
#         # 验证数据
#         if np.isinf(data_array).any() or np.isnan(data_array).any():
#             logger.warning(f"{fname}: 原始数据包含inf/NaN，但将继续复制")
        
#         # 为每个前缀创建副本
#         for prefix in PREFIXES:
#             output_name = f"{prefix}_{fname}"
#             output_path = os.path.join(out_dir, output_name)
            
#             # 保存为float32（保持一致性）
#             np.save(output_path, data_array.astype(np.float32))
        
#         return True, fname
    
#     except Exception as e:
#         logger.error(f"[ERROR] {fname}: {e}")
#         return False, fname

# def process_one_file_smart(args):
#     """智能处理：如果数据没问题则直接复制，否则加载验证"""
#     fname, src_dir, out_dir = args
#     try:
#         src_path = os.path.join(src_dir, fname)
        
#         # 尝试直接复制（最快）
#         try:
#             for prefix in PREFIXES:
#                 output_name = f"{prefix}_{fname}"
#                 output_path = os.path.join(out_dir, output_name)
#                 shutil.copy2(src_path, output_path)
            
#             # 验证一个复制后的文件
#             test_prefix = PREFIXES[0]
#             test_path = os.path.join(out_dir, f"{test_prefix}_{fname}")
#             test_data = np.load(test_path)
            
#             # 快速检查
#             if np.isinf(test_data).any() or np.isnan(test_data).any():
#                 logger.warning(f"{fname}: 复制后数据有inf/NaN，重新处理")
#                 raise ValueError("Data has inf/NaN after copying")
            
#             return True, fname
            
#         except Exception as copy_error:
#             # 如果直接复制失败或有数据问题，使用加载验证的方法
#             logger.debug(f"{fname}: 直接复制失败，使用加载验证方法: {copy_error}")
            
#             data_array = np.load(src_path)
            
#             # 修复inf/NaN（如果存在）
#             if np.isinf(data_array).any() or np.isnan(data_array).any():
#                 logger.warning(f"{fname}: 修复数据中的inf/NaN")
#                 # 用0替换inf/NaN
#                 data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
            
#             for prefix in PREFIXES:
#                 output_name = f"{prefix}_{fname}"
#                 output_path = os.path.join(out_dir, output_name)
#                 np.save(output_path, data_array.astype(np.float32))
            
#             return True, fname
    
#     except Exception as e:
#         logger.error(f"[ERROR] {fname}: {e}")
#         return False, fname

# def main_fast():
#     """最快版本：直接复制文件"""
#     # 获取文件列表
#     files_gt = [f for f in os.listdir(SRC_DIR_GT) if f.endswith(".npy")]
#     logger.info(f"Found {len(files_gt)} GT patches.")
    
#     # 准备参数（只需要文件名和目录）
#     gt_args = [(f, SRC_DIR_GT, TARGET_DIR) for f in files_gt]
    
#     num_workers = min(10, os.cpu_count() or 1)
#     logger.info(f"Using {num_workers} processes.")
    
#     # 处理所有文件
#     success_count = 0
#     fail_count = 0
    
#     with Pool(num_workers) as pool:
#         results = tqdm(
#             pool.imap_unordered(process_one_file_fast, gt_args),
#             total=len(gt_args),
#             desc="Copying files"
#         )
        
#         for success, fname in results:
#             if success:
#                 success_count += 1
#             else:
#                 fail_count += 1
    
#     logger.info(f"Processing completed. Success: {success_count}, Failed: {fail_count}")
    
#     # 验证输出文件数量
#     expected_count = len(files_gt) * len(PREFIXES)
#     actual_count = len(os.listdir(TARGET_DIR))
#     logger.info(f"Expected {expected_count} files, actual {actual_count} files")
    
#     if expected_count == actual_count:
#         logger.info("✅ 文件数量匹配")
#     else:
#         logger.warning(f"⚠️ 文件数量不匹配: 期望 {expected_count}, 实际 {actual_count}")


# def batch_rename_direct():
#     """直接批量重命名（单线程，简单但可能更快）"""
#     files_gt = [f for f in os.listdir(SRC_DIR_GT) if f.endswith(".npy")]
#     logger.info(f"Found {len(files_gt)} GT patches.")
    
#     success_count = 0
    
#     for fname in tqdm(files_gt, desc="Copying files"):
#         src_path = os.path.join(SRC_DIR_GT, fname)
        
#         if not os.path.exists(src_path):
#             logger.error(f"File not found: {src_path}")
#             continue
        
#         try:
#             # 直接复制，不加载数据
#             for prefix in PREFIXES:
#                 output_name = f"{prefix}_{fname}"
#                 output_path = os.path.join(TARGET_DIR, output_name)
#                 shutil.copy2(src_path, output_path)
            
#             success_count += 1
            
#         except Exception as e:
#             logger.error(f"[ERROR] {fname}: {e}")
    
#     logger.info(f"Processing completed. Success: {success_count}/{len(files_gt)}")
    
#     # 验证
#     expected_count = len(files_gt) * len(PREFIXES)
#     actual_count = len(os.listdir(TARGET_DIR))
#     logger.info(f"Expected {expected_count} files, actual {actual_count} files")

# def main():
#     """主函数 - 选择最快的方法"""
    
#     # 方法1: 直接复制（最快）
#     logger.info("使用方法1: 直接文件复制（最快）")
#     main_fast()

# if __name__ == "__main__":
#     main()
