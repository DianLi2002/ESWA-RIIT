import os
import shutil
import json

# ==================== 配置路径 ====================
scene_path = '/opt/data/private/ld/data/BJTU_resample/rw'
gt_path = '/opt/data/private/ld/data/BJTU_resample/gt'
split_json = 'real_light_split.json'
output_root = '/opt/data/private/ld/data/BJTU_resample/split_data'

# ==================== 辅助函数 ====================
def mat_to_npy(filename):
    """将 .mat 扩展名替换为 .npy"""
    if filename.endswith('.mat'):
        return filename[:-4] + '.npy'
    return filename

# ==================== 读取划分 ====================
with open(split_json, 'r') as f:
    splits = json.load(f)   # {"train": ["rw_001.mat", ...], "val": [...], "test": [...]}

os.makedirs(output_root, exist_ok=True)

# ==================== 复制函数 ====================
def copy_split_files(split_name, file_list):
    split_dir = os.path.join(output_root, split_name)
    input_dir = os.path.join(split_dir, 'input')
    gt_dir = os.path.join(split_dir, 'gt')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for fname in file_list:
        # 将 .mat 转换为 .npy
        npy_fname = mat_to_npy(fname)
        
        # 输入文件
        src_input = os.path.join(scene_path, npy_fname)
        if not os.path.exists(src_input):
            print(f"警告：输入文件不存在 {src_input}")
            continue
        dst_input = os.path.join(input_dir, npy_fname)
        
        # 生成 GT 文件名：rw_xxx.mat -> rw_xxx -> img_id = xxx -> gtRef_xxx.npy
        # 注意：原始 fname 可能是 rw_001.mat，转换为 npy_fname = rw_001.npy
        base_name = npy_fname[:-4]          # 去掉 .npy
        img_id = base_name[3:]              # 去掉 'rw_'
        gt_npy_fname = f'gtRef_{img_id}.npy'
        src_gt = os.path.join(gt_path, gt_npy_fname)
        if not os.path.exists(src_gt):
            print(f"警告：GT 文件不存在 {src_gt}")
            continue
        dst_gt = os.path.join(gt_dir, gt_npy_fname)
        
        shutil.copy2(src_input, dst_input)
        shutil.copy2(src_gt, dst_gt)
    
    print(f"[{split_name}] 完成，共处理 {len(file_list)} 个样本。")

# ==================== 执行 ====================
for split_name, file_list in splits.items():
    if split_name not in ['train', 'val', 'test']:
        print(f"跳过未知 split: {split_name}")
        continue
    print(f"开始处理 {split_name} 集...")
    copy_split_files(split_name, file_list)

print("所有数据划分完成！")
for sp in ['train', 'val', 'test']:
    sp_dir = os.path.join(output_root, sp)
    if os.path.exists(sp_dir):
        input_cnt = len(os.listdir(os.path.join(sp_dir, 'input'))) if os.path.exists(os.path.join(sp_dir, 'input')) else 0
        gt_cnt = len(os.listdir(os.path.join(sp_dir, 'gt'))) if os.path.exists(os.path.join(sp_dir, 'gt')) else 0
        print(f"  {sp}/: input {input_cnt} 文件, gt {gt_cnt} 文件")