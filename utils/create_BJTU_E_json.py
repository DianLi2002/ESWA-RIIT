import json
import os

# 配置
input_file = "real_light_split.json"  # 原始JSON文件路径
output_file = "real_light_split_bjtu_e.json"  # 输出JSON文件路径
prefixes = ['du', 'sd', 'cd', 'rd', 'sh', 'ye', 'pu', 're', 'bl', 'gr']

def load_json_file(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, filepath):
    """保存数据为JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"文件已保存: {filepath}")

def add_prefixes_to_dataset(original_data, prefixes):
    """为数据集中的所有文件名添加前缀"""
    new_data = {}
    
    for split_name, file_list in original_data.items():
        new_file_list = []
        
        for filename in file_list:
            # 为每个文件添加所有前缀
            for prefix in prefixes:
                new_filename = f"{prefix}_{filename}"
                new_file_list.append(new_filename)
        
        new_data[split_name] = new_file_list
    
    return new_data

def main():
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return
    
    # 加载原始数据
    print(f"正在加载原始数据: {input_file}")
    original_data = load_json_file(input_file)
    
    # 计算统计数据
    original_count = sum(len(file_list) for file_list in original_data.values())
    print(f"原始数据集统计:")
    for split_name, file_list in original_data.items():
        print(f"  {split_name}: {len(file_list)} 个文件")
    print(f"总计: {original_count} 个文件")
    
    # 生成新数据
    print(f"\n正在为每个文件添加 {len(prefixes)} 个前缀: {prefixes}")
    new_data = add_prefixes_to_dataset(original_data, prefixes)
    
    # 计算新数据的统计数据
    new_count = sum(len(file_list) for file_list in new_data.values())
    print(f"新数据集统计:")
    for split_name, file_list in new_data.items():
        print(f"  {split_name}: {len(file_list)} 个文件")
    print(f"总计: {new_count} 个文件")
    print(f"扩展比例: {original_count} × {len(prefixes)} = {new_count}")
    
    # 保存新数据
    save_json_file(new_data, output_file)
    
    # 显示示例
    print(f"\n文件名格式示例:")
    for split_name in original_data.keys():
        original_sample = original_data[split_name][0]
        new_samples = new_data[split_name][:3]  # 前3个新文件名
        print(f"{split_name}:")
        print(f"  原始: {original_sample}")
        print(f"  新文件: {new_samples}")
        if len(new_data[split_name]) > 3:
            print(f"        ... 和 {len(new_data[split_name]) - 3} 个更多文件")
        print()

if __name__ == "__main__":
    main()