import os
import shutil
import pandas as pd
import argparse
from pathlib import Path  # 添加导入

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="根据 sample_group.csv 重命名并移动 .bin 文件，并删除空文件夹")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录路径，包含样本文件夹和 .bin 文件"
    )
    parser.add_argument(
        "--sample_group",
        type=str,
        default="sample_group.csv",
        help="sample_group.csv 文件路径（默认在 input_dir 查找）"
    )
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()

    # 输入目录
    root_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(root_dir):
        print(f"错误: {root_dir} 不是有效的目录")
        exit(1)

    # 确定 sample_group.csv 路径
    sample_group_file = Path(args.sample_group)
    if not sample_group_file.exists():
        # 尝试在 input_dir 查找
        sample_group_file = Path(root_dir).parent / "sample_group.csv"
        if not sample_group_file.exists():
            print(f"错误: sample_group.csv 不存在于 {args.sample_group} 或 {Path(root_dir).parent}")
            exit(1)
    print(f"读取 sample_group.csv: {sample_group_file}")

    # 加载 CSV 文件
    df = pd.read_csv(sample_group_file)
    # 创建 sample_name 到 group 的映射
    if 'sample_name' not in df.columns or 'group' not in df.columns:
        print("错误: sample_group.csv 必须包含 'sample_name' 和 'group' 列")
        exit(1)
    sample_to_group = dict(zip(df['sample_name'], df['group']))

    # 存储重命名后的文件路径
    renamed_files = {}

    # 第一步：遍历文件夹，重命名 .bin 文件
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        # 只处理文件夹
        if os.path.isdir(entry_path):
            # 查找文件夹中的 .bin 文件
            for file in os.listdir(entry_path):
                if file.endswith(".bin"):
                    source_path = os.path.join(entry_path, file)
                    new_filename = f"{entry}.bin"
                    dest_path = os.path.join(entry_path, new_filename)
                    print(f"重命名: {source_path} → {dest_path}")
                    shutil.move(source_path, dest_path)
                    renamed_files[entry] = dest_path
                    break  # 每个文件夹只处理一个 .bin 文件

    # 第二步：按 group 移动文件
    # 获取唯一的 group
    groups = set(sample_to_group.values())
    for group in groups:
        # 创建以 group 命名的目标文件夹
        group_dir = os.path.join(root_dir, group)
        os.makedirs(group_dir, exist_ok=True)
        print(f"创建目标文件夹: {group_dir}")

        # 找到属于该 group 的 sample_name
        samples = [s for s, g in sample_to_group.items() if g == group]
        for sample in samples:
            if sample in renamed_files:
                source_path = renamed_files[sample]
                dest_path = os.path.join(group_dir, os.path.basename(source_path))
                print(f"移动: {source_path} → {dest_path}")
                shutil.move(source_path, dest_path)
            else:
                print(f"警告: {sample} 的 .bin 文件未找到")

    # 第三步：删除空文件夹
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            # 检查文件夹是否为空
            if not os.listdir(entry_path):  # 空文件夹
                print(f"删除空文件夹: {entry_path}")
                shutil.rmtree(entry_path)
            else:
                print(f"文件夹 {entry_path} 不为空，保留")

    print("处理完成")

if __name__ == "__main__":
    main()
    