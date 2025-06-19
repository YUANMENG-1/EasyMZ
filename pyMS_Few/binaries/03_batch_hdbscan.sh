#!/bin/bash

# 记录开始时间
start_time=$(date +%s)
echo "开始处理时间: $(date)"

# 获取脚本所在目录（binaries/）
script_dir="$(dirname "$0")"
# 获取模块根目录（site-packages/pyMS_Few/）
base_dir="$(realpath "$script_dir/..")"
# 拼接 fix_rpath.py 和 fenqu.py 路径
fix_rpath_script="$base_dir/scripts/fix_rpath.py"
fenqu_script="$base_dir/scripts/fenqu.py"

# 检查 fix_rpath.py 是否存在
if [ ! -f "$fix_rpath_script" ]; then
    echo "错误: $fix_rpath_script 不存在"
    exit 1
fi

# 检查 fenqu.py 是否存在
if [ ! -f "$fenqu_script" ]; then
    echo "错误: $fenqu_script 不存在"
    exit 1
fi

# 运行 fix_rpath.py
echo "运行 fix_rpath.py 以修复动态库路径"
python "$fix_rpath_script"
if [ $? -ne 0 ]; then
    echo "错误: fix_rpath.py 执行失败"
    exit 1
fi

# 获取输入文件夹路径
if [ $# -eq 0 ]; then
    echo -n "请输入存放 mzML 文件的文件夹路径: "
    read -r input_dir
elif [ $# -eq 1 ]; then
    input_dir="$1"
else
    echo "用法: $0 [mzml_folder]"
    exit 1
fi

# 检查输入文件夹是否存在
if [ ! -d "$input_dir" ]; then
    echo "错误: 文件夹 $input_dir 不存在"
    exit 1
fi

# 切换到输入文件夹
cd "$input_dir" || { echo "错误: 无法切换到指定目录 $input_dir"; exit 1; }

# 创建输出目录 hdbscan_output
output_dir="hdbscan_output"
mkdir -p "$output_dir" || { echo "错误: 无法创建输出目录 $output_dir"; exit 1; }

# 检查是否有 .mzML 文件
shopt -s nullglob
mzml_files=(*.mzML *.mzml)
if [ ${#mzml_files[@]} -eq 0 ]; then
    echo "警告: 没有找到 .mzML 文件"
    exit 1
fi

# 计数器
total_files=0
processed_files=0

# 统计总文件数
for file in "${mzml_files[@]}"; do
    ((total_files++))
done

echo "找到 $total_files 个 mzML 文件，开始处理..."

# 处理每个 mzML 文件
for file in "${mzml_files[@]}"; do
    if [ -f "$file" ]; then
        echo "处理文件: $file"
        # 构造完整输入文件路径
        input_file="$input_dir/$file"
        # 调用 fenqu.py
        python "$fenqu_script" "$input_file" "$output_dir" --num_partitions 8 --max_workers 8 --min_intensity 250 --output_format bin
        if [ $? -eq 0 ]; then
            ((processed_files++))
            echo "成功处理: $file"
        else
            echo "处理失败: $file"
        fi
    fi
done

# 记录结束时间并计算总耗时
end_time=$(date +%s)
total_seconds=$((end_time - start_time))
minutes=$((total_seconds / 60))
seconds=$((total_seconds % 60))

echo "所有文件处理完成"
echo "总耗时: ${minutes} 分钟 ${seconds} 秒"