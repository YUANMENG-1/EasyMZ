#!/bin/bash

# 检查是否提供了目标目录作为参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <目标目录>"
    echo "示例: $0 /Volumes/ymy2025/tidumzml/posmzml/data_mix3/hdbscan_output"
    exit 1
fi

# 获取目标目录
BASE_DIR="$1"
# 获取脚本自身的目录
SCRIPT_DIR="$(dirname "$0")"
# 定义 calculate_metrics_all 的完整路径
CALC_METRICS="$SCRIPT_DIR/04_calculate_metrics_all"

# 检查目标目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误：目录 $BASE_DIR 不存在"
    exit 1
fi

# 检查 calculate_metrics_all 是否存在
if [ ! -f "$CALC_METRICS" ]; then
    echo "错误：可执行文件 $CALC_METRICS 不存在"
    exit 1
fi

# 确保 calculate_metrics_all 可执行
if [ ! -x "$CALC_METRICS" ]; then
    echo "设置 $CALC_METRICS 为可执行"
    chmod +x "$CALC_METRICS"
fi

# 遍历目标目录下的所有子文件夹
for dir in "$BASE_DIR"/*/ ; do
    if [ -d "$dir" ]; then
        echo "处理文件夹: $dir"
        # 进入子文件夹
        cd "$dir" || continue
        # 查找所有 .bin 文件并执行
        for bin_file in *.bin; do
            if [ -f "$bin_file" ]; then
                echo "处理文件: $bin_file"
                # 执行 calculate_metrics_all，使用完整路径，传入 .bin 文件的完整路径
                "$CALC_METRICS" "$dir$bin_file"
                if [ $? -ne 0 ]; then
                    echo "处理 $bin_file 失败"
                else
                    echo "处理 $bin_file 成功"
                fi
            fi
        done
        # 返回上级目录
        cd "$BASE_DIR" || exit 1
    fi
done

echo "所有文件处理完成"
