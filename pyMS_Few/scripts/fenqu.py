import argparse
import os
import sys
from pathlib import Path
from pyopenms import MSExperiment, MzXMLFile, MzMLFile, MzDataFile
import numpy as np
import threading

# 添加 binaries/ 路径
binaries_dir = Path(__file__).parent.parent / "binaries"
sys.path.append(str(binaries_dir))

import hdbscan_cpp

def readms(file_path):
    """读取 mzML/mzXML/mzData 文件，返回 (m/z, intensity, RT)"""
    ms_format = file_path.lower().split('.')[-1]
    msdata = MSExperiment()
    
    if ms_format == 'mzxml':
        file = MzXMLFile()
    elif ms_format == 'mzml':
        file = MzMLFile()
    elif ms_format == 'mzdata':
        file = MzDataFile()
    else:
        raise Exception(f"ERROR: Unsupported file format {file_path}")
    
    file.load(file_path, msdata)

    mz_list, intensity_list, rt_list = [], [], []

    for spectrum in msdata:
        if spectrum.getMSLevel() == 1:
            rt = spectrum.getRT()
            mz_values, intensity_values = [], []

            for peak in spectrum:
                if peak.getIntensity() != 0:
                    mz_values.append(peak.getMZ())
                    intensity_values.append(peak.getIntensity())

            if mz_values:
                mz_values = np.array(mz_values, dtype=np.float64)
                intensity_values = np.array(intensity_values, dtype=np.float64)
                rt_list.append(np.full(len(mz_values), rt, dtype=np.float64))
                mz_list.append(mz_values)
                intensity_list.append(intensity_values)

    mz_values = np.concatenate(mz_list) if mz_list else np.array([])
    intensity_values = np.concatenate(intensity_list) if intensity_list else np.array([])
    rt_values = np.concatenate(rt_list) if rt_list else np.array([])

    print(f"Read {len(mz_values)} peaks from {file_path}")
    print(f"Total m/z values: {len(mz_values)}")
    print(f"Total RT values (scans): {len(np.unique(rt_values))}")
    return mz_values, intensity_values, rt_values

def run_hpic(file_in, file_out, mz_values, intensity_values, rt_values, min_intensity, 
             target_num_partitions, max_workers, output_format):
    """封装 hpic 调用"""
    print(f"Starting hpic call with {max_workers} workers...")
    hdbscan_cpp.hpic(file_in, file_out, mz_values, intensity_values, rt_values, 
                     min_intensity, target_num_partitions, max_workers, output_format)
    print("hpic call completed.")

def main():
    parser = argparse.ArgumentParser(description="提取液相色谱-质谱数据集的纯离子色谱图")
    parser.add_argument('input_file', type=str, help="输入的 mzML/mzXML/mzData 文件路径")
    parser.add_argument('output_folder', type=str, help="存储结果的文件夹")
    parser.add_argument('--min_intensity', type=float, default=250.0, help="峰的最小强度（默认：250.0）")
    parser.add_argument('--num_partitions', type=int, default=8, help="目标分区数（默认：8）")
    parser.add_argument('--max_workers', type=int, default=8, help="工作进程数量（默认：8）")
    parser.add_argument('--output_format', type=str, choices=['csv', 'bin'], default='bin', help="输出格式：csv 或 bin（默认：bin）")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return

    try:
        mz_values, intensity_values, rt_values = readms(args.input_file)
        thread = threading.Thread(target=run_hpic, args=(
            args.input_file, args.output_folder, mz_values.tolist(), intensity_values.tolist(),
            rt_values.tolist(), args.min_intensity, args.num_partitions, args.max_workers,
            args.output_format))
        thread.daemon = True
        thread.start()
        thread.join(timeout=7200)
        if thread.is_alive():
            print("Error: HDBSCAN clustering timed out after 7200 seconds")
            return
        print(f"Clustering results saved to: {args.output_folder}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    main()
'''
import argparse
import os
import json
from pyopenms import MSExperiment, MzXMLFile, MzMLFile, MzDataFile
import numpy as np
import hdbscan_cpp
import threading

def readms(file_path):
    """读取 mzML/mzXML/mzData 文件，返回 (m/z, intensity, RT)"""
    ms_format = file_path.lower().split('.')[-1]
    msdata = MSExperiment()
    
    if ms_format == 'mzxml':
        file = MzXMLFile()
    elif ms_format == 'mzml':
        file = MzMLFile()
    elif ms_format == 'mzdata':
        file = MzDataFile()
    else:
        raise Exception(f"ERROR: Unsupported file format {file_path}")
    
    file.load(file_path, msdata)

    mz_list, intensity_list, rt_list = [], [], []

    for spectrum in msdata:
        if spectrum.getMSLevel() == 1:
            rt = spectrum.getRT()
            mz_values, intensity_values = [], []

            for peak in spectrum:
                if peak.getIntensity() != 0:
                    mz_values.append(peak.getMZ())
                    intensity_values.append(peak.getIntensity())

            if mz_values:
                mz_values = np.array(mz_values, dtype=np.float64)
                intensity_values = np.array(intensity_values, dtype=np.float64)
                rt_list.append(np.full(len(mz_values), rt, dtype=np.float64))
                mz_list.append(mz_values)
                intensity_list.append(intensity_values)

    mz_values = np.concatenate(mz_list) if mz_list else np.array([])
    intensity_values = np.concatenate(intensity_list) if intensity_list else np.array([])
    rt_values = np.concatenate(rt_list) if rt_list else np.array([])

    print(f"Read {len(mz_values)} peaks from {file_path}")
    print(f"Total m/z values: {len(mz_values)}")
    print(f"Total RT values (scans): {len(np.unique(rt_values))}")
    return mz_values, intensity_values, rt_values

def compute_mz_differences(mz_values):
    """计算相邻的 m/z 差值，并返回对应的 m/z 值和差值"""
    mz_sorted = np.sort(mz_values)
    mz_diffs = np.diff(mz_sorted)
    print(f"m/z values for differences: {len(mz_sorted)-1}, m/z diffs: {len(mz_diffs)}")
    return mz_sorted, mz_diffs

def compute_rt_mean_interval(rt_values):
    """计算 RT 均值间隔"""
    rt_unique = np.sort(np.unique(rt_values))
    if len(rt_unique) < 2:
        print("Warning: Insufficient unique RT values, returning 1.0")
        return 1.0
    rt_diffs = np.diff(rt_unique)
    rt_mean_interval = np.mean(rt_diffs)
    print(f"Average RT interval: {rt_mean_interval:.6f} seconds")
    return rt_mean_interval

def calculate_initial_thresholds(mz_values, mz_diffs, num_partitions):
    """初始分区域计算 m/z 差值的 99% 阈值，基于点数均分"""
    mz_sorted = np.sort(mz_values)
    total_points = len(mz_values)
    points_per_partition = total_points // num_partitions
    bin_edges = [mz_sorted[0]]
    for i in range(1, num_partitions):
        bin_edges.append(mz_sorted[i * points_per_partition])
    bin_edges.append(mz_sorted[-1])
    mz_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_partitions)]
    thresholds = []
    point_counts = []

    for i in range(num_partitions):
        mask = (mz_values >= bin_edges[i]) & (mz_values < bin_edges[i + 1])
        region_mz = mz_values[mask]
        point_counts.append(len(region_mz))
        if len(region_mz) > 1:
            region_diffs = np.diff(np.sort(region_mz))
            if len(region_diffs) > 0:
                threshold = np.percentile(region_diffs, 99)
                thresholds.append(threshold)
            else:
                thresholds.append(0.0)
        else:
            thresholds.append(0.0)

    print("Initial partition point counts:")
    for i, count in enumerate(point_counts):
        print(f"Partition {i}: {count} points (m/z range: {bin_edges[i]:.4f} to {bin_edges[i+1]:.4f})")
    print("m/z midpoints:", [f"{m:.2f}" for m in mz_midpoints])

    return bin_edges, np.array(thresholds), mz_midpoints

def find_partition_points(mz_values, mz_diffs, target_num_partitions, bin_edges, thresholds):
    """基于第一次分区的阈值选择跳跃点进行第二次划分"""
    mz_sorted = np.sort(mz_values)
    total_points = len(mz_values)
    points_per_partition = total_points // target_num_partitions

    # 为每个 m/z 分配对应分区的阈值
    threshold_map = np.zeros(len(mz_diffs))
    for i, mz in enumerate(mz_sorted[:-1]):
        region_idx = np.searchsorted(bin_edges, mz, side='right') - 1
        if region_idx >= len(thresholds):
            region_idx = len(thresholds) - 1
        threshold_map[i] = thresholds[region_idx]

    # 寻找大于阈值的跳跃点
    large_jumps = np.where(mz_diffs > threshold_map)[0]
    partition_points = [float(mz_sorted[idx] + mz_diffs[idx] * 0.5) for idx in large_jumps]
    partition_points = sorted(partition_points)
    print(f"Found {len(partition_points)} large jumps")

    # 选择 n-1 个断点
    selected_points = []
    if len(partition_points) >= target_num_partitions - 1:
        # 按点数均分选择跳跃点
        jump_indices = np.linspace(0, len(partition_points) - 1, target_num_partitions - 1, dtype=int)
        selected_points = [partition_points[int(i)] for i in jump_indices]
    else:
        # 跳跃点不足，从第一次分区的 bin_edges 补充
        needed = target_num_partitions - 1 - len(partition_points)
        available_edges = bin_edges[1:-1]  # 排除最小和最大边界
        # 优先选择靠近点数均分的边界点
        sorted_indices = np.argsort(mz_values)
        step = total_points // (target_num_partitions + 1)
        additional_points = [mz_values[sorted_indices[i * step]] for i in range(1, needed + 1)]
        selected_points = sorted(partition_points + additional_points)[:target_num_partitions - 1]

    # 输出分区点数
    bins = [-np.inf] + selected_points + [np.inf]
    point_counts = np.histogram(mz_values, bins=bins)[0]
    print("Final partition point counts:")
    for i, count in enumerate(point_counts):
        print(f"Partition {i}: {count} points (m/z range: {bins[i]:.4f} to {bins[i+1]:.4f})")
    print(f"Selected {len(selected_points)} partition points: {selected_points}")

    return selected_points

def calculate_new_thresholds(mz_values, partition_points):
    """根据新分区重新计算 m/z 差值 99% 阈值"""
    bin_edges = [np.min(mz_values)] + partition_points + [np.max(mz_values)]
    thresholds = []
    mz_sorted = np.sort(mz_values)
    mz_diffs = np.diff(mz_sorted)

    for i in range(len(bin_edges) - 1):
        mask = (mz_sorted >= bin_edges[i]) & (mz_sorted < bin_edges[i + 1])
        region_mz = mz_sorted[mask]
        if len(region_mz) > 1:
            region_diffs = np.diff(region_mz)
            if len(region_diffs) > 0:
                threshold = np.percentile(region_diffs, 99)
                thresholds.append(threshold)
            else:
                thresholds.append(0.0)
        else:
            thresholds.append(0.0)

    return bin_edges, np.array(thresholds)

def estimate_hdbscan_params(bin_edges, thresholds, rt_mean_interval, target_num_partitions):
    """估算每个分区的 HDBSCAN 参数"""
    k = 3.0
    rt_density = 1.0 / rt_mean_interval if rt_mean_interval > 0 else 1.0
    target_rt_points = 100.0
    hdbscan_params = []

    for i in range(target_num_partitions):
        thresh = thresholds[i] if i < len(thresholds) else 0.0
        mz_mid = (bin_edges[i] + bin_edges[i + 1]) / 2 if i < len(bin_edges) - 1 else bin_edges[i]
        if thresh == 0.0:
            hdbscan_params.append({
                "mz_midpoint": float(mz_mid),
                "mz_threshold": 0.0,
                "mz_range_half": 0.0,
                "rt_range_half": float(round(target_rt_points / (2 * rt_density))),
                "estimated_points": 0.0
            })
            continue

        mz_range_half = k * thresh
        mz_density = 1.0 / thresh if thresh > 0 else 1.0
        mz_points = 2 * mz_range_half * mz_density
        rt_range_half = target_rt_points / (2 * rt_density)
        rt_range_half_rounded = round(rt_range_half)
        estimated_points = (2 * mz_range_half * mz_density) * (2 * rt_range_half_rounded * rt_density)

        hdbscan_params.append({
            "mz_midpoint": float(mz_mid),
            "mz_threshold": float(thresh),
            "mz_range_half": float(mz_range_half),
            "rt_range_half": float(rt_range_half_rounded),
            "estimated_points": float(estimated_points)
        })

    print("HDBSCAN parameters:")
    for param in hdbscan_params:
        print(f"m/z = {param['mz_midpoint']:.2f}: threshold = {param['mz_threshold']:.6f}, "
              f"m/z range = ±{param['mz_range_half']:.4f} Da, "
              f"RT range = ±{param['rt_range_half']:.0f} s, "
              f"estimated points = {param['estimated_points']:.0f}")
    return bin_edges, hdbscan_params

def save_hdbscan_params(partition_points, hdbscan_params, output_file):
    """保存分区点和 HDBSCAN 参数到 JSON 文件"""
    params_to_save = {
        "partition_points": partition_points,
        "hdbscan_params": [
            {
                "mz_range_half": param["mz_range_half"],
                "rt_range_half": param["rt_range_half"]
            } for param in hdbscan_params
        ]
    }
    with open(output_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"HDBSCAN parameters and partition points saved to: {output_file}")

def run_hpic(file_in, file_out, mz_values, intensity_values, rt_values, min_intensity, 
             target_num_partitions, max_workers, output_format, params_file):
    """封装 hpic 调用"""
    print(f"Starting hpic call with {max_workers} workers...")
    hdbscan_cpp.hpic(file_in, file_out, mz_values, intensity_values, rt_values, 
                     min_intensity, target_num_partitions, max_workers, output_format, params_file)
    print("hpic call completed.")

def main():
    parser = argparse.ArgumentParser(description="提取液相色谱-质谱数据集的纯离子色谱图（带分区）")
    parser.add_argument('input_file', type=str, help="输入的 mzML/mzXML/mzData 文件路径")
    parser.add_argument('output_folder', type=str, help="存储结果的文件夹")
    parser.add_argument('--min_intensity', type=float, default=250.0, help="峰的最小强度（默认：250.0）")
    parser.add_argument('--num_partitions', type=int, default=60, help="目标分区数（默认：60）")
    parser.add_argument('--max_workers', type=int, default=8, help="工作进程数量（默认：8）")
    parser.add_argument('--output_format', type=str, choices=['csv', 'bin'], default='csv', help="输出格式：csv 或 bin（默认：csv）")
    parser.add_argument('--params_file', type=str, default="hdbscan_params.json", 
                        help="HDBSCAN 参数输出文件路径（默认：hdbscan_params.json）")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return

    try:
        mz_values, intensity_values, rt_values = readms(args.input_file)
        mz_sorted, mz_diffs = compute_mz_differences(mz_values)
        rt_mean_interval = compute_rt_mean_interval(rt_values)

        bin_edges, thresholds, mz_midpoints = calculate_initial_thresholds(mz_values, mz_diffs, args.num_partitions)
        print("Initial regional Critical Δm/z (99%):")
        for i, (mz_mid, thresh) in enumerate(zip(mz_midpoints, thresholds)):
            if thresh > 0:
                print(f"Partition {i} (m/z = {mz_mid:.2f}): {thresh:.6f}")

        partition_points = find_partition_points(mz_values, mz_diffs, args.num_partitions, bin_edges, thresholds)
        bin_edges, thresholds = calculate_new_thresholds(mz_values, partition_points)
        print("New regional Critical Δm/z (99%):")
        for i, thresh in enumerate(thresholds):
            if thresh > 0:
                print(f"Partition {i}: {thresh:.6f}")

        bin_edges, hdbscan_params = estimate_hdbscan_params(bin_edges, thresholds, rt_mean_interval, args.num_partitions)
        save_hdbscan_params(partition_points, hdbscan_params, args.params_file)

        thread = threading.Thread(target=run_hpic, args=(
            args.input_file, args.output_folder, mz_values.tolist(), intensity_values.tolist(),
            rt_values.tolist(), args.min_intensity, args.num_partitions, args.max_workers,
            args.output_format, args.params_file))
        thread.daemon = True
        thread.start()
        thread.join(timeout=7200)
        if thread.is_alive():
            print("Error: HDBSCAN clustering timed out after 7200 seconds")
            return
        print(f"Clustering results saved to: {args.output_folder}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    main()
'''
'''
import argparse
import os
import json
from pyopenms import MSExperiment, MzXMLFile, MzMLFile, MzDataFile
import numpy as np
import hdbscan_cpp
import threading

def readms(file_path):
    """读取 mzML/mzXML/mzData 文件，返回 (m/z, intensity, RT)"""
    ms_format = file_path.lower().split('.')[-1]
    msdata = MSExperiment()
    
    if ms_format == 'mzxml':
        file = MzXMLFile()
    elif ms_format == 'mzml':
        file = MzMLFile()
    elif ms_format == 'mzdata':
        file = MzDataFile()
    else:
        raise Exception(f"ERROR: Unsupported file format {file_path}")
    
    file.load(file_path, msdata)

    mz_list, intensity_list, rt_list = [], [], []

    for spectrum in msdata:
        if spectrum.getMSLevel() == 1:
            rt = spectrum.getRT()
            mz_values, intensity_values = [], []

            for peak in spectrum:
                if peak.getIntensity() != 0:
                    mz_values.append(peak.getMZ())
                    intensity_values.append(peak.getIntensity())

            if mz_values:
                mz_values = np.array(mz_values, dtype=np.float64)
                intensity_values = np.array(intensity_values, dtype=np.float64)
                rt_list.append(np.full(len(mz_values), rt, dtype=np.float64))
                mz_list.append(mz_values)
                intensity_list.append(intensity_values)

    mz_values = np.concatenate(mz_list) if mz_list else np.array([])
    intensity_values = np.concatenate(intensity_list) if intensity_list else np.array([])
    rt_values = np.concatenate(rt_list) if rt_list else np.array([])

    print(f"Read {len(mz_values)} peaks from {file_path}")
    print(f"Total m/z values: {len(mz_values)}")
    print(f"Total RT values (scans): {len(np.unique(rt_values))}")
    return mz_values, intensity_values, rt_values

def compute_mz_differences(mz_values):
    """计算相邻的 m/z 差值，并返回对应的 m/z 值和差值"""
    mz_sorted = np.sort(mz_values)
    mz_diffs = np.diff(mz_sorted)
    print(f"m/z values for differences: {len(mz_sorted)-1}, m/z diffs: {len(mz_diffs)}")
    return mz_sorted, mz_diffs

def compute_rt_mean_interval(rt_values):
    """计算 RT 均值间隔"""
    rt_unique = np.sort(np.unique(rt_values))
    if len(rt_unique) < 2:
        print("Warning: Insufficient unique RT values, returning 1.0")
        return 1.0
    rt_diffs = np.diff(rt_unique)
    rt_mean_interval = np.mean(rt_diffs)
    print(f"Average RT interval: {rt_mean_interval:.6f} seconds")
    return rt_mean_interval

def calculate_initial_thresholds(mz_values, mz_diffs, num_partitions):
    """初始分区域计算 m/z 差值的 99% 阈值，基于 m/z 均分"""
    mz_min, mz_max = np.min(mz_values), np.max(mz_values)
    bin_edges = np.linspace(mz_min, mz_max, num_partitions + 1)
    mz_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_partitions)]
    thresholds = []
    point_counts = []

    for i in range(num_partitions):
        mask = (mz_values >= bin_edges[i]) & (mz_values < bin_edges[i + 1])
        region_mz = mz_values[mask]
        point_counts.append(len(region_mz))
        if len(region_mz) > 1:
            region_diffs = np.diff(np.sort(region_mz))
            if len(region_diffs) > 0:
                threshold = np.percentile(region_diffs, 99)
                thresholds.append(threshold)
            else:
                thresholds.append(0.0)
        else:
            thresholds.append(0.0)

    print("Initial partition point counts:")
    for i, count in enumerate(point_counts):
        print(f"Partition {i}: {count} points (m/z range: {bin_edges[i]:.4f} to {bin_edges[i+1]:.4f})")
    print("m/z midpoints:", [f"{m:.2f}" for m in mz_midpoints])

    return bin_edges, np.array(thresholds), mz_midpoints

def find_partition_points(mz_values, mz_diffs, target_num_partitions, bin_edges, thresholds):
    """从大于阈值的跳跃点中选择 n-1 个断点，优化点数均分"""
    mz_sorted = np.sort(mz_values)
    total_points = len(mz_values)
    points_per_partition = total_points // target_num_partitions

    # 动态阈值分配
    threshold_map = np.zeros(len(mz_diffs))
    for i, mz in enumerate(mz_sorted[:-1]):
        region_idx = np.searchsorted(bin_edges, mz, side='right') - 1
        if region_idx >= len(thresholds):
            region_idx = len(thresholds) - 1
        threshold_map[i] = thresholds[region_idx]

    # 寻找大于阈值的跳跃点
    large_jumps = np.where(mz_diffs > threshold_map)[0]
    partition_points = [float(mz_sorted[idx]) for idx in large_jumps]
    partition_points = sorted(partition_points)

    # 选择 n-1 个断点，优化点数均分
    selected_points = []
    if len(partition_points) >= target_num_partitions - 1:
        # 贪心选择：逐步添加断点，保持点数接近均分
        target_count = total_points // target_num_partitions
        for _ in range(target_num_partitions - 1):
            best_point = None
            min_diff = float('inf')
            current_bins = [-np.inf] + selected_points + [np.inf]
            current_counts = np.histogram(mz_values, bins=current_bins)[0]

            for point in partition_points:
                if point in selected_points:
                    continue
                test_bins = [-np.inf] + sorted(selected_points + [point]) + [np.inf]
                test_counts = np.histogram(mz_values, bins=test_bins)[0]
                variance = np.var(test_counts)
                if variance < min_diff:
                    min_diff = variance
                    best_point = point

            if best_point:
                selected_points.append(best_point)
                selected_points = sorted(selected_points)
                partition_points.remove(best_point)  # 避免重复选择
    else:
        # 跳跃点不足，补充基于点数均分的断点
        sorted_indices = np.argsort(mz_values)
        needed = target_num_partitions - 1 - len(partition_points)
        step = total_points // (needed + 1)
        additional_points = [mz_values[sorted_indices[i * step]] for i in range(1, needed + 1)]
        selected_points = sorted(partition_points + additional_points)[:target_num_partitions - 1]

    # 输出分区点数
    bins = [-np.inf] + selected_points + [np.inf]
    point_counts = np.histogram(mz_values, bins=bins)[0]
    print("Final partition point counts:")
    for i, count in enumerate(point_counts):
        print(f"Partition {i}: {count} points (m/z range: {bins[i]:.4f} to {bins[i+1]:.4f})")
    print(f"Selected {len(selected_points)} partition points: {selected_points}")

    return selected_points

def calculate_new_thresholds(mz_values, partition_points):
    """根据新分区重新计算 m/z 差值 99% 阈值"""
    bin_edges = [np.min(mz_values)] + partition_points + [np.max(mz_values)]
    thresholds = []
    mz_sorted = np.sort(mz_values)
    mz_diffs = np.diff(mz_sorted)

    for i in range(len(bin_edges) - 1):
        mask = (mz_sorted >= bin_edges[i]) & (mz_sorted < bin_edges[i + 1])
        region_mz = mz_sorted[mask]
        if len(region_mz) > 1:
            region_diffs = np.diff(region_mz)
            if len(region_diffs) > 0:
                threshold = np.percentile(region_diffs, 99)
                thresholds.append(threshold)
            else:
                thresholds.append(0.0)
        else:
            thresholds.append(0.0)

    return bin_edges, np.array(thresholds)

def estimate_hdbscan_params(bin_edges, thresholds, rt_mean_interval, target_num_partitions):
    """估算每个分区的 HDBSCAN 参数"""
    k = 3.0
    rt_density = 1.0 / rt_mean_interval if rt_mean_interval > 0 else 1.0
    target_rt_points = 100.0
    hdbscan_params = []

    for i in range(target_num_partitions):
        thresh = thresholds[i] if i < len(thresholds) else 0.0
        mz_mid = (bin_edges[i] + bin_edges[i + 1]) / 2 if i < len(bin_edges) - 1 else bin_edges[i]
        if thresh == 0.0:
            hdbscan_params.append({
                "mz_midpoint": float(mz_mid),
                "mz_threshold": 0.0,
                "mz_range_half": 0.0,
                "rt_range_half": float(round(target_rt_points / (2 * rt_density))),
                "estimated_points": 0.0
            })
            continue

        mz_range_half = k * thresh
        mz_density = 1.0 / thresh if thresh > 0 else 1.0
        mz_points = 2 * mz_range_half * mz_density
        rt_range_half = target_rt_points / (2 * rt_density)
        rt_range_half_rounded = round(rt_range_half)
        estimated_points = (2 * mz_range_half * mz_density) * (2 * rt_range_half_rounded * rt_density)

        hdbscan_params.append({
            "mz_midpoint": float(mz_mid),
            "mz_threshold": float(thresh),
            "mz_range_half": float(mz_range_half),
            "rt_range_half": float(rt_range_half_rounded),
            "estimated_points": float(estimated_points)
        })

    print("HDBSCAN parameters:")
    for param in hdbscan_params:
        print(f"m/z = {param['mz_midpoint']:.2f}: threshold = {param['mz_threshold']:.6f}, "
              f"m/z range = ±{param['mz_range_half']:.4f} Da, "
              f"RT range = ±{param['rt_range_half']:.0f} s, "
              f"estimated points = {param['estimated_points']:.0f}")
    return bin_edges, hdbscan_params

def save_hdbscan_params(partition_points, hdbscan_params, output_file):
    """保存分区点和 HDBSCAN 参数到 JSON 文件"""
    params_to_save = {
        "partition_points": partition_points,
        "hdbscan_params": [
            {
                "mz_range_half": param["mz_range_half"],
                "rt_range_half": param["rt_range_half"]
            } for param in hdbscan_params
        ]
    }
    with open(output_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"HDBSCAN parameters and partition points saved to: {output_file}")

def run_hpic(file_in, file_out, mz_values, intensity_values, rt_values, min_intensity, 
             target_num_partitions, max_workers, output_format, params_file):
    """封装 hpic 调用"""
    print(f"Starting hpic call with {max_workers} workers...")
    hdbscan_cpp.hpic(file_in, file_out, mz_values, intensity_values, rt_values, 
                     min_intensity, target_num_partitions, max_workers, output_format, params_file)
    print("hpic call completed.")

def main():
    parser = argparse.ArgumentParser(description="提取液相色谱-质谱数据集的纯离子色谱图（带分区）")
    parser.add_argument('input_file', type=str, help="输入的 mzML/mzXML/mzData 文件路径")
    parser.add_argument('output_folder', type=str, help="存储结果的文件夹")
    parser.add_argument('--min_intensity', type=float, default=250.0, help="峰的最小强度（默认：250.0）")
    parser.add_argument('--num_partitions', type=int, default=60, help="目标分区数（默认：60）")
    parser.add_argument('--max_workers', type=int, default=8, help="工作进程数量（默认：8）")
    parser.add_argument('--output_format', type=str, choices=['csv', 'bin'], default='csv', help="输出格式：csv 或 bin（默认：csv）")
    parser.add_argument('--params_file', type=str, default="hdbscan_params.json", 
                        help="HDBSCAN 参数输出文件路径（默认：hdbscan_params.json）")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return

    try:
        mz_values, intensity_values, rt_values = readms(args.input_file)
        mz_sorted, mz_diffs = compute_mz_differences(mz_values)
        rt_mean_interval = compute_rt_mean_interval(rt_values)

        bin_edges, thresholds, mz_midpoints = calculate_initial_thresholds(mz_values, mz_diffs, args.num_partitions)
        print("Initial regional Critical Δm/z (99%):")
        for i, (mz_mid, thresh) in enumerate(zip(mz_midpoints, thresholds)):
            if thresh > 0:
                print(f"Partition {i} (m/z = {mz_mid:.2f}): {thresh:.6f}")

        partition_points = find_partition_points(mz_values, mz_diffs, args.num_partitions, bin_edges, thresholds)
        bin_edges, thresholds = calculate_new_thresholds(mz_values, partition_points)
        print("New regional Critical Δm/z (99%):")
        for i, thresh in enumerate(thresholds):
            if thresh > 0:
                print(f"Partition {i}: {thresh:.6f}")

        bin_edges, hdbscan_params = estimate_hdbscan_params(bin_edges, thresholds, rt_mean_interval, args.num_partitions)
        save_hdbscan_params(partition_points, hdbscan_params, args.params_file)

        thread = threading.Thread(target=run_hpic, args=(
            args.input_file, args.output_folder, mz_values.tolist(), intensity_values.tolist(),
            rt_values.tolist(), args.min_intensity, args.num_partitions, args.max_workers,
            args.output_format, args.params_file))
        thread.daemon = True
        thread.start()
        thread.join(timeout=7200)
        if thread.is_alive():
            print("Error: HDBSCAN clustering timed out after 7200 seconds")
            return
        print(f"Clustering results saved to: {args.output_folder}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    main()
'''