import subprocess
import platform
import os
import sys
from pathlib import Path
import argparse
import pkg_resources  # 导入 pkg_resources

def get_binary_path(binary_name, base_dir):
    system = platform.system().lower()
    arch = platform.machine()
    if system != "darwin":
        raise RuntimeError("This tool is only supported on macOS")
    if arch != "x86_64":
        raise RuntimeError("This tool is only supported on macOS Intel x86_64 architecture")
    binary_dir = Path(base_dir) / "pyMS_Few" / "binaries"
    return binary_dir / binary_name

def run_command(cmd, cwd=None):
    print(f"Executing command: {cmd}")
    if isinstance(cmd, list) and len(cmd) > 0:
        make_executable(cmd[0])
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False
        )
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0:
            print(f"Error: Command failed with return code {process.returncode}")
            sys.exit(process.returncode)
        return process
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

def make_executable(path):
    try:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | 0o755)
    except Exception as e:
        print(f"Warning: Could not set executable permission for {path}: {e}")

def is_hdbscan_output_valid(hdbscan_output):
    """检查 hdbscan_output 目录是否有效（存在且包含非空子文件夹）"""
    if not hdbscan_output.exists() or not hdbscan_output.is_dir():
        return False
    for subdir in hdbscan_output.iterdir():
        if subdir.is_dir():
            if any(subdir.iterdir()):
                return True
    return False

def main():
    # 动态获取版本号
    try:
        version = pkg_resources.get_distribution("pyMS_Few").version
    except pkg_resources.DistributionNotFound:
        version = "unknown"  # 如果未找到包，显示 unknown

    parser = argparse.ArgumentParser(description="pyMS_Few: Process mzML files on macOS Intel x86_64")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")  # 动态版本号
    parser.add_argument("input_dir", help="Path to input directory containing mzML files (e.g., /Volumes/ymy2025/tidumzml/posmzml/data_mix6)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)

    hdbscan_output = input_dir / "hdbscan_output"

    # Step 1: 03_batch_hdbscan.sh
    if is_hdbscan_output_valid(hdbscan_output):
        print(f"Skipping Step 1: hdbscan_output exists and contains non-empty subdirectories at {hdbscan_output}")
    else:
        print("Running Step 1: 03_batch_hdbscan.sh")
        run_command([str(get_binary_path("03_batch_hdbscan.sh", base_dir)), str(input_dir)])

    # Step 2: 04_bin_rename_and_move.py
    print("Running Step 2: 04_bin_rename_and_move.py")
    run_command([
        sys.executable, str(base_dir / "pyMS_Few" / "scripts" / "04_bin_rename_and_move.py"),
        "--input_dir", str(hdbscan_output),
        "--sample_group", str(input_dir / "sample_group.csv")
    ])

    # Step 3: 05_batch_metrics_filter.sh
    print("Running Step 3: 05_batch_metrics_filter.sh")
    run_command([str(get_binary_path("05_batch_metrics_filter.sh", base_dir)), str(hdbscan_output)])

    # Step 4: 06_generate_cluster_info
    print("Running Step 4: 06_generate_cluster_info")
    run_command([str(get_binary_path("06_generate_cluster_info_macos_x86_64", base_dir)), str(hdbscan_output)])

    # Step 5: 07_align_fill_check_group_deduplicate
    print("Running Step 5: 07_align_fill_check_group_deduplicate")
    run_command([str(get_binary_path("07_align_fill_check_group_deduplicate_macos_x86_64", base_dir)), str(hdbscan_output)])

    # Step 6: 08_group_na_fill
    print("Running Step 6: 08_group_na_fill")
    run_command([str(get_binary_path("08_group_na_fill_macos_x86_64", base_dir)), str(hdbscan_output)])

    # Step 7: 09_filter_metrics
    print("Running Step 7: 09_filter_metrics")
    run_command([str(get_binary_path("09_filter_metrics_macos_x86_64", base_dir)), str(hdbscan_output)])

    # Step 8: 11_filter_metrics_all
    print("Running Step 8: 11_filter_metrics_all")
    run_command([str(get_binary_path("11_filter_metrics_all_macos_x86_64", base_dir)), str(hdbscan_output)])

    # Step 9: 10_metrics_plot.py
    print("Running Step 9: 10_metrics_plot.py")
    run_command([sys.executable, str(base_dir / "pyMS_Few" / "scripts" / "10_metrics_plot.py")])

    # Step 10: 11_filter_metrics_all
    print("Running Step 10: 11_filter_metrics_all")
    run_command([str(get_binary_path("11_filter_metrics_all_macos_x86_64", base_dir)), str(hdbscan_output)])

if __name__ == "__main__":
    main()