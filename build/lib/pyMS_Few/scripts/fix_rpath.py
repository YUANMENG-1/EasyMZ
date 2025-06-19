import subprocess
import os
from pathlib import Path

def fix_rpath():
    script_dir = Path(__file__).parent
    binaries_dir = script_dir.parent / "binaries"
    so_file = binaries_dir / "hdbscan_cpp.cpython-39-darwin.so"

    if not so_file.exists():
        print(f"Error: {so_file} not found")
        return False

    target_rpaths = [
        "/Users/jianglab1/opt/miniconda3/lib",
        "/usr/local/Cellar/boost/1.87.0_1/lib"
    ]

    result = subprocess.run(
        ["otool", "-l", str(so_file)],
        capture_output=True, text=True, check=True
    )
    output = result.stdout

    rpaths = []
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if "cmd LC_RPATH" in line:
            path_line = lines[i + 2]
            if "path" in path_line:
                path = path_line.split("path ")[1].split(" (")[0]
                rpaths.append(path)

    rpath_count = {}
    for path in rpaths:
        rpath_count[path] = rpath_count.get(path, 0) + 1

    for path, count in rpath_count.items():
        if count > 1:
            for _ in range(count - 1):
                print(f"Removing duplicate LC_RPATH: {path}")
                subprocess.run(
                    ["install_name_tool", "-delete_rpath", path, str(so_file)],
                    check=True
                )

    for rpath in target_rpaths:
        if rpath not in rpaths:
            print(f"Adding RPATH: {rpath}")
            subprocess.run(
                ["install_name_tool", "-add_rpath", rpath, str(so_file)],
                check=True
            )

    result = subprocess.run(
        ["otool", "-l", str(so_file)],
        capture_output=True, text=True, check=True
    )
    print("LC_RPATH after fixing:")
    for line in result.stdout.splitlines():
        if "LC_RPATH" in line or "path " in line:
            print(line)

    return True

if __name__ == "__main__":
    if not fix_rpath():
        exit(1)
        