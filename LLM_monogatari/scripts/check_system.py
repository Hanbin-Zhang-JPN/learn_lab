#!/usr/bin/env python3
"""给新手看的只读环境检查，不安装也不修改系统。"""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]


def line(ok: bool, label: str, detail: str) -> None:
    mark = "通过" if ok else "注意"
    print(f"[{mark}] {label}: {detail}")


def memory_gib() -> float | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
        )
        return int(output.strip()) / (1024**3)
    except (OSError, ValueError, subprocess.SubprocessError):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024**3)
        except (OSError, ValueError):
            return None


def main() -> None:
    is_macos = platform.system() == "Darwin"
    is_arm = platform.machine() == "arm64"
    line(is_macos, "操作系统", f"{platform.system()} {platform.release()}")
    line(is_arm, "处理器架构", platform.machine())
    line(sys.version_info >= (3, 11), "Python", platform.python_version())
    executable = Path(sys.executable).absolute()
    line(executable.is_relative_to(PROJECT_DIR / ".venv"), "隔离环境", sys.executable)

    memory = memory_gib()
    line(memory is not None and memory >= 15.0, "内存", f"约 {memory:.1f} GiB" if memory else "未知")
    free = shutil.disk_usage(PROJECT_DIR).free / (1024**3)
    line(free >= 8.0, "剩余磁盘", f"约 {free:.1f} GiB（训练前建议至少 8 GiB）")

    cloudflared = shutil.which("cloudflared")
    line(cloudflared is not None, "分享工具（可选）", cloudflared or "未安装，不影响前 9 课")
    mlx_installed = importlib.util.find_spec("mlx_lm") is not None
    line(mlx_installed, "MLX-LM（第 8 课才需要）", "已安装" if mlx_installed else "尚未安装")

    if not (is_macos and is_arm):
        print("\n基础课程仍可运行，但最终 MLX 训练路线专为 Apple Silicon macOS 准备。")


if __name__ == "__main__":
    main()
