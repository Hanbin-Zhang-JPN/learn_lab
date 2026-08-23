"""第 0 课：只检查环境，不修改任何东西。"""

import platform
import sys

import torch


print("Python:", sys.version.split()[0])
print("芯片:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS 可用:", torch.backends.mps.is_available())

if platform.machine() == "arm64" and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

x = torch.tensor([1.0, 2.0, 3.0], device=device)
print("实际计算设备:", x.device)
print("1² + 2² + 3² =", (x * x).sum().item())
print("\n准备完成，可以从第 1 课开始。")

