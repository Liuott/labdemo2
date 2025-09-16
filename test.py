import torch
print(torch.__version__, torch.version.cuda)   # 有版本号说明是 CUDA 版
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
import os; print(os.getcwd())