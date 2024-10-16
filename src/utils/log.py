
import torch
import numpy as np


def log_structure_and_shape(data, indent=0):
    if isinstance(data, dict):
        print("  " * indent + "{")
        for key, value in data.items():
            print("  " * (indent + 1) + f"'{key}':")
            log_structure_and_shape(value, indent + 2)
        print("  " * indent + "}")
    elif isinstance(data, (list, tuple)):
        print("  " * indent + f"{type(data).__name__}[")
        for item in data:
            log_structure_and_shape(item, indent + 1)
        print("  " * indent + "]")
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        print("  " * indent +
              f"{type(data).__name__}(shape={data.shape}, dtype={data.dtype})")
    else:
        print("  " * indent + f"{type(data).__name__}({data})")
