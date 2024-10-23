
import torch
import numpy as np
import logging
from typing import Dict, Any


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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_memory_usage(location: str, additional_info: Dict[str, Any] = None):
    """Log CUDA memory usage at a specific location in the code."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        log_msg = f"Memory Usage at {location}:\n" \
            f"  Allocated: {allocated:.2f} GB\n" \
            f"  Reserved:  {reserved:.2f} GB\n" \
            f"  Max Allocated: {max_allocated:.2f} GB"

        if additional_info:
            log_msg += "\nAdditional Info:"
            for key, value in additional_info.items():
                if isinstance(value, torch.Tensor):
                    log_msg += f"\n  {key}: shape={tuple(value.shape)}, dtype={value.dtype}"
                else:
                    log_msg += f"\n  {key}: {value}"

        logger.info(log_msg)
