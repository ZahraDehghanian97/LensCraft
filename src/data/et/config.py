import torch

STANDARDIZATION_CONFIG = {
    "norm_mean": [7.93987673e-05, -9.98621393e-05, 4.12940653e-04],
    "norm_std": [0.027841, 0.01819818, 0.03138536],
    "shift_mean": [0.00201079, -0.27488501, -1.23616805],
    "shift_std": [1.13433516, 1.19061042, 1.58744263],
    "norm_mean_h": [6.676e-05, -5.084e-05, -7.782e-04],
    "norm_std_h": [0.0105, 0.006958, 0.01145],
    "velocity": True
}

STANDARDIZATION_CONFIG_TORCH = {
    "norm_mean": torch.tensor([7.93987673e-05, -9.98621393e-05, 4.12940653e-04]),
    "norm_std": torch.tensor([0.027841, 0.01819818, 0.03138536]),
    "shift_mean": torch.tensor([0.00201079, -0.27488501, -1.23616805]),
    "shift_std": torch.tensor([1.13433516, 1.19061042, 1.58744263]),
    "norm_mean_h": torch.tensor([6.676e-05, -5.084e-05, -7.782e-04]),
    "norm_std_h": torch.tensor([0.0105, 0.006958, 0.01145]),
    "velocity": True
}