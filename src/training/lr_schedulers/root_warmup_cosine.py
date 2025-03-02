import math
from math import pi, cos

class RootWarmupCosineDecayLR:
    def __init__(self, optimizer, warmup_steps, total_steps, rate, n_th_root, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr # DEFAULT: [0.0001]
        self.total_steps = total_steps
        self.rate = rate
        self.n_th_root = n_th_root
        print("LR: ", self.base_lr)

    def get_lr(self, lr, step):
        if step < self.warmup_steps:
            return lr * math.pow(step, 1 / self.n_th_root) / math.pow(self.warmup_steps, 1 / self.n_th_root)
        else:
            # fmt: off
            return (0.5 * lr * (1 + cos(self.rate * pi * (step - self.warmup_steps)
                    / (self.total_steps - self.warmup_steps))))
            # fmt: on

    def step(self, step):
        if self.base_lr is None:
            self.base_lr = [param_group["lr"] for param_group in self.optimizer.param_groups]
        for param_group, base_lr_group in zip(self.optimizer.param_groups, self.base_lr):
            param_group["lr"] = self.get_lr(base_lr_group, step)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
