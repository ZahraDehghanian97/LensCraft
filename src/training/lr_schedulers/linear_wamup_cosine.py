import math
import warnings
from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-7,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        self.base_lrs_after_warmup = []
        for group in optimizer.param_groups:
            self.base_lrs_after_warmup.append(group['lr'])
            
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)
            
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs_after_warmup]
        
        else:
            epoch = epoch - self.warmup_epochs
            cosine_epochs = self.max_epochs - self.warmup_epochs
            
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * epoch / cosine_epochs))
                   for base_lr in self.base_lrs_after_warmup]
    
    def _get_closed_form_lr(self) -> List[float]:
        return self.get_lr()