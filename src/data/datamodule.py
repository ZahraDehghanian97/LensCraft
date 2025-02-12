import os
import platform
import psutil
import hydra
import lightning as L
from torch.utils.data import random_split, DataLoader
from data.simulation.dataset import collate_fn
from data.et.dataset import collate_fn as et_collate_fn


class CameraTrajectoryDataModule(L.LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers=None, val_size=0.1, test_size=0.1):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.dataset_mode = 'et' if 'ETDataset' in dataset_config['_target_'] else 'simulation'
        self.collate_fn = et_collate_fn if self.dataset_mode == 'et' else collate_fn
        
        self.is_mac = platform.system() == 'Darwin'
        self.setup_platform_specific(num_workers)

    def setup_platform_specific(self, num_workers):
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        if num_workers is None:
            if self.is_mac:
                self.num_workers = min(cpu_count - 1, 4)
            else:
                self.num_workers = min(cpu_count - 1, 8)
        else:
            self.num_workers = num_workers

        if memory_gb < 4:
            self.num_workers = max(1, self.num_workers // 2)

        if self.is_mac:
            self.mp_context = 'fork'  # macOS performs better with fork
            self.persistent_workers = False  # Avoid memory issues on macOS
            self.prefetch_factor = 2  # Lower prefetch for better memory management
        else:
            self.mp_context = 'spawn'  # Linux performs better with spawn
            self.persistent_workers = True  # Good for Linux
            self.prefetch_factor = 4  # Higher prefetch for better throughput

    def setup(self, stage=None):
        full_dataset = hydra.utils.instantiate(self.dataset_config)
        
        if hasattr(full_dataset, 'preprocess'):
            full_dataset.preprocess()

        train_size = int((1 - self.val_size - self.test_size) * len(full_dataset))
        val_size = int(self.val_size * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def _get_common_dataloader_kwargs(self, shuffle=False):
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'collate_fn': self.collate_fn,
            'multiprocessing_context': self.mp_context,
            'pin_memory': True,  # Beneficial for both platforms when using GPU
            'shuffle': shuffle,
        }

    def train_dataloader(self):
        kwargs = self._get_common_dataloader_kwargs(shuffle=True)
        
        if not self.is_mac:
            kwargs.update({
                'persistent_workers': self.persistent_workers,
                'prefetch_factor': self.prefetch_factor
            })

        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = self._get_common_dataloader_kwargs()
        
        if not self.is_mac:
            kwargs['persistent_workers'] = self.persistent_workers

        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self):
        kwargs = self._get_common_dataloader_kwargs()
        
        if not self.is_mac:
            kwargs['persistent_workers'] = self.persistent_workers

        return DataLoader(self.test_dataset, **kwargs)

    def get_platform_info(self):
        return {
            'platform': 'macOS' if self.is_mac else 'Linux',
            'num_workers': self.num_workers,
            'multiprocessing_context': self.mp_context,
            'persistent_workers': self.persistent_workers if not self.is_mac else False,
            'prefetch_factor': self.prefetch_factor if not self.is_mac else 2,
            'cpu_count': os.cpu_count(),
            'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3)
        }
