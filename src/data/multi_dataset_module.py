import os
import platform
import psutil
import lightning as L
from torch.utils.data import DataLoader, random_split
import hydra
from hydra.utils import instantiate

class MultiDatasetModule(L.LightningDataModule):
    """
    DataModule that handles multiple datasets (simulation and CCDM) simultaneously.
    """
    def __init__(
        self, 
        simulation_config, 
        ccdm_config, 
        batch_size, 
        num_workers=None, 
        val_size=0.1, 
        test_size=0.1,
        sim_ratio=0.5  # Ratio of simulation samples in each batch
    ):
        super().__init__()
        self.simulation_config = simulation_config
        self.ccdm_config = ccdm_config
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.sim_ratio = sim_ratio
        
        self.sim_batch_size = max(1, int(batch_size * sim_ratio))
        self.ccdm_batch_size = batch_size - self.sim_batch_size
        
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
        from data.simulation.dataset import collate_fn as sim_collate_fn
        self.sim_collate_fn = sim_collate_fn
        simulation_dataset = hydra.utils.instantiate(self.simulation_config)
        
        if hasattr(simulation_dataset, 'preprocess'):
            simulation_dataset.preprocess()
            
        from data.ccdm.dataset import collate_fn as ccdm_collate_fn
        self.ccdm_collate_fn = ccdm_collate_fn
        ccdm_dataset = hydra.utils.instantiate(self.ccdm_config)
        
        if hasattr(ccdm_dataset, 'preprocess'):
            ccdm_dataset.preprocess()

        sim_train_size = int((1 - self.val_size - self.test_size) * len(simulation_dataset))
        sim_val_size = int(self.val_size * len(simulation_dataset))
        sim_test_size = len(simulation_dataset) - sim_train_size - sim_val_size
        
        self.sim_train, self.sim_val, self.sim_test = random_split(
            simulation_dataset, [sim_train_size, sim_val_size, sim_test_size]
        )
        
        ccdm_train_size = int((1 - self.val_size - self.test_size) * len(ccdm_dataset))
        ccdm_val_size = int(self.val_size * len(ccdm_dataset))
        ccdm_test_size = len(ccdm_dataset) - ccdm_train_size - ccdm_val_size
        
        self.ccdm_train, self.ccdm_val, self.ccdm_test = random_split(
            ccdm_dataset, [ccdm_train_size, ccdm_val_size, ccdm_test_size]
        )

    def _get_dataloader_kwargs(self, shuffle=False):
        kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True,
            'shuffle': shuffle,
        }
        
        if not self.is_mac:
            kwargs.update({
                'persistent_workers': self.persistent_workers,
                'prefetch_factor': self.prefetch_factor,
                'multiprocessing_context': self.mp_context,
            })
            
        return kwargs

    def train_dataloader(self):
        kwargs = self._get_dataloader_kwargs(shuffle=True)
        
        sim_loader = DataLoader(
            self.sim_train, 
            batch_size=self.sim_batch_size, 
            collate_fn=self.sim_collate_fn,
            **kwargs
        )
        
        ccdm_loader = DataLoader(
            self.ccdm_train, 
            batch_size=self.ccdm_batch_size, 
            collate_fn=self.ccdm_collate_fn,
            **kwargs
        )
        
        return {
            'simulation': sim_loader,
            'ccdm': ccdm_loader
        }

    def val_dataloader(self):
        kwargs = self._get_dataloader_kwargs()
        
        sim_loader = DataLoader(
            self.sim_val, 
            batch_size=self.sim_batch_size, 
            collate_fn=self.sim_collate_fn,
            **kwargs
        )
        
        ccdm_loader = DataLoader(
            self.ccdm_val, 
            batch_size=self.ccdm_batch_size, 
            collate_fn=self.ccdm_collate_fn,
            **kwargs
        )
        
        return [sim_loader, ccdm_loader]

    def test_dataloader(self):
        kwargs = self._get_dataloader_kwargs()
        
        sim_loader = DataLoader(
            self.sim_test, 
            batch_size=self.sim_batch_size, 
            collate_fn=self.sim_collate_fn,
            **kwargs
        )
        
        ccdm_loader = DataLoader(
            self.ccdm_test, 
            batch_size=self.ccdm_batch_size, 
            collate_fn=self.ccdm_collate_fn,
            **kwargs
        )
        
        return [sim_loader, ccdm_loader]

    def get_platform_info(self):
        return {
            'platform': 'macOS' if self.is_mac else 'Linux',
            'num_workers': self.num_workers,
            'multiprocessing_context': self.mp_context if not self.is_mac else 'fork',
            'persistent_workers': self.persistent_workers if not self.is_mac else False,
            'prefetch_factor': self.prefetch_factor if not self.is_mac else 2,
            'cpu_count': os.cpu_count(),
            'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'simulation_batch_size': self.sim_batch_size,
            'ccdm_batch_size': self.ccdm_batch_size
        }
