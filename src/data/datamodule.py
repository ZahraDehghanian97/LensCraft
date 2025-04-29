import os
import hydra
import lightning as L
from torch.utils.data import random_split, DataLoader
from data.simulation.dataset import collate_fn
from data.et.dataset import collate_fn as et_collate_fn
from data.ccdm.dataset import collate_fn as ccdm_collate_fn


class CameraTrajectoryDataModule(L.LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers=None, val_size=0.1, test_size=0.1):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
        
        if 'ETDataset' in self.dataset_config['_target_']:
            self.dataset_mode = 'et'
            self.collate_fn = et_collate_fn
        elif 'CCDMDataset' in self.dataset_config['_target_']:
            self.dataset_mode = 'ccdm'
            self.collate_fn = ccdm_collate_fn
        else:
            self.dataset_mode = 'simulation'
            self.collate_fn = collate_fn

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
