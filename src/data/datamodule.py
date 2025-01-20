import hydra
import lightning as L
from torch.utils.data import random_split, DataLoader
from data.simulation.dataset import collate_fn
from data.et.dataset import collate_fn as et_collate_fn


class CameraTrajectoryDataModule(L.LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers, val_size, test_size):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.dataset_mode = 'et' if 'ETDataset' in dataset_config['_target_'] else 'simulation'
        self.collate_fn = et_collate_fn if self.dataset_mode == 'et' else collate_fn

    def setup(self, stage=None):
        full_dataset = hydra.utils.instantiate(self.dataset_config)
        train_size = int((1 - self.val_size - self.test_size)
                         * len(full_dataset))
        val_size = int(self.val_size * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            multiprocessing_context='fork',
            pin_memory=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            multiprocessing_context='fork'
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            multiprocessing_context='fork'
        )
