"""SIM-only CCDM recipe comparisons with one common validation operator."""

from training.domain_augmentation import DomainAugmentation
from training.transfer_robust_trainer import TransferRobustTrainer


class CCDMFocusTrainer(TransferRobustTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Training recipes may differ in caption retention and native target
        # projection. Their selection scores must use identical held-out
        # targets and captions, independent of those training choices.
        self.ccdm_validation_augmentation = DomainAugmentation(
            {
                'enabled': True,
                'et_probability': 0.,
                'ccdm_probability': .75,
                'fill': 'zero',
                'et_coordinate_frame': 'world',
                'ccdm_static_motion': True,
                'ccdm_native_manifold': True,
            },
        )

    def _validate_domain_batch(self, batch, batch_idx):
        training_augmentation = self.domain_augmentation
        self.domain_augmentation = self.ccdm_validation_augmentation
        try:
            return super()._validate_domain_batch(batch, batch_idx)
        finally:
            self.domain_augmentation = training_augmentation
