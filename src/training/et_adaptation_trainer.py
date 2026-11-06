"""ET adaptation with simulation replay and independent validation in both domains."""

from contextlib import contextmanager
import math

import torch

from training.autoencoder_trainer import LightningLensCraft


class ETAdaptationTrainer(LightningLensCraft):
    def __init__(self, *args, simulation_replay_ratio=0.0,
                 simulation_validation_weight=1.0,
                 et_validation_objective='balanced', **kwargs):
        if et_validation_objective not in ('balanced', 'prompt_generation'):
            raise ValueError('et_validation_objective must be balanced or prompt_generation')
        super().__init__(*args, **kwargs)
        if not math.isfinite(simulation_replay_ratio) or not 0 <= simulation_replay_ratio < 1:
            raise ValueError('simulation_replay_ratio must be finite and in [0, 1)')
        if not math.isfinite(simulation_validation_weight) or simulation_validation_weight < 0:
            raise ValueError('simulation_validation_weight must be finite and nonnegative')
        if not self._validate_conditioning:
            raise ValueError('ET adaptation requires conditioning validation in both domains')
        if self._validate_caption_missingness:
            raise ValueError('Use native caption masks for adaptation validation; sparse validation is a separate experiment')
        self.simulation_replay_ratio = simulation_replay_ratio
        self.simulation_validation_weight = simulation_validation_weight
        self.et_validation_objective = et_validation_objective
        self.reference_simulation_score = None
        self._metric_domain = None
        self._domain_scores = {}
        self._domain_prompt_scores = {}

    @contextmanager
    def _domain(self, name):
        previous = self._metric_domain, self.dataset_mode, self.caption_missingness
        self._metric_domain = name
        self.dataset_mode = 'et' if name == 'et' else 'simulation'
        # ET already supplies native incomplete annotations. Augment only SIM
        # replay while retaining each batch's observed-field mask and metadata.
        if name == 'et':
            self.caption_missingness = None
        try:
            yield
        finally:
            self._metric_domain, self.dataset_mode, self.caption_missingness = previous

    def log(self, name, value, *args, **kwargs):
        domain = self._metric_domain
        if domain is not None:
            if name == 'val_conditioning_score':
                self._domain_scores[domain] = value.detach()
            elif name == 'val_conditioning/prompt_generation/score':
                self._domain_prompt_scores[domain] = value.detach()
            if domain == 'simulation':
                name = ('val_sim_' + name[4:] if name.startswith('val_') else
                        'train_sim_' + name[6:] if name.startswith('train_') else
                        'sim/' + name)
            elif name.startswith('train_'):
                name = 'train_et_' + name[6:]
        # Domain names are explicit; automatic /dataloader_idx suffixes would
        # make the checkpoint monitor differ between validate() and fit().
        kwargs['add_dataloader_idx'] = False
        return super().log(name, value, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.simulation_replay_ratio:
            et_batch, simulation_batch = batch['et'], batch['simulation']
        else:
            et_batch, simulation_batch = batch, None
        with self._domain('et'):
            et_loss = self._step(et_batch, batch_idx, 'train')
        total = et_loss
        if simulation_batch is not None:
            with self._domain('simulation'):
                sim_loss = self._step(simulation_batch, batch_idx, 'train')
            total = (1 - self.simulation_replay_ratio) * et_loss + self.simulation_replay_ratio * sim_loss
        self.log('train_loss', total, on_step=True, on_epoch=True,
                 batch_size=len(et_batch['camera_trajectory']))
        return total

    def _validation_state(self):
        return {name: value for name, value in vars(self).items()
                if name.startswith(('_conditioning_', '_caption_missingness_'))}

    def on_validation_epoch_start(self):
        self._domain_states = {}
        self._domain_scores = {}
        self._domain_prompt_scores = {}
        for domain in ('et', 'simulation'):
            with self._domain(domain):
                super().on_validation_epoch_start()
                self._domain_states[domain] = self._validation_state()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx not in (0, 1):
            raise ValueError('ET adaptation expects ET then simulation validation loaders')
        domain = ('et', 'simulation')[dataloader_idx]
        self.__dict__.update(self._domain_states[domain])
        with self._domain(domain):
            loss = self._step(batch, batch_idx, 'val')
        self._domain_states[domain] = self._validation_state()
        return loss

    def set_reference_simulation_score(self, score):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Initial simulation validation score must be finite')
        self.reference_simulation_score = score

    @property
    def et_validation_score(self):
        """ET objective before the unchanged balanced-SIM regression penalty.

        Prompt selection reuses the parent's complete pose-and-motion score;
        supplied-camera modes remain logged without selecting this checkpoint.
        The default retains the existing equal-weight conditioning objective.
        """
        scores = (self._domain_prompt_scores if self.et_validation_objective == 'prompt_generation'
                  else self._domain_scores)
        if 'et' not in scores:
            raise RuntimeError('ET validation must complete before reading its selection score')
        return scores['et']

    def on_fit_start(self):
        if self.reference_simulation_score is None:
            raise RuntimeError('Validate the initial checkpoint and set its SIM reference score before adaptation')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['et_adaptation_reference_simulation_score'] = self.reference_simulation_score

    def on_load_checkpoint(self, checkpoint):
        value = checkpoint.get('et_adaptation_reference_simulation_score')
        if value is not None:
            self.set_reference_simulation_score(value)

    def on_validation_epoch_end(self):
        for domain in ('et', 'simulation'):
            self.__dict__.update(self._domain_states[domain])
            with self._domain(domain):
                super().on_validation_epoch_end()
        et_score, sim_score = self.et_validation_score, self._domain_scores['simulation']
        if not torch.isfinite(torch.stack((et_score, sim_score))).all():
            raise RuntimeError('ET/SIM validation must produce finite conditioning scores')
        reference = sim_score if self.reference_simulation_score is None else self.reference_simulation_score
        regression = sim_score - reference
        self.log('val_sim_reference_score', reference, on_step=False, on_epoch=True)
        self.log('val_sim_regression', regression, on_step=False, on_epoch=True)
        self.log('val_et_selection_score', et_score, on_step=False, on_epoch=True)
        # Better SIM scores do not compensate for worse ET scores; forgetting
        # contributes a penalty relative to the same model before adaptation.
        self.log('val_adaptation_score',
                 et_score + self.simulation_validation_weight * regression.clamp_min(0),
                 on_step=False, on_epoch=True, prog_bar=True)

    def _get_total_steps(self):
        return self.trainer.estimated_stepping_batches
