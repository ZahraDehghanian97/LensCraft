"""Run paired simulation-only fine-tuning with and without sparse captions.

Both runs start from the same checkpoint and select on the same simulation
validation objective. ET references and annotations are not training inputs.
Run from the prepared server environment; outputs never overwrite the baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--train-batches', type=int, default=100)
    parser.add_argument('--val-batches', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--variants', nargs='+', choices=['dense', 'sparse'], default=['dense', 'sparse'])
    args = parser.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    manifest = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'initial_checkpoint_sha256': digest(args.checkpoint),
        'source_sha256': {str(p.relative_to(args.repo)): digest(p)
                          for directory in ('src', 'config')
                          for p in sorted((args.repo / directory).rglob('*'))
                          if p.is_file() and p.suffix in ('.py', '.yaml')},
        'protocol': 'Simulation-only adaptation; original seed-42 60/20/20 split. '
                    'Synthetic masks only. Equal optimization budgets and sparse validation '
                    'objective for both variants; no ET-based checkpoint selection.',
        'runs': {},
    }
    save(args.output / 'manifest.json', manifest)
    for variant in args.variants:
        out = args.output / variant
        if (out / 'status.json').exists():
            raise FileExistsError(f'Use a new output directory: {out}')
        out.mkdir(parents=True, exist_ok=True)
        overrides = {
            'training': 'et_robust',
            'initialize_from_checkpoint': str(args.checkpoint.resolve()),
            'training.optimizer.lr': args.lr,
            'training.lr_scheduler': None,
            'training.caption_missingness.probability': 0.0 if variant == 'dense' else 0.75,
            'training.conditioning.mode_weights.prompt_generation': 3.0,
            'training.validation_conditioning.max_batches': args.val_batches,
            'training.validation_caption_missingness.max_batches': args.val_batches,
            'trainer.max_epochs': args.epochs,
            'trainer.accelerator': 'gpu', 'trainer.devices': [0],
            '+trainer.limit_train_batches': args.train_batches,
            '+trainer.limit_val_batches': args.val_batches,
            '+trainer.num_sanity_val_steps': 0,
            '+trainer.enable_progress_bar': False,
            'data.batch_size': args.batch_size, 'data.num_workers': 4,
            'data.val_size': 0.2, 'data.test_size': 0.2,
            'seed': 42, 'patience': args.epochs + 1,
            'hydra.run.dir': str(out),
            'callbacks.train_checkpoint.save_top_k': 0,
        }
        command = [sys.executable, str(args.repo / 'src/train.py')]
        command.extend(k + '=' + json.dumps(v) for k, v in overrides.items())
        manifest['runs'][variant] = {'command': command, 'output': str(out)}
        save(args.output / 'manifest.json', manifest)
        started = time.monotonic()
        print(f'Starting {variant}: {out}', flush=True)
        with (out / 'train.log').open('w') as log:
            result = subprocess.run(command, cwd=args.repo, env=os.environ.copy(),
                                    stdout=log, stderr=subprocess.STDOUT)
        status = {'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - started,
                  'completed_at': datetime.now(timezone.utc).isoformat(),
                  'checkpoints': [str(p) for p in out.rglob('*.ckpt')]}
        save(out / 'status.json', status)
        print(json.dumps({variant: status}), flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
