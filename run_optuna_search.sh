#!/bin/bash

cd "$(dirname "$0")"

export PYTHONPATH=$PYTHONPATH:$(pwd)

python src/optimize.py --multirun --config-dir=config --config-name=config \
  hydra/sweeper=optuna \
  hydra.sweeper.n_trials=20 \
  hydra.sweeper.n_jobs=1 \
  hydra.sweeper.direction=minimize
