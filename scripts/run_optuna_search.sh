#!/bin/bash
# Hyperparameter search (Optuna sweeper + joblib launcher).

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python src/train.py -m hydra/launcher=joblib hydra=optuna_sweeper hydra/sweeper=optuna
