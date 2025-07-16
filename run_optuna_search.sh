#!/bin/bash

python src/train.py -m hydra/launcher=joblib hydra=optuna_sweeper hydra/sweeper=optuna
