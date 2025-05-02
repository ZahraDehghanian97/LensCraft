#!/bin/bash

python src/train.py -m hydra=optuna_sweeper hydra/sweeper=optuna
