#!/bin/bash

python src/train.py -m hydra/sweeper=optuna
