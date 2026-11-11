# Full paper evaluation

Run from `LensCraft` with the same Python environment used for training:

```bash
python scripts/run_paper_evaluation.py \
  --dataset-path /data/simulation \
  --lenscraft-checkpoint '/runs/categorical/last.pt' \
  --lenscraft-config /runs/train/.hydra/config.yaml \
  --clatr-checkpoint /runs/clatr/checkpoints/best.ckpt \
  --test-fraction 0.1 \
  --output-dir /runs/paper_evaluation
```

The example selects the first 10% of seed-42 held-out indices before static/dynamic
filtering, shared by every model. It does not queue a later full evaluation.
Review these exploratory results before choosing a full run with a fresh output
directory and `--test-fraction 1.0` (the default).

The runner evaluates the selected held-out static and dynamic samples for four models,
then measures each model at batch size 16 and creates tables. The main evaluation
batch size is 128, workers 8, bootstrap replicates 500, device `cuda:0`.
LensCraft includes 13 modes (K=1,2,4,8,26); baselines include both native and shared-normalized
coordinate modes and retain their native sequence lengths.
Movement filtering happens after the seed-42 split. The training configuration
must have unfiltered 60/20/20 splits. Supply `--split-manifest` to verify the saved
split explicitly. Shell environment values take precedence over `.env`.

Pass a separately frozen semantic encoder using
`--semantic-evaluator-checkpoint` and `--semantic-evaluator-config`; it is not
selected implicitly from the categorical generator. Native CLaTr
requires its own explicit checkpoint. Baseline paths come from the environment.
Historical architecture ablation results require their frozen original sources;
the current project exposes one categorical generator. Geometry means have no bootstrap uncertainty; baseline
FLOPs remain unavailable where the implementation cannot measure them.

Preview commands with `--dry-run`. Use a fresh output directory for each new
experiment. `--resume` only skips validated completed stages when source hashes,
checkpoint hashes, dataset inventory and commands are identical. Failed stages
rerun and append to their existing log. Changes require a new directory.

To survive an SSH disconnect, launch inside tmux:

```bash
tmux new -s lenscraft-paper
# Run the command above, then detach with Ctrl-b d.
tmux attach -t lenscraft-paper
```

Track the run using:

```bash
python -m json.tool /runs/paper_evaluation/status.json
tail -f /runs/paper_evaluation/logs/evaluation_lenscraft_static.log
nvidia-smi
```

`status.json` records the current stage, runner/child PIDs, timestamps, logs,
completion status and validated output hashes. Results live directly under
`results/`; the aggregator reads that directory without recursion. Final Markdown
tables and `all_metrics.csv` appear under `tables/`. Failures stop the runner and
preserve all logs and prior completed stages.
