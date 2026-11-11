# ET-shaped training and native ET evaluation

The maintained generator uses the same categorical bottleneck for SIM, ET and
CCDM. The former CLIP continuation/adaptation launchers have been retired.
Their historical results require their frozen source and receipts; they are
not alternative generator modes in this checkout.

Train directly on annotated ET data with:

```bash
python src/train.py data/dataset=et
```

For SIM-only observation policies, use `training=et_robust` or
`training=transfer_robust data.compact_batches=false`. Missing facts are zero
categorical tokens excluded from attention. These policies retain categorical
CE and detached cycle KL; they do not use a CLIP vocabulary or native test masks.

Native ET conversion, subject context and normalization remain dataset adapter
concerns. Keep source-video grouping and cohort membership fixed for evaluation.
See [training recipes](training_and_evaluation.md), [ET evaluation](et_evaluation.md),
and [fixed judges](evaluation.md).
