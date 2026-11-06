# Cleanup candidates

The source/configuration map is in [training and evaluation](training_and_evaluation.md).
The earlier recipes still support distinct experiments; no unused source file
has been identified for removal. The generated artifacts below were removed on
2026-11-06. The cleanup removed 43 paths (9,509 file entries), approximately
297 MB of allocated file space, while preserving all staged source files and
the existing submodule changes. Python and developer tools may recreate caches.

| Location | Approximate size | Cleanup status |
| --- | --- | --- |
| `LensCraft/**/__pycache__/` | 6 MB | Removed; no tracked files were deleted. |
| `LensCraft/.pytest_cache/` | 84 KB | Removed. |
| `LensCraft/.ruff_cache/` | 20 KB | Removed. |
| Untracked `.DS_Store` files | Small | Removed from the checkout and old t-SNE directory. The existing tracked GenDoP deletion is documented separately in [third-party patches](third_party_patches.md). |
| `performance_reports/cross_dataset_20260928/tsne/.venv/` | 277 MiB | Removed after checking that no local t-SNE process used it. Python 3.14.3 and all 17 runtime packages match the preserved `requirements-local.lock.txt`; recreate the environment before rerunning its plotting scripts. |

The old t-SNE bytecode cache was also removed. Its features, figures, receipts,
scripts and dependency lock were preserved. The exact deletion inventory,
installed package versions and Git integrity checks are recorded outside Git in
`performance_reports/commit_cleanup_20261106/deletion_receipt.json`.

The `performance_reports` tree is a workspace sibling of the Git checkout,
approximately 3.8 GB in this review. The active November run reads historical
helpers, configurations, CLIP caches, evaluator/cohort manifests and report
guards from multiple experiment directories. Keep those directories and the
active observer files until the run and final integrity audit finish. Its
dependency manifests are the basis for a later archive/removal decision.

The 1.1 GB `lenscraft_dataset-human.zip` is a dataset archive. Its extension and
size alone do not establish that it is redundant; verify another complete copy
before treating it as a deletion candidate.

Web build directories such as `node_modules`, `.next` and `dist` in sibling
projects are generated too, but belong to separate projects and this source
cleanup does not modify them.
