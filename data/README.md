# Data

This folder ships the exact datasets used in the paper, gzip-compressed to keep
the repository small (all files are well under GitHub's size limits). Decompress
before running the pipeline:

```bash
gunzip data/train.csv.gz
gunzip data/processed_*/*.csv.gz
```

(`pandas.read_csv` can also read `.csv.gz` directly without decompressing --
just point `train_file`/`eval_file` in the config at the `.gz` path.)

## Contents

| Path | Description |
|---|---|
| `train.csv.gz` | Raw human-scientist idea ratings -- the source dataset consumed by `dataset_building/`. |
| `processed_general/` | Pooled pairwise preference dataset (`built_train.csv.gz` + per-field `built_ID_test_*`/`built_OOD_test_*` splits) used to train the `general` release checkpoint. |
| `processed_biology/` | Same, filtered to biology, used to train the `biology` release checkpoint. |
| `processed_chemistry/` | Same, filtered to chemistry, used to train the `chemistry` release checkpoint. |
| `processed_medicine/` | Same, filtered to medicine, used to train the `medicine` release checkpoint. |
| `processed_social/` | Same, filtered to social science, used to train the `social` release checkpoint. |

Each `processed_<field>/built_train.csv.gz` was built from `train.csv.gz` via
`dataset_building/build_dataset.py` (see the top-level README for the exact
command and config). The `built_ID_test_*`/`built_OOD_test_*` files are the
corresponding in-domain/out-of-domain evaluation splits, auto-discovered by
`training/train.py`.
