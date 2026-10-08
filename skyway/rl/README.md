# Training and Validation Models

## Workspace

Training data and runs live in a workspace: `$SKYWAY_WORKSPACE` if set, otherwise the path in
`~/.skyway_profile` if that file exists, otherwise `~/.skyway`. Relative data and run paths in configs are
resolved against it.

## Training a Model

```bash
# Generate training lookup tables (written to <data dir>/lookups)
build-train-lookups --fits_path <path/to/train/data/in/fits/format> -o <data dir> [--acceptance des_per_band]

# Precompute the feature cache (once per data dir; add --field_features for field-level models)
precompute-train-features --data_dir <data dir> [--field_features]

# Run training
run-train -c <path/to/config>
# or, alternatively
run-train --config <path/to/config>
```

The training routine saves training results and the best model in
`<config parent_dir>/<experiment_name>/run_<YYYYMMDD_HHMMSS>/`:

```
run_<YYYYMMDD_HHMMSS>/
├─ configs/
│  ├─ resolved_config.yaml
│  └─ split.json
├─ checkpoints/
│  ├─ checkpoint_epoch_<epoch_num>_metric_<metric_val>.pt
│  ├─ checkpoint_history.json
│  ├─ latest_checkpoint.pt
│  ├─ model.pt
│  └─ normalization_stats.json
├─ metrics/
├─ logs/
└─ figures/
```

## Running Validation/Evaluation

```bash
# Evaluate a trained model on its test (or validation) nights
run-eval -c <run dir>/configs/resolved_config.yaml [--split val]

# Compare several trained models
run-model-compare -c <run dir 1>/configs/resolved_config.yaml <run dir 2>/configs/resolved_config.yaml

# Feature-importance analysis
run-explain --model_dir <run dir> [--sage] [--permutation]
```

To schedule future nights with a trained model, see the offline scheduler section of the
[root README](../../README.md); pass `-m <run dir>` to `run-offline-scheduler`.
