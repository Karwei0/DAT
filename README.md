# DAT:Time Series Anomaly Detection Based on Dual Anomaly-Aware

DAT is an unsupervised anomaly detector for multivariate time series. It learns
what normal behaviour looks like from unlabelled training windows and assigns
an anomaly score to every future window. No anomaly labels are needed during
training.

## Model overview

DAT is built around a **dual anomaly-aware** training scheme:

- **Dual augmented branches.** Every window is augmented twice. A weak branch
  applies jittering and scaling, while a strong branch applies jittering, time
  warping and permutation. Both branches are embedded by separate linear
  projections and encoded by a **shared Transformer encoder**.
- **Anomaly-aware injection.** During training, synthetic anomalies are injected
  into both branches with three complementary strategies:
  *blended substitution*, *static override* and *peak implantation*. The
  injected positions provide the supervision used by the predictive branch.
- **Predictive branch.** The shared predictive head outputs a per-timestep
  anomaly score for each branch. The default objective is binary cross-entropy
  against the injection mask (`MSE`, `MAE`, `KL` and `CE` are also available).
- **Contextual contrastive branch.** Two branch-specific projections produce
  contextual representations, and a contrastive loss pulls together the
  weak/strong views of the same window while pushing apart different windows.
- **Adaptive loss balancing.** Both losses are divided by exponential moving
  averages of their own values, so neither objective dominates early training.
- **Scoring.** At test time the strong and weak predictions are weighted
  (default `0.6 / 0.4`), softmaxed, and flattened to one anomaly score per
  timestep. A threshold converts scores to binary labels: `score > threshold`
  is anomalous.

Training also supports **early stopping** on an averaged epoch loss and saves
every run in a timestamped directory, so experiments never overwrite each
other.

## Metrics

Evaluation reports the following metrics (see `metrics/get_all_evaluation_score.py`):

| Metric | Description |
| --- | --- |
| Precision / Recall / F1 (`*_ori`) | Point-wise precision, recall and F1 |
| Precision / Recall / F1 (`*_pa`) | Point-adjusted precision, recall and F1 |
| Point AUC | Point-wise area under the ROC curve |
| Range AUC | Range-based area under the ROC curve |
| Affiliation precision / recall | Event-level affiliation metrics |
| R_AUC_ROC / R_AUC_PR | Range-based AUC (ROC and PR) |
| VUS_ROC / VUS_PR | Volume under the surface (ROC and PR) |

The threshold used to turn scores into labels can be selected in four ways:

| `--threshold_method` | Threshold |
| --- | --- |
| `spot` (default) | Original SPOT / POT threshold fitted on train and test scores |
| `quantile_test` | Top `--threshold_q`% of the test-set scores |
| `quantile_train` | Top `--threshold_q`% of the train-set scores |
| `quantile_train_test` | Top `--threshold_q`% of the concatenated train+test scores |

`--threshold_q` defaults to `4.0`, i.e. the 96th percentile of the chosen
score distribution.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the datasets

The benchmark datasets are available here:

<https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR?usp=sharing>

Place each dataset under `data/`. The loaders expect:

```text
data/PSM/     train.csv        test.csv        test_label.csv
data/MSL/     MSL_train.npy    MSL_test.npy    MSL_test_label.npy
data/SMAP/    SMAP_train.npy   SMAP_test.npy   SMAP_test_label.npy
data/SMD/     SMD_train.npy    SMD_test.npy    SMD_test_label.npy
data/SWAT/    swat_train2.csv  swat2.csv
```

### 3. Train

```bash
python train.py \
    --dataset=PSM --data_path=data/PSM \
    --patch_size=10 --win_size=25 --step=10 \
    --num_epochs=10 --exp_id=psm_baseline
```

A run is written to `logs/<dataset>_<exp_id>_<timestamp>/`, and
`logs/<dataset>_<exp_id>_latest.json` points to the newest run for that
experiment. The directory contains:

- `DAT_model.pt` – model definition,
- `state_dict.pt` – final weights (best weights when early stopping restores them),
- `best_state_dict.pt` – best monitored epoch (when early stopping is enabled),
- `state/state_dict_epoch_<n>.pt` – per-epoch checkpoints,
- `hyperparameters.txt`, `run_info.json`, `early_stopping.json`.

Early stopping is enabled by default and monitors the averaged `total_loss`:

```bash
# stop after 15 epochs without an improvement larger than 1e-4
--early_stop_patience=15 --early_stop_min_delta=1e-4

# monitor another loss, disable early stopping, or keep the last epoch's weights
--early_stop_monitor=contextual_loss
--no-early_stopping
--no-restore_best
```

### 4. Evaluate

Because run directories are timestamped, evaluation can resolve the newest
checkpoint from the experiment name instead of hard-coding a path:

```bash
python estimate.py \
    --dataset=PSM --data_path=data/PSM --exp_id=psm_baseline \
    --threshold_method=quantile_test --threshold_q=4 \
    --anomaly_score_outfiledir=logs/PSM_psm_baseline_result
```

`--log_dir` pins a specific run, while `--model` and `--model_state_dict` still
accept explicit checkpoint paths. The output directory receives
`anomaly_score.npy`, `train_anomaly_score.npy`, `anomaly_true_label.npy` and
`eval_res.csv` (including the selected threshold and method).

## Acknowledgement

This project builds on the baseline and datasets provided by
[DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector), and we thank
[Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) and
[OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) for their
implementations and resources. We are also grateful to all the models
mentioned in this work.
