# -*- coding = utf-8 -*-
# @Time: 2025/2/11 22:00
# @Author: wisehone
# @File: estimate.py
# @SoftWare: PyCharm
"""Evaluate a trained DAT model.

Anomaly scores are computed for the test and train splits, a threshold is
selected with one of the strategies in :mod:`utils.threshold`, and scores
strictly above the threshold are labelled as anomalies.

The checkpoint can be given explicitly (``--model`` + ``--model_state_dict``)
or resolved from a training run (``--exp_id``), in which case the newest
timestamped run directory for that experiment is used.
"""

import argparse, glob, json, os

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data.data_loaders import get_data_loader
from metrics.get_all_evaluation_score import get_all_evaluation_score
from utils.threshold import THRESHOLD_METHODS, select_threshold


def estimate(model, options, out_dim, device):
    win_size = model.max_seq_len * model.patch_size
    step = win_size
    batch_size = options.batch_size

    assert options.strong_pred_weight + options.weak_pred_weight == 1.0

    test_weak_data_loader = get_data_loader(0, options.data_path, options.batch_size, win_size=win_size, step=step,
                                            mode='test', dataset=options.dataset,
                                            augment='weak', w_jittering=options.w_jittering, scaling=options.scaling,
                                            num_workers=1)
    if len(test_weak_data_loader) < 1:
        raise RuntimeError('no test data')

    test_strong_data_loader = get_data_loader(0, options.data_path, options.batch_size, win_size=win_size, step=step,
                                              mode='test', dataset=options.dataset,
                                              augment='strong', s_jittering=options.s_jittering,
                                              time_wrapping=options.time_wrapping,
                                              permutation=options.permutation, num_workers=1)

    anomaly_true_label = []
    pred_anomaly_score = []

    for (strong_x, labels), (weak_x, _) in zip(test_strong_data_loader, test_weak_data_loader):
        strong_x = strong_x.to(device)
        weak_x = weak_x.to(device)

        strong_x_pred, weak_x_pred, _, _ = model(strong_x, weak_x)

        strong_x_pred = strong_x_pred.view(batch_size, -1)
        weak_x_pred = weak_x_pred.view(batch_size, -1)

        weighted_anomaly_score = F.softmax(
            strong_x_pred * options.strong_pred_weight + weak_x_pred * options.weak_pred_weight, dim=-1)

        pred_anomaly_score.append(weighted_anomaly_score.detach().cpu().numpy())
        anomaly_true_label.append(labels.detach().cpu().numpy())

    anomaly_true_label = np.concatenate(anomaly_true_label, axis=0).reshape(-1)
    pred_anomaly_score = np.concatenate(pred_anomaly_score, axis=0).reshape(-1)
    anomaly_true_label = np.array(anomaly_true_label)
    pred_anomaly_score = np.array(pred_anomaly_score)

    print('anomaly score shape: {}'.format(pred_anomaly_score.shape))
    return pred_anomaly_score, anomaly_true_label


def estimate_train(model, options, out_dim, device):
    win_size = model.max_seq_len * model.patch_size
    step = win_size
    batch_size = options.batch_size

    train_weak_data_loader = get_data_loader(0, options.data_path, options.batch_size, win_size=win_size, step=step,
                                             mode='train', dataset=options.dataset,
                                             augment='weak', w_jittering=options.w_jittering, scaling=options.scaling,
                                             num_workers=1)
    if len(train_weak_data_loader) < 1:
        print('no train data')
        return np.array([], dtype=np.float64)

    train_strong_data_loader = get_data_loader(0, options.data_path, options.batch_size, win_size=win_size, step=step,
                                               mode='train', dataset=options.dataset,
                                               augment='strong', s_jittering=options.s_jittering,
                                               time_wrapping=options.time_wrapping,
                                               permutation=options.permutation, num_workers=1)

    train_anomaly_score = []
    for (strong_x, _), (weak_x, _) in zip(train_strong_data_loader, train_weak_data_loader):
        strong_x = strong_x.to(device)
        weak_x = weak_x.to(device)

        strong_x_pred, weak_x_pred, _, _ = model(strong_x, weak_x)
        strong_x_pred = strong_x_pred.view(batch_size, -1)
        weak_x_pred = weak_x_pred.view(batch_size, -1)
        anomaly_score = F.softmax(strong_x_pred * options.strong_pred_weight + weak_x_pred * options.weak_pred_weight,
                                  dim=-1)
        anomaly_score = anomaly_score.detach().cpu().numpy()
        train_anomaly_score.append(anomaly_score)

    train_anomaly_score = np.concatenate(train_anomaly_score, axis=0).reshape(-1)
    return np.array(train_anomaly_score)


# ---------------------------------------------------------------------------
# run-directory / checkpoint resolution
# ---------------------------------------------------------------------------
def resolve_run_dir(options):
    """Find the newest run directory for ``--exp_id`` (or use ``--log_dir``)."""
    if options.log_dir is not None:
        if not os.path.isdir(options.log_dir):
            raise FileNotFoundError(f'log directory not found: {options.log_dir}')
        return options.log_dir

    if options.exp_id is None:
        raise ValueError('either --exp_id or --log_dir is required when '
                         '--model/--model_state_dict are not both provided')

    log_root = 'logs'
    pointer = os.path.join(log_root, f'{options.dataset}_{options.exp_id}_latest.json')
    if os.path.isfile(pointer):
        with open(pointer) as f:
            run_info = json.load(f)
        run_dir = run_info.get('log_dir')
        if run_dir and os.path.isdir(run_dir):
            return run_dir
        print(f'Pointer {pointer} references a missing directory, falling back to disk search')

    # Backwards compatible with the old layout logs/<dataset>_<exp_id>, and
    # works for the timestamped logs/<dataset>_<exp_id>_<timestamp> layout.
    prefix = f'{options.dataset}_{options.exp_id}'
    candidates = []
    if os.path.isdir(log_root):
        for name in os.listdir(log_root):
            path = os.path.join(log_root, name)
            if os.path.isdir(path) and (name == prefix or name.startswith(prefix + '_')):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f'no run directory found for dataset={options.dataset!r}, '
            f'exp_id={options.exp_id!r}; pass --log_dir or explicit '
            f'--model/--model_state_dict')
    return max(candidates, key=os.path.getmtime)


def _with_pt(path):
    """Tolerate checkpoints given without the .pt suffix (old scripts did this)."""
    if os.path.exists(path):
        return path
    if os.path.exists(path + '.pt'):
        return path + '.pt'
    return path


def _newest_epoch_checkpoint(run_dir):
    files = glob.glob(os.path.join(run_dir, 'state', 'state_dict_epoch_*.pt'))
    if not files:
        return None

    def epoch_number(path):
        name = os.path.basename(path)
        try:
            return int(name[len('state_dict_epoch_'):-len('.pt')])
        except ValueError:
            return -1

    return max(files, key=epoch_number)


def resolve_checkpoint_paths(options):
    """Fill in ``options.model`` / ``options.model_state_dict`` when omitted."""
    needs_run_dir = options.model is None or options.model_state_dict is None
    run_dir = resolve_run_dir(options) if needs_run_dir else None
    if run_dir is not None:
        print(f'Using run directory: {run_dir}')

    if options.model is None:
        options.model = _with_pt(os.path.join(run_dir, 'DAT_model.pt'))
        if not os.path.exists(options.model):
            raise FileNotFoundError(f'no DAT_model.pt in {run_dir}')
    else:
        options.model = _with_pt(options.model)

    if options.model_state_dict is None:
        for name in ('state_dict.pt', 'best_state_dict.pt'):
            candidate = os.path.join(run_dir, name)
            if os.path.exists(candidate):
                options.model_state_dict = candidate
                break
        else:
            options.model_state_dict = _newest_epoch_checkpoint(run_dir)
        if options.model_state_dict is None:
            raise FileNotFoundError(f'no checkpoint found in {run_dir}')
    else:
        options.model_state_dict = _with_pt(options.model_state_dict)

    print(f'Using model: {options.model}')
    print(f'Using state dict: {options.model_state_dict}')


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(options):
    data_lm = {
        'MSL': (0.03, 4.485),
        'SMAP': (0.03, 3.24),
        'PSM': (0.08, 2.22),
        'SWAT': (0.08, 5.8),
        'SMD': (0.2, 7.6),
    }
    lm = data_lm[options.dataset]

    resolve_checkpoint_paths(options)

    device = torch.device('cuda:{}'.format(options.gpu_id))
    if not options.use_gpu:
        device = torch.device('cpu')

    model = torch.load(options.model, map_location=device, weights_only=False)
    print('load model success')
    if options.model_state_dict is not None:
        model.load_state_dict(torch.load(options.model_state_dict, map_location='cpu'))
        print('load model state dict success')
    model.eval()

    anomaly_score, anomaly_true_label = estimate(model, options, out_dim=1, device=device)
    train_anomaly_score = estimate_train(model, options, out_dim=1, device=device)

    if options.anomaly_score_outfiledir is not None:
        os.makedirs(options.anomaly_score_outfiledir, exist_ok=True)
        np.save(file=os.path.join(options.anomaly_score_outfiledir, 'anomaly_score.npy'), arr=anomaly_score)
        np.save(file=os.path.join(options.anomaly_score_outfiledir, 'anomaly_true_label.npy'), arr=anomaly_true_label)
        np.save(file=os.path.join(options.anomaly_score_outfiledir, 'train_anomaly_score.npy'), arr=train_anomaly_score)
        print('anomaly score saved in {}'.format(options.anomaly_score_outfiledir))

    # threshold selection
    threshold = select_threshold(
        options.threshold_method, train_anomaly_score, anomaly_score,
        quantile=options.threshold_q, spot_level=lm[0], spot_scale=lm[1])
    pred_label = (anomaly_score > threshold).astype(np.float32)

    eval_res = get_all_evaluation_score(pred_label, anomaly_true_label)
    eval_res = {
        key: round(value, 8) if isinstance(value, float) else value
        for key, value in eval_res.items()
    }
    eval_res['threshold'] = round(float(threshold), 8)
    eval_res['threshold_method'] = options.threshold_method
    if options.threshold_method != 'spot':
        eval_res['threshold_q'] = options.threshold_q

    eval_df = pd.DataFrame(eval_res, index=[0])
    if options.anomaly_score_outfiledir is not None:
        eval_df.to_csv(os.path.join(options.anomaly_score_outfiledir, 'eval_res.csv'), index=False)

    print(f'threshold method: {options.threshold_method}, threshold: {threshold}')
    print(f'precision_PA: {eval_res["precision_pa"]}, recall_PA: {eval_res["recall_pa"]}, '
          f'f1_PA: {eval_res["f_score_pa"]}')
    print(f'Affiliation Precision: {eval_res["Affiliation precision"]}, '
          f'Affiliation Recall: {eval_res["Affiliation recall"]}')
    print(f'thres: {threshold}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument('--dataset', type=str, default='PSM', choices=['MSL', 'PSM', 'SMAP', 'SMD', 'SWAT'],
                        help='the dataset to train, PSM(default)/MSL/SMAP/SMD/SWAT')
    parser.add_argument('--data_path', type=str, default='data/PSM', help='the input data path')

    # checkpoint: either explicit paths, or an experiment id resolved from logs/
    parser.add_argument('--model', type=str, default=None,
                        help='the model file(.pt) to detect anomaly; resolved from --exp_id when omitted')
    parser.add_argument('--model_state_dict', type=str, default=None,
                        help='the model state dict file(.pt); resolved from --exp_id when omitted')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='experiment id/name used during training; the newest matching run is used')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='explicit run directory; overrides --exp_id resolution')
    parser.add_argument("--anomaly_score_outfiledir", default=None, type=str,
                        help='output file dir name (.npy) to save anomaly scores')

    # threshold selection
    parser.add_argument('--threshold_method', type=str, default='spot', choices=list(THRESHOLD_METHODS),
                        help='spot(default)/quantile_test/quantile_train/quantile_train_test')
    parser.add_argument('--threshold_q', type=float, default=4.0,
                        help='top q%% of the score distribution used by the quantile methods, 4.0(default)')

    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument('--w_jittering', type=float, default=0.5,
                        help='the weight of jittering in weak data augmentation, 0.5(default)')
    parser.add_argument('--scaling', type=float, default=0.5,
                        help='the weight of scaling in weak data augment, 0.5(default)')

    parser.add_argument('--s_jittering', type=float, default=0.3,
                        help='the weight of jittering in strong data augmentation, 0.3(default)')
    parser.add_argument('--time_wrapping', type=float, default=0.6,
                        help='the weight of time wrapping in strong data augmentation, 0.6(default)')
    parser.add_argument('--permutation', type=float, default=0.1,
                        help='the weight of permutation in strong data augmentation, 0.1(default)')

    parser.add_argument('--strong_pred_weight', type=float, default=0.6,
                        help='the weight of strong augment data predicition, 0.5(default)')
    parser.add_argument('--weak_pred_weight', type=float, default=0.4,
                        help='the weight of weak augment data prediction, 0.5(default)')
    options = parser.parse_args()
    main(options)
