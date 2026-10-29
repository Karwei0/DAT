# -*- coding = utf-8 -*-
# @Time: 2025/2/11 22:00
# @Author: wisehone
# @File: train.py
# @SoftWare: PyCharm
"""Train the DAT model.

On top of the original training recipe (dual weak/strong branches, anomaly
injection, predictive + contextual losses) this script adds:

* optional early stopping on an averaged epoch loss;
* run directories named ``<dataset>_<exp_id>_<timestamp>`` plus a
  ``logs/<dataset>_<exp_id>_latest.json`` pointer, so estimate.py can locate
  the newest checkpoint for an experiment without knowing the timestamp.

Run ``python train.py -h`` for all options.
"""

import argparse, json, math, os, time

import torch
from torch import nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from data import data_loaders
from models.DAT import get_DAT_model
from models.contextual_loss import ContextualLoss
from utils.anomalies import Anomalies_injection
from utils.loss_func import KL_loss, MSE_loss, MAE_loss, CE_loss, BCE_loss

# epoch-level metrics that early stopping can monitor
EPOCH_METRICS = ('total_loss', 'predictive_loss', 'contextual_loss')
PRED_LOSS_FUNCS = {'KL': KL_loss, 'MSE': MSE_loss, 'MAE': MAE_loss,
                   'CE': CE_loss, 'BCE': BCE_loss}


class EarlyStopping:
    """Early stopping on an averaged epoch-level loss.

    A value counts as an improvement only when it beats the best value seen so
    far by more than ``min_delta``. Training should stop after ``patience``
    consecutive epochs without an improvement; ``patience <= 0`` means the
    metric is still tracked (so the best checkpoint can be kept) but training
    never stops early.
    """

    def __init__(self, patience=15, min_delta=1e-4, monitor='total_loss', mode='min'):
        self.patience = int(patience)
        self.min_delta = abs(float(min_delta))
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, score, epoch):
        """Record ``score`` for ``epoch``; return True when it is the new best."""
        score = float(score)
        if not math.isfinite(score):
            # NaN/Inf cannot be an improvement, but it should not crash training
            self.num_bad_epochs += 1
        else:
            if self.mode == 'min':
                improved = score < self.best_score - self.min_delta
            else:
                improved = score > self.best_score + self.min_delta
            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.num_bad_epochs = 0
                return True
            self.num_bad_epochs += 1
        if self.patience > 0 and self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------
def resolve_device(options):
    if options.use_gpu:
        return torch.device('cuda:{}'.format(options.gpu_id))
    return torch.device('cpu')


def build_run_dir(options):
    """Create a timestamped run directory and remember it for estimate.py.

    The user supplied ``exp_id`` is kept as the readable prefix; the timestamp
    makes every run unique, so experiments never overwrite each other.
    """
    log_root = 'logs'
    os.makedirs(log_root, exist_ok=True)

    dataset = getattr(options, 'dataset', 'run')
    exp_id = str(getattr(options, 'exp_id', 'run')).strip() or 'run'
    exp_id = exp_id.replace(os.sep, '_').replace(' ', '_')
    options.exp_id = exp_id

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    run_name = f'{dataset}_{exp_id}_{timestamp}'
    log_dir = os.path.join(log_root, run_name)
    suffix = 1
    while os.path.exists(log_dir):  # two runs started in the same second
        suffix += 1
        run_name = f'{dataset}_{exp_id}_{timestamp}_{suffix}'
        log_dir = os.path.join(log_root, run_name)
    os.makedirs(os.path.join(log_dir, 'state'))

    run_info = {
        'dataset': dataset,
        'exp_id': exp_id,
        'run_name': run_name,
        'log_dir': log_dir,
        'created_at': timestamp,
    }
    with open(os.path.join(log_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)

    # pointer used by estimate.py to resolve the newest run for this experiment
    pointer = os.path.join(log_root, f'{dataset}_{exp_id}_latest.json')
    with open(pointer, 'w') as f:
        json.dump(run_info, f, indent=2)

    return log_dir


def build_data_loaders(options, data_seq_len):
    """Build the weak/strong/random loaders and the anomaly injector."""
    abl = options.ablation
    num_workers = options.num_workers
    contamination_rate = getattr(options, 'r', 0.0)

    # weak augment data
    assert options.w_jittering + options.scaling == 1.0
    weak_loader = data_loaders.get_data_loader(
        options.index, options.data_path, options.batch_size,
        win_size=data_seq_len, step=options.step, mode='train',
        dataset=options.dataset, num_workers=num_workers,
        augment='weak' if abl != 'aug' else None,
        w_jittering=options.w_jittering, scaling=options.scaling,
        contamination_rate=contamination_rate)


    # strong augment data
    assert options.s_jittering + options.time_wrapping + options.permutation >= 0.99998
    strong_loader = data_loaders.get_data_loader(
        options.index, options.data_path, options.batch_size,
        win_size=data_seq_len, step=options.step, mode='train',
        dataset=options.dataset, num_workers=num_workers,
        augment='strong' if abl != 'aug' else None,
        s_jittering=options.s_jittering, time_wrapping=options.time_wrapping,
        permutation=options.permutation, contamination_rate=contamination_rate)

    # random original data for anomaly injection
    random_loader = data_loaders.get_data_loader(
        options.index, options.data_path, options.batch_size,
        win_size=data_seq_len, step=options.step, mode='random',
        dataset=options.dataset, num_workers=num_workers)

    anomaly_inj = Anomalies_injection(
        random_loader, options.anomaly_rate, options.blended_substitution,
        options.static_override, options.peak_implantation,
        options.flip_replacing_interval, options.replacing_weight)

    return weak_loader, strong_loader, anomaly_inj, random_loader.dataset.num_features


def build_model(options, num_features, device):
    model = get_DAT_model(
        input_d_data=num_features,
        output_d_data=1,
        patch_size=options.patch_size,
        d_embed=options.d_embed,
        hidden_dim_rate=4.,
        max_seq_len=options.win_size,
        positional_encoding=None,
        relative_position_embedding=True,
        transformer_n_layer=options.n_layers,
        transformer_n_head=options.n_heads,
        dropout=options.dropout,
        d_pre_rate=options.d_pre_rate,
        d_con_rate=options.d_con_rate)
    model.to(device)
    return model


def build_pred_loss_func(options):
    if options.ablation == 'ano':
        return None
    return PRED_LOSS_FUNCS.get(getattr(options, 'pre_loss_fuc', 'BCE'))


def build_early_stopper(options):
    """Return (EarlyStopping | None, monitor name, restore_best flag)."""
    if not getattr(options, 'early_stopping', True):
        print('Early stopping disabled')
        return None, getattr(options, 'early_stop_monitor', 'total_loss'), False

    monitor = getattr(options, 'early_stop_monitor', 'total_loss')
    if monitor not in EPOCH_METRICS:
        print(f"Unknown early stopping monitor '{monitor}', fallback to 'total_loss'")
        monitor = 'total_loss'
    restore_best = getattr(options, 'restore_best', True)
    stopper = EarlyStopping(
        patience=getattr(options, 'early_stop_patience', 15),
        min_delta=getattr(options, 'early_stop_min_delta', 1e-4),
        monitor=monitor)
    print(f'Early stopping enabled: monitor={monitor}, patience={stopper.patience}, '
          f'min_delta={stopper.min_delta}, restore_best={restore_best}')
    return stopper, monitor, restore_best


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_one_epoch(model, weak_loader, strong_loader, anomaly_inj, contextual_loss,
                    optimizer, scheduler, pred_loss_func, options, device, epoch):
    """Run a single epoch and return its averaged losses."""
    i, epoch_total_loss, epoch_contextual_loss, epoch_predictive_loss = 0, 0., 0., 0.
    alaph, ema_con, ema_pre = 0.98, 10., 10.
    epoch_time = time.time()

    for (weak_data, _), (strong_data, _) in zip(weak_loader, strong_loader):
        optimizer.zero_grad()

        # anomaly injection -> return data and anomaly label
        if options.ablation != 'ano':
            strong_data_anomaly, weak_data_anomaly, true_anomaly_label = \
                anomaly_inj.anomaly_injection(weak_data, strong_data)
            true_anomaly_label = true_anomaly_label.to(device)
        else:
            if i == 0:
                print('without anomaly injection')
            strong_data_anomaly, weak_data_anomaly, true_anomaly_label = \
                strong_data, weak_data, None

        strong_data_anomaly = strong_data_anomaly.to(device)
        weak_data_anomaly = weak_data_anomaly.to(device)

        strong_predictive, weak_predictive, strong_contextual, weak_contextual = \
            model(strong_data_anomaly, weak_data_anomaly)

        strong_predictive = strong_predictive.view(options.batch_size, -1)
        weak_predictive = weak_predictive.view(options.batch_size, -1)

        # predictive loss
        predictive_loss = torch.tensor(0.)
        if pred_loss_func is not None:
            strong_predictive = F.sigmoid(strong_predictive)
            weak_predictive = F.sigmoid(weak_predictive)
            predictive_loss = (pred_loss_func(strong_predictive, true_anomaly_label) +
                               pred_loss_func(weak_predictive, true_anomaly_label)) / 2

        # contextual loss
        con_loss = torch.tensor(0.)
        if options.ablation != 'con':
            strong_contextual = strong_contextual.to(device)
            weak_contextual = weak_contextual.to(device)
            con_loss = contextual_loss(strong_contextual, weak_contextual)

        ema_con = alaph * ema_con + (1 - alaph) * con_loss.detach()
        ema_pre = alaph * ema_pre + (1 - alaph) * predictive_loss.detach()
        total_loss = (options.pre_loss_weight * predictive_loss / ema_pre +
                      options.con_loss_weight * con_loss / ema_con)

        # detach when accumulating so one epoch does not keep its whole graph alive
        epoch_predictive_loss += predictive_loss.detach()
        epoch_contextual_loss += con_loss.detach()
        epoch_total_loss += total_loss.detach()

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)
        optimizer.step()
        i += 1
        scheduler.step_update(i)

    if i == 0:
        raise RuntimeError(
            f'Epoch {epoch}: no training batches were produced; check that '
            f'win_size/step/batch_size fit the dataset size')

    metrics = {
        'total_loss': epoch_total_loss / i,
        'predictive_loss': epoch_predictive_loss / i,
        'contextual_loss': epoch_contextual_loss / i,
    }
    print(f'Epoch {epoch}, Cost time {(time.time() - epoch_time):.4f}s \t'
          f'predictive loss: {metrics["predictive_loss"]:.4f}, '
          f'contextual loss: {metrics["contextual_loss"]:.4f}, '
          f'total loss: {metrics["total_loss"]:.4f}')
    return metrics


def run_training(model, loaders, contextual_loss, optimizer, scheduler,
                 pred_loss_func, options, device, log_dir):
    weak_loader, strong_loader, anomaly_inj = loaders
    early_stopper, monitor, restore_best = build_early_stopper(options)
    best_state_path = os.path.join(log_dir, 'best_state_dict.pt')
    state_dir = os.path.join(log_dir, 'state')

    print('start training...')
    time_start = time.time()
    epochs_run, stopped_early = 0, False

    for epoch in range(options.num_epochs):
        metrics = train_one_epoch(
            model, weak_loader, strong_loader, anomaly_inj, contextual_loss,
            optimizer, scheduler, pred_loss_func, options, device, epoch)
        epochs_run = epoch + 1
        torch.save(model.state_dict(),
                   os.path.join(state_dir, f'state_dict_epoch_{epoch}.pt'))

        # early stopping: monitor the selected averaged epoch metric
        if early_stopper is not None:
            value = float(metrics[monitor])
            if early_stopper.step(value, epoch):
                torch.save(model.state_dict(), best_state_path)
                print(f'  {monitor}: {value:.6f} (best), saved to {best_state_path}')
            elif early_stopper.should_stop:
                stopped_early = True
                print(f'Early stopping at epoch {epoch}: {monitor} has not improved '
                      f'for {early_stopper.num_bad_epochs} epochs (best '
                      f'{early_stopper.best_score:.6f} at epoch '
                      f'{early_stopper.best_epoch})')
                break
            else:
                print(f'  {monitor}: {value:.6f} (no improvement for '
                      f'{early_stopper.num_bad_epochs}/{early_stopper.patience} epochs)')

    training_time = round(time.time() - time_start, 4)
    if epochs_run > 0:
        print(f'Training finished, cost time {training_time}s, '
              f'{training_time / epochs_run:.4f} per epoch')
    else:
        print(f'Training finished, cost time {training_time}s')

    if early_stopper is not None:
        summary = {
            'monitor': monitor,
            'patience': early_stopper.patience,
            'min_delta': early_stopper.min_delta,
            'best_score': (early_stopper.best_score
                           if math.isfinite(early_stopper.best_score) else None),
            'best_epoch': early_stopper.best_epoch,
            'epochs_run': epochs_run,
            'stopped_early': stopped_early,
        }
        with open(os.path.join(log_dir, 'early_stopping.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Early stopping summary: {summary}')
        if restore_best and early_stopper.best_epoch >= 0 and os.path.exists(best_state_path):
            model.load_state_dict(torch.load(best_state_path, map_location=device))
            print(f'Restored best model (epoch {early_stopper.best_epoch}, '
                  f'{monitor}={early_stopper.best_score:.6f}) from {best_state_path}')

    torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))
    print(f'Model state_dict saved in {os.path.join(log_dir, "state_dict.pt")}')


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(options):
    if options.ablation is not None and options.ablation == 'ano':
        options.pre_loss_weight = 0.0
        options.con_loss_weight = 1.0
    elif options.ablation is not None and options.ablation == 'con':
        options.pre_loss_weight = 1.0
        options.con_loss_weight = 0.0

    # win_size is the number of patches in a window; patch_size is the number
    # of data points embedded at once, so the raw sequence length is their product.
    data_seq_len = options.win_size * options.patch_size
    device = resolve_device(options)

    weak_loader, strong_loader, anomaly_inj, num_features = \
        build_data_loaders(options, data_seq_len)

    model = build_model(options, num_features, device)
    contextual_loss = ContextualLoss(options.win_size, options.temperature,
                                     options.cosine_similarity, device=device)
    contextual_loss.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=options.lr,
                                  weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=options.num_epochs,
                                  lr_min=options.lr * 0.01,
                                  warmup_lr_init=options.lr * 0.001,
                                  warmup_t=options.num_epochs // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False)

    if options.checkpoint is not None:
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'))

    log_dir = build_run_dir(options)
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)
    torch.save(model, os.path.join(log_dir, 'DAT_model.pt'))
    print(f'Run directory: {log_dir}')

    run_training(
        model, (weak_loader, strong_loader, anomaly_inj), contextual_loss,
        optimizer, scheduler, build_pred_loss_func(options), options, device, log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pre_loss_fuc', type=str, default='BCE',
                        help='the loss function for preditive and truth, BCE(default)/CE/MSE/MAE/KL',
                        choices=['KL', 'MSE', 'MAE', 'CE', 'BCE'])
    parser.add_argument('--n_layers', type=int, default=4, help='the number of transformer encoder in each branch, 4(default)')
    parser.add_argument('--n_heads', type=int, default=8, help='the number of attention head in one transformer encoder, 8(default)')
    parser.add_argument('--dropout', type=float, default=0.2, help='the rate of dropout in transformer encoder (MLP)')

    parser.add_argument('--d_embed', type=int, default=256, help='the dimension of embed input in transformer encoder, 256(Default)')
    parser.add_argument('--d_con_rate', type=float, default=2., help='the rate to embedded in contextual projection')
    parser.add_argument('--d_pre_rate', type=float, default=3., help='the rate to embedded in predictive projection')

    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use one gpu')
    parser.add_argument('--gpu_id', type=int, default=0, help='the index of this gpu used')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers to load data')

    parser.add_argument('--index', type=int, default=120, help='the begin index of input data')
    parser.add_argument('--patch_size', type=int, default=4, help='the number of data point in a patch to embed once')
    parser.add_argument('--batch_size', type=int, default=64, help=' , 64(Default)')
    parser.add_argument('--win_size', type=int, default=512, help='the length of subsequence of input data, 512(Default)')
    parser.add_argument('--step', type=int, default=5, help='the step of sliding window, 1(default)')

    # !!
    parser.add_argument('--num_epochs', type=int, default=3, help='the number of epoch to train, 20(defualt)')
    parser.add_argument('--dataset', type=str, default='PSM', choices=['MSL', 'PSM', 'SMAP', 'SMD', 'SWAT'], help='the dataset to train, PSM(default)/MSL/SMAP/SMD/SWAT')
    parser.add_argument('--data_path', type=str, default='data/PSM', help='the input data path')

    parser.add_argument('--anomaly_rate', type=float, default=0.4, help='the anomaly rate for anomaly-injection, 0.05(Default)')
    parser.add_argument('--blended_substitution', type=float, default=0.5, help='probability for blended substitution')
    parser.add_argument('--static_override', type=float, default=0.15, help='probability for static override')
    parser.add_argument('--peak_implantation', type=float, default=0.25, help='probability for spike implantation')
    parser.add_argument("--flip_replacing_interval", default='all', type=str,
                        help='allowance for random flipping in soft replacement; vertical/horizontal/all/none')
    parser.add_argument("--replacing_weight", default=0.7, type=float, help='weight for external interval in blended substitution')

    parser.add_argument('--w_jittering', type=float, default=0.5, help='the weight of jittering in weak data augmentation, 0.5(default)')
    parser.add_argument('--scaling', type=float, default=0.5, help='the weight of scaling in weak data augment, 0.5(default)')
    parser.add_argument('--s_jittering', type=float, default=0.3, help='the weight of jittering in strong data augmentation, 0.3(default)')
    parser.add_argument('--time_wrapping', type=float, default=0.6, help='the weight of time wrapping in strong data augmentation, 0.6(default)')
    parser.add_argument('--permutation', type=float, default=0.1, help='the weight of permutation in strong data augmentation, 0.1(default)')

    parser.add_argument('--pre_loss_weight', type=float, default=0.8, help='the weight of the loss of the prediction, lambda')
    parser.add_argument('--con_loss_weight', type=float, default=0.2, help='the weight of the loss of the contextual contrastion')
    parser.add_argument('--temperature', type=float, default=0.2, help='the temperature in contextual contrastive loss')
    parser.add_argument("--cosine_similarity", default=True, type=bool, help='whether to use cosine similarity in contextual contrastive loss')

    parser.add_argument('--ablation', type=str, default=None, help='change some part of model to do ablation experiment, '
                                                                   'None(default)/aug/ano/con, aug means no data augment, '
                                                                   'ano means no anomaly injection and predictive loss, '
                                                                   'con means no contextual contrast loss')
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", default=None, type=str, help='load checkpoint file')
    parser.add_argument('--default_options', type=str, default=None, help='some default options save for train; None(default)/SMAP/MSL/SMD/SWaT/WADI')

    parser.add_argument('--exp_id', type=str, default='1',
                        help='experiment name/id; a timestamp suffix is appended to the run directory')

    # early stopping
    parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction, default=True,
                        help='enable early stopping on the averaged epoch loss; use --no-early_stopping to disable')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='stop after this many consecutive epochs without improvement, 15(default)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='minimum decrease of the monitored loss that counts as an improvement, 1e-4(default)')
    parser.add_argument('--early_stop_monitor', type=str, default='total_loss', choices=list(EPOCH_METRICS),
                        help='averaged epoch loss to monitor, total_loss(default)/predictive_loss/contextual_loss')
    parser.add_argument('--restore_best', action=argparse.BooleanOptionalAction, default=True,
                        help='restore the best epoch weights before saving state_dict.pt')

    parser.add_argument('-r', type=float, default=0.0, help='contamination rate of training data')

    options = parser.parse_args()
    if options.default_options is not None:
        if options.default_options.startswith('SMD'):
            default_options = options.default_options
            options = torch.load('data/default_options_SMD.pt')
            options.dataset = default_options
        else:
            options = torch.load('data/default_options_' + options.default_options + '.pt')

    main(options)
