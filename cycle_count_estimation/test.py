#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import sys
import glob
import os
import random
sys.path.append('..')

from utils.ImprovedResNet import ImprovedResNet

import matplotlib.pyplot as plt


def top_k_percent_mean(x, k=5):
    """Mean of top k% largest values in 1D numpy array."""
    if len(x) == 0:
        return np.nan
    thresh = np.percentile(x, 100 - k)
    return x[x >= thresh].mean()

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load normalization constants
# -------------------------------
mu = torch.tensor(
    np.loadtxt("norm_const/mu.txt", delimiter=','),
    dtype=torch.float32
)
std = torch.tensor(
    np.loadtxt("norm_const/std.txt", delimiter=','),
    dtype=torch.float32
)

mu_y = torch.tensor(
    np.loadtxt("norm_const/output_mu.txt", delimiter=','),
    dtype=torch.float32,
    device=device
)
std_y = torch.tensor(
    np.loadtxt("norm_const/output_std.txt", delimiter=','),
    dtype=torch.float32,
    device=device
)

# -------------------------------
# Benchmarks
# -------------------------------
BENCHMARKS = (
    "adpcm aes blowfish core crc32 edn gsm huffbench jpeg md5 mips motion "
    "nettle-aes nsichneu picojpeg primecount qrduino sglib-combined sha "
    "slre statemate tarfind ud wikisort matmult-int"
)
benchs = BENCHMARKS.split()

# -------------------------------
# Load benchmark evaluation data
# -------------------------------
all_bench_eval = {}
for bench in benchs:
    data = np.load(f"bench_val_data/{bench}.npy").astype(np.float32)
    inp = torch.tensor(data[:, :-1])
    out = torch.tensor(data[:, -1:])

    inp = (inp - mu) / std
    all_bench_eval[bench] = (inp, out)
    print(f"{bench:<15} {data.shape}")


# -------------------------------
# Fine-tuning
# -------------------------------
def hot_train_models(models, program_name, fine_tune_data_amount=100):
    epochs = 100
    n = len(models)
    lr_list = np.linspace(0.0002, 0.0004, n)
    wd_list = np.linspace(0.00003, 0.00005, n)

    np.random.seed(42)
    lr_values = np.random.choice(lr_list, size=n, replace=False).tolist()
    wd_values = np.random.choice(wd_list, size=n, replace=False).tolist()

    hot_start_array = np.load(f"bench_fine_tune_data/{program_name}.npy").astype(np.float32)
    hot_start_array = hot_start_array[hot_start_array[:, -1] <= 1]
    hot_start_array = hot_start_array[hot_start_array[:, -1] > 0.01]
    hot_start_array = hot_start_array[:fine_tune_data_amount]

    inp_raw = torch.tensor(hot_start_array[:, :-1])
    out_raw = torch.tensor(hot_start_array[:, -1:], device=device)

    inp_norm = (inp_raw - mu) / std
    out_norm = (out_raw - mu_y) / std_y

    dataset = TensorDataset(inp_norm, out_norm.cpu())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.MSELoss()

    for model_it, model in enumerate(models):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr_values[model_it] / 100,
            weight_decay=wd_values[model_it],
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        model.train()
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x).view(-1)
                loss = criterion(pred, y.view(-1))
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()

    return models


# -------------------------------
# Evaluation
# -------------------------------
def eval_models(n_models, hot_train=False):
    model_dir = "models"
    all_ckpts = glob.glob(os.path.join(model_dir, "*.pth"))

    all_bench_errors = []
    all_bench_var = []
    all_gt = []
    all_pred = []
    mses = []

    for i, bench in enumerate(benchs):
        ckpts = random.sample(all_ckpts, n_models)

        ensemble_models = []
        for ckpt in ckpts:
            model = ImprovedResNet(
                input_dim=722,
                features=350,
                depth=2,
                num_outputs=1,
                dropout=0.00,
                activation="silu"
            ).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            model.eval()
            ensemble_models.append(model)

        if hot_train:
            ensemble_models = hot_train_models(ensemble_models, bench)

        bt_inp, bt_out = all_bench_eval[bench]
        bt_inp = bt_inp.to(device)
        bt_out = bt_out.to(device).view(-1)

        with torch.no_grad():
            mu_preds = []
            for model in ensemble_models:
                mu_preds.append(model(bt_inp).view(-1))

            mu_preds = torch.stack(mu_preds)
            mu_ensemble = mu_preds.mean(dim=0)
            mu_ensemble = mu_ensemble * std_y + mu_y

            err = torch.abs(mu_ensemble - bt_out) / torch.clamp(bt_out.abs(), min=1e-9)

            mses.append(torch.mean((mu_ensemble - bt_out) ** 2, dim=0).cpu().numpy())
            all_bench_errors.extend(err.cpu().numpy())
            all_bench_var.extend(torch.std(mu_preds, dim=0).cpu().numpy())
            all_gt.append(bt_out.cpu().numpy())
            all_pred.append(mu_ensemble.cpu().numpy())

        err_np = err.cpu().numpy()
        top5_err = top_k_percent_mean(err_np, k=5)
        print(f"{bench:<15} | Avg Error: {err.mean():.6f} | Top5%: {top5_err:.6f}")

    all_bench_errors = np.array(all_bench_errors)
    mse = float(np.mean(mses))
    top5_err = top_k_percent_mean(all_bench_errors, k=1)
    label = "Fine-tuned" if hot_train else "Baseline"
    print(
        f"\n{label} Ensemble ({n_models} models) | "
        f"MSE: {mse:.6f} | "
        f"Top1%: {top5_err:.6f} | "
        f"Avg MAE: {all_bench_errors.mean():.6f}"
    )

    return all_gt, all_pred


n_models = 2

#all_gt_base, all_pred_base = eval_models(n_models, hot_train=False)

all_gt_ft, all_pred_ft = eval_models(n_models, hot_train=False)

