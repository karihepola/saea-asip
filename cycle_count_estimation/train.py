#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
from torch.utils.data import TensorDataset, DataLoader
sys.path.append('..')
from sklearn.model_selection import train_test_split
from utils.ImprovedResNet import ImprovedResNet

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Ensemble setup
# -------------------------------
ensemble_size = 2
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# -------------------------------
# Load data
# -------------------------------

#Increase if necessary, limit to 1.5M to speedup training. 2M was used for the paper
NUM_DATA=1_500_000
og_data = np.load("data/data_normalized.npy")[:NUM_DATA]
og_data = og_data.astype(np.float32)

# -------------------------------
# Input normalization
# -------------------------------
mu_x = torch.tensor(
    np.loadtxt("norm_const/mu.txt", delimiter=','),
    dtype=torch.float32,
    device=device
)
std_x = torch.tensor(
    np.loadtxt("norm_const/std.txt", delimiter=','),
    dtype=torch.float32,
    device=device
)

mu_y = float(np.loadtxt("norm_const/output_mu.txt", delimiter=","))
std_y = float(np.loadtxt("norm_const/output_std.txt", delimiter=","))

# -------------------------------
# Output normalization
# -------------------------------
train_outputs = og_data[:, -1]

# -------------------------------
# Benchmark evaluation sets
# -------------------------------
BENCHMARKS = (
    "adpcm aes blowfish core crc32 edn gsm huffbench jpeg md5 "
    "mips motion nettle-aes nsichneu picojpeg primecount qrduino "
    "sglib-combined sha slre statemate tarfind ud wikisort matmult-int"
)
benchs = BENCHMARKS.split()

all_bench_eval = {}

for bench in benchs:
    data = np.load(f"bench_val_data/{bench}.npy").astype(np.float32)
    data = data[:min(500, data.shape[0])]

    inp = (data[:, :-1] - mu_x.cpu().numpy()) / std_x.cpu().numpy()
    out = (data[:, -1] - mu_y) / std_y

    # Keep on CPU, move to GPU in batches during evaluation
    all_bench_eval[bench] = (
        torch.tensor(inp, dtype=torch.float32),
        torch.tensor(out[:, None], dtype=torch.float32),
    )

# -------------------------------
# Utility function for DataLoader
# -------------------------------
def get_loader(inputs, outputs, batch_size, shuffle=False):
    dataset = TensorDataset(inputs, outputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

# -------------------------------
# Train ensemble
# -------------------------------
for ens_id in range(ensemble_size):

    print("\n============================")
    print(f" Training ensemble model {ens_id+1}/{ensemble_size}")
    print("============================\n")

    # Reproducibility
    seed = random.randint(0, 999999)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data, val_data = train_test_split(
        og_data, test_size=0.05, random_state=seed
    )

    # Normalize inputs
    inp_train_cpu = torch.tensor(train_data[:, :-1], dtype=torch.float32)
    out_train_cpu = torch.tensor(((train_data[:, -1] - mu_y) / std_y)[:, None], dtype=torch.float32)
    inp_val_cpu = torch.tensor(val_data[:, :-1], dtype=torch.float32)
    out_val_cpu = torch.tensor(((val_data[:, -1] - mu_y) / std_y)[:, None], dtype=torch.float32)

    batch_size = 512
    train_loader = get_loader(inp_train_cpu, out_train_cpu, batch_size=batch_size, shuffle=True)
    val_loader = get_loader(inp_val_cpu, out_val_cpu, batch_size=1024, shuffle=False)

    # -------------------------------
    # Model
    # -------------------------------
    model = ImprovedResNet(
        input_dim=722,
        features=350,
        depth=2,
        num_outputs=1,
        dropout=0.05,
        activation="silu"
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr = 2e-4 * np.random.uniform(0.95, 1.05),
        weight_decay = 1e-3 * np.random.uniform(0.95, 1.05)
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    criterion = nn.MSELoss()
    epochs = 30

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        scheduler.step()

        # -------------------------------
        # Validation
        # -------------------------------
        val_loss = 0.0
        with torch.no_grad():
            # Validation
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                val_pred = model(x_val)
                val_loss += criterion(val_pred, y_val).item() * x_val.size(0)
            val_mse = val_loss / len(inp_val_cpu)

            # Benchmark evaluation
            errs, mses = [], []
            for bench in benchs:
                x_eval_cpu, y_eval_cpu = all_bench_eval[bench]
                x_eval_loader = get_loader(x_eval_cpu, y_eval_cpu, batch_size=1024, shuffle=False)

                bench_mae, bench_mse = 0.0, 0.0
                n_eval = 0
                for x_batch, y_batch in x_eval_loader:
                    x_batch = x_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)

                    pred_norm = model(x_batch)
                    pred = pred_norm * std_y + mu_y
                    truth = y_batch * std_y + mu_y

                    bench_mae += torch.sum(torch.abs(pred - truth) / truth.abs()).item()
                    bench_mse += torch.sum((pred - truth) ** 2).item()
                    n_eval += x_batch.size(0)

                errs.append(bench_mae / n_eval)
                mses.append(bench_mse / n_eval)



        print(
            f"[Model {ens_id+1}] Epoch {epoch}/{epochs} | "
            f"Loss: {total_loss/len(inp_train_cpu):.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Test MAE: {np.mean(errs):.6f} | "
            f"Test MSE: {np.mean(mses):.6f}"
        )

    # -------------------------------
    # Save model
    # -------------------------------
    model_path = f"{model_dir}/{seed}_ens{ens_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved → {model_path}")

    del model, optimizer, inp_train_cpu, out_train_cpu, inp_val_cpu, out_val_cpu
    torch.cuda.empty_cache()
