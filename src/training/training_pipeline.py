from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate
from src.training.early_stopping import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torch import amp


# ==========================================================
# Hyperparameter Search Phase (with Early Stopping)
# ==========================================================
def train_single_configuration(train_df, val_df, device, config):

    train_dataset = GlobalLoadDataset(train_df)
    val_dataset = GlobalLoadDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=getattr(config, 'num_layers', 1),
        dropout=config.dropout
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.search_epochs
    )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config.search_patience,
        min_delta=config.min_delta,
        save_path=config.checkpoint_path
    )

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None
    best_val_loss = float("inf")

    epoch_bar = tqdm(
        range(config.search_epochs),
        desc="  Search",
        unit="ep",
        ncols=90,
        leave=True,
    )

    for epoch in epoch_bar:

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
        )

        val_loss = validate(model, val_loader, criterion, device)

        best_val_loss = min(best_val_loss, val_loss)
        epoch_bar.set_postfix({"val": f"{val_loss:.5f}", "best": f"{best_val_loss:.5f}"})

        early_stopper.step(val_loss, model)
        scheduler.step()

        if early_stopper.early_stop:
            epoch_bar.set_description("  Search [stopped]")
            break

    return best_val_loss


# ==========================================================
# Final Retraining Phase (NO Early Stopping)
# ==========================================================
def retrain_and_evaluate(train_df, val_df, test_df, device,
                         config, scaling_params):

    combined_df = pd.concat([train_df, val_df], axis=0)

    dataset = GlobalLoadDataset(combined_df)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
    )

    model = LSTMModel(
        hidden_dim=config.hidden_dim,
        num_layers=getattr(config, 'num_layers', 1),
        dropout=config.dropout
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.retrain_epochs
    )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\n[Final Retraining] epochs={config.retrain_epochs}")

    epoch_bar = tqdm(
        range(config.retrain_epochs),
        desc="  Retrain",
        unit="ep",
        ncols=90,
        leave=True,
    )

    for epoch in epoch_bar:

        train_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, scaler,
        )
        scheduler.step()

        epoch_bar.set_postfix({"train": f"{train_loss:.5f}"})

    # Save final trained model
    torch.save(model.state_dict(), config.checkpoint_path)

    # -------------------------
    # Test Evaluation
    # -------------------------
    test_dataset = GlobalLoadDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    model.eval()
    all_squared_errors = []
    all_abs_errors = []
    all_target_values = []
    household_columns = train_df.columns.tolist()

    # Pre-compute scaling arrays once (avoids per-sample dict lookups)
    _mins = np.array([scaling_params[col]["min"] for col in household_columns])
    _maxs = np.array([scaling_params[col]["max"] for col in household_columns])

    with torch.no_grad():
        for x, y, household_idx in test_loader:

            x = x.to(device)

            outputs = model(x).cpu().numpy()   # (batch, 24)
            targets = y.numpy()                # (batch, 24)

            h_idx  = household_idx.numpy()            # (batch,)
            scale  = _maxs[h_idx] - _mins[h_idx]      # (batch,)
            min_b  = _mins[h_idx]                      # (batch,)

            pred_inv   = outputs * scale[:, None] + min_b[:, None]   # (batch, 24)
            target_inv = targets * scale[:, None] + min_b[:, None]   # (batch, 24)

            all_squared_errors.append((pred_inv - target_inv) ** 2)
            all_abs_errors.append(np.abs(pred_inv - target_inv))
            all_target_values.append(np.abs(target_inv))

    squared = np.concatenate(all_squared_errors)
    abs_err = np.concatenate(all_abs_errors)
    abs_tgt = np.concatenate(all_target_values)

    rmse = float(np.sqrt(np.mean(squared)))
    mae = float(np.mean(abs_err))
    mape = float(np.mean(abs_err / np.clip(abs_tgt, 1e-8, None)) * 100)

    print(f"[Final Test RMSE]: {rmse:.4f}  MAE: {mae:.4f}  MAPE: {mape:.2f}%")
    print(f"Model saved to: {config.checkpoint_path}")

    return {"rmse": rmse, "mae": mae, "mape": mape}
