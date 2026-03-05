from src.data.dataset import GlobalLoadDataset
from src.models.lstm import LSTMModel
from src.training.trainer import train_one_epoch, validate
from src.training.early_stopping import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
        dropout=config.dropout
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        save_path=config.checkpoint_path
    )

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None
    best_val_loss = float("inf")
    verbose = (config.mode == "dev")

    for epoch in range(config.epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            verbose=verbose
        )

        val_loss = validate(model, val_loader, criterion, device,
                            verbose=verbose)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        best_val_loss = min(best_val_loss, val_loss)

        early_stopper.step(val_loss, model)
        scheduler.step()

        if early_stopper.early_stop:
            print("Early stopping triggered.")
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
        dropout=config.dropout
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None
    verbose = (config.mode == "dev")

    print(f"\n[Final Retraining] epochs={config.epochs}")

    for epoch in range(config.epochs):

        train_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, scaler,
            verbose=verbose
        )
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | Train MSE: {train_loss:.6f}")

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

    rmse = np.sqrt(np.mean(np.concatenate(all_squared_errors)))

    print(f"[Final Test RMSE]: {rmse:.4f}")
    print(f"Model saved to: {config.checkpoint_path}")

    return float(rmse)
