"""Training pipeline for single-step and multi-step unrolled world models."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from driving_wm.data.dataset import (
    DrivingDataset,
    Normalizer,
    compute_normalization_stats,
)
from driving_wm.data.state import ACTION_DIM, STATE_DIM, TARGET_DIM
from driving_wm.models.dynamics import WorldModel
from driving_wm.utils.log import setup_logger
from driving_wm.utils.seeding import seed_everything

logger = setup_logger("models.train")


def get_git_commit() -> str:
    """Retrieve current short git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train_epoch_one_step(
    model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch with single-step target prediction loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        norm_s = batch["norm_state"].to(device)
        a = batch["action"].to(device)
        norm_target = batch["norm_target"].to(device)

        optimizer.zero_grad()
        pred_norm_target = model(norm_s, a)
        loss = criterion(pred_norm_target, norm_target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def train_epoch_multistep(
    model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    state_norm: Normalizer,
    target_norm: Normalizer,
    device: torch.device,
) -> float:
    """Train for one epoch unrolling K steps through predictions."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # states: (B, K, 4), actions: (B, K, 2), targets: (B, K, 7)
        states = batch["state"].to(device)
        actions = batch["action"].to(device)
        targets = batch["target"].to(device)

        B, K, _ = actions.shape
        optimizer.zero_grad()

        curr_state = states[:, 0, :]
        step_losses = []

        for k in range(K):
            action_k = actions[:, k, :]
            gt_target_k = targets[:, k, :]
            gt_norm_target_k = target_norm.normalize(gt_target_k)

            norm_curr_s = state_norm.normalize(curr_state)
            pred_norm_target_k = model(norm_curr_s, action_k)

            step_loss = criterion(pred_norm_target_k, gt_norm_target_k)
            step_losses.append(step_loss)

            # Autoregressive next state step for rollout
            pred_target_k = target_norm.denormalize(pred_norm_target_k)
            d_state = pred_target_k[:, : model.state_dim]
            next_state = curr_state + d_state
            v_long = torch.clamp(next_state[:, 0:1], min=0.0)
            steer = torch.clamp(next_state[:, 3:4], min=-1.0, max=1.0)
            curr_state = torch.cat([v_long, next_state[:, 1:3], steer], dim=-1)

        loss = torch.stack(step_losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_loss(
    model: WorldModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute validation loss under eval mode."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            norm_s = batch["norm_state"].to(device)
            a = batch["action"].to(device)
            norm_target = batch["norm_target"].to(device)

            pred = model(norm_s, a)
            loss = criterion(pred, norm_target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def check_overfit_one_batch(
    train_npz: str | Path,
    device: torch.device,
    steps: int = 150,
) -> float:
    """Sanity check: verify that a single mini-batch can be overfit to near-zero loss."""
    logger.info("Running overfit-one-batch sanity test...")
    state_norm, target_norm = compute_normalization_stats(train_npz)
    ds = DrivingDataset(
        train_npz, unroll_steps=1, state_normalizer=state_norm, target_normalizer=target_norm
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))

    model = WorldModel(hidden_dim=128, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.HuberLoss()

    norm_s = batch["norm_state"].to(device)
    a = batch["action"].to(device)
    norm_y = batch["norm_target"].to(device)

    initial_loss = float("inf")
    final_loss = float("inf")

    for i in range(steps):
        model.train()
        optimizer.zero_grad()
        pred = model(norm_s, a)
        loss = criterion(pred, norm_y)
        loss.backward()
        optimizer.step()
        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    logger.info(f"Overfit test: initial loss = {initial_loss:.5f}, final loss = {final_loss:.5f}")
    assert final_loss < 0.05, f"Overfit sanity test failed: final loss {final_loss} >= 0.05"
    return final_loss


def train_model(
    train_npz: str | Path,
    val_npz: str | Path,
    config: dict[str, Any],
    checkpoint_dir: str | Path = "checkpoints",
    multistep_ablation: bool = False,
    plot_path: str | Path | None = None,
) -> Path:
    """Execute complete training loop and save checkpoint with metadata."""
    seed = config.get("seed", 42)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = config.get("model", {})
    hidden_dim = m_cfg.get("hidden_dim", 256)
    num_layers = m_cfg.get("num_layers", 3)
    activation = m_cfg.get("activation", "silu")
    lr = m_cfg.get("lr", 1e-3)
    weight_decay = m_cfg.get("weight_decay", 1e-4)
    batch_size = m_cfg.get("batch_size", 256)
    epochs = m_cfg.get("epochs", 30)
    unroll_steps = m_cfg.get("unroll_steps", 5) if multistep_ablation else 1
    patience = m_cfg.get("patience", 7)

    # 1. Compute normalization statistics strictly on training data
    state_norm, target_norm = compute_normalization_stats(train_npz)

    # 2. Build datasets
    train_ds = DrivingDataset(
        train_npz,
        unroll_steps=unroll_steps,
        state_normalizer=state_norm,
        target_normalizer=target_norm,
    )
    val_ds = DrivingDataset(
        val_npz,
        unroll_steps=1,
        state_normalizer=state_norm,
        target_normalizer=target_norm,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = WorldModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_weights: dict[str, Any] | None = None
    epochs_no_improve = 0

    suffix = "multistep" if multistep_ablation else "onestep"
    logger.info(
        f"Starting training ({suffix}) on {device}: {len(train_ds)} train transitions, "
        f"{len(val_ds)} val transitions, {epochs} epochs max."
    )

    start_time = time.time()
    for ep in range(epochs):
        if multistep_ablation:
            t_loss = train_epoch_multistep(
                model, train_loader, optimizer, criterion, state_norm, target_norm, device
            )
        else:
            t_loss = train_epoch_one_step(model, train_loader, optimizer, criterion, device)

        v_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            logger.info(
                f"Epoch {ep + 1:02d}/{epochs:02d} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f} | Best: {best_val_loss:.5f}"
            )

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {ep + 1} (patience={patience}).")
            break

    elapsed = time.time() - start_time
    logger.info(f"Training finished in {elapsed:.1f} s. Best Val Loss: {best_val_loss:.5f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save checkpoint
    ckpt_path = checkpoint_dir / f"world_model_{suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_norm": state_norm.to_dict(),
            "target_norm": target_norm.to_dict(),
            "model_config": {
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "target_dim": TARGET_DIM,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "activation": activation,
                "multistep": multistep_ablation,
            },
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "commit_hash": get_git_commit(),
            "elapsed_seconds": elapsed,
        },
        ckpt_path,
    )
    logger.info(f"Saved checkpoint to {ckpt_path} ({ckpt_path.stat().st_size / 1024:.1f} KB)")

    # Plot loss curves
    if plot_path is not None:
        p_path = Path(plot_path)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(train_losses, label="Train Loss (Huber)", lw=1.8)
        plt.plot(val_losses, label="Validation Loss (Huber)", lw=1.8)
        plt.title(f"World Model Training & Validation Loss ({suffix})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(p_path, dpi=150)
        plt.close()
        logger.info(f"Saved loss plot to {p_path}")

    return ckpt_path
