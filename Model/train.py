"""
Training Script for SUB3_V2 with W&B Integration

Features:
- GPU support (CUDA)
- Weights & Biases experiment tracking
- Masked loss (V2 improvement)
- Early stopping
- Learning rate scheduling
- Checkpoint saving

Author: Riccardo
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import wandb
from datetime import datetime

from lstm import HeartRateLSTM_V2
from loss import MaskedMSELoss, MaskedMAELoss, compute_masked_mae, compute_masked_rmse


def get_device(force_cpu=False):
    """Get the best available device (GPU or CPU)."""
    if force_cpu:
        return torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("! GPU not available, using CPU")

    return device


def load_data(data_dir, device):
    """Load preprocessed tensors."""
    print(f"\nLoading data from {data_dir}...")

    train_data = torch.load(f'{data_dir}/train.pt', map_location=device)
    val_data = torch.load(f'{data_dir}/val.pt', map_location=device)

    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"  Train: {train_data['features'].shape[0]} samples")
    print(f"  Val: {val_data['features'].shape[0]} samples")

    return train_data, val_data, metadata


def create_dataloaders(train_data, val_data, batch_size, num_workers=4):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(
        train_data['features'],
        train_data['heart_rate'],
        train_data['mask'],
        train_data['original_lengths']
    )

    val_dataset = TensorDataset(
        val_data['features'],
        val_data['heart_rate'],
        val_data['mask'],
        val_data['original_lengths']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for features, heart_rate, mask, lengths in pbar:
        # Data already on device from loading

        # Forward pass
        predictions = model(features)

        # Compute loss (masked)
        loss = criterion(predictions, heart_rate, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Metrics
        with torch.no_grad():
            mae = compute_masked_mae(predictions, heart_rate, mask)

        total_loss += loss.item()
        total_mae += mae
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae:.2f}'
        })

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches

    return avg_loss, avg_mae


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation {epoch}")

        for features, heart_rate, mask, lengths in pbar:
            # Data already on device from loading

            # Forward pass
            predictions = model(features)

            # Compute loss
            loss = criterion(predictions, heart_rate, mask)

            # Metrics
            mae = compute_masked_mae(predictions, heart_rate, mask)
            rmse = compute_masked_rmse(predictions, heart_rate, mask)

            total_loss += loss.item()
            total_mae += mae
            total_rmse += rmse
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae:.2f}'
            })

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches

    return avg_loss, avg_mae, avg_rmse


def save_checkpoint(model, optimizer, epoch, val_mae, filepath, metadata=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mae': val_mae,
        'model_config': model.get_model_info(),
        'metadata': metadata
    }
    torch.save(checkpoint, filepath)
    print(f"  ✓ Saved checkpoint to {filepath}")


def train(args):
    """Main training loop."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device
    device = get_device(force_cpu=args.cpu)

    # Initialize W&B
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name if args.run_name else None,
            config={
                'input_size': 11,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'bidirectional': args.bidirectional,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'epochs': args.epochs,
                'early_stop_patience': args.patience,
                'loss_function': args.loss_fn,
                'device': str(device),
                'seed': args.seed
            }
        )
        print(f"\n✓ W&B initialized: {wandb.run.name}")

    # Load data
    train_data, val_data, metadata = load_data(args.data_dir, device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, args.batch_size, num_workers=args.num_workers
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    model = HeartRateLSTM_V2(
        input_size=11,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    ).to(device)

    model_info = model.get_model_info()
    print(f"\nModel created:")
    print(f"  Parameters: {model_info['num_parameters']:,}")
    print(f"  Trainable: {model_info['num_trainable_parameters']:,}")

    # Loss function
    if args.loss_fn == 'mse':
        criterion = MaskedMSELoss()
    elif args.loss_fn == 'mae':
        criterion = MaskedMAELoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    print(f"  Loss: {args.loss_fn.upper()}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.scheduler_patience,
        verbose=True
    )

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_mae, val_rmse = validate_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Learning rate scheduling
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} BPM")
        print(f"  Val Loss:   {val_loss:.4f} | Val MAE:   {val_mae:.2f} BPM | Val RMSE: {val_rmse:.2f} BPM")
        print(f"  LR: {current_lr:.6f}")

        # Log to W&B
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/mae': train_mae,
                'val/loss': val_loss,
                'val/mae': val_mae,
                'val/rmse': val_rmse,
                'lr': current_lr
            })

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0

            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(
                model, optimizer, epoch, val_mae, checkpoint_path,
                metadata={'training_args': vars(args), 'data_metadata': metadata}
            )

            print(f"  🌟 New best model! Val MAE: {val_mae:.2f} BPM")

            # Log to W&B
            if not args.no_wandb:
                wandb.run.summary['best_val_mae'] = val_mae
                wandb.run.summary['best_epoch'] = epoch
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        # Save latest model
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, val_mae, checkpoint_path)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹ Early stopping triggered after {epoch} epochs")
            print(f"  Best Val MAE: {best_val_mae:.2f} BPM")
            break

    # Training complete
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val MAE: {best_val_mae:.2f} BPM")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/best_model.pt")

    # Finish W&B
    if not args.no_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train SUB3_V2 Heart Rate Prediction Model')

    # Data
    parser.add_argument('--data-dir', default='DATA/processed', help='Data directory')

    # Model
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss-fn', default='mse', choices=['mse', 'mae'], help='Loss function')

    # Optimization
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--scheduler-patience', type=int, default=5, help='LR scheduler patience')

    # System
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Checkpoints
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')

    # W&B
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--wandb-project', default='heart-rate-prediction', help='W&B project name')
    parser.add_argument('--run-name', default=None, help='W&B run name')

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("SUB3_V2 Training Configuration")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model: LSTM (hidden={args.hidden_size}, layers={args.num_layers}, dropout={args.dropout})")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Loss: {args.loss_fn.upper()}")
    print(f"Device: {'CPU (forced)' if args.cpu else 'Auto-detect (GPU preferred)'}")
    print(f"W&B: {'Disabled' if args.no_wandb else f'Enabled (project: {args.wandb_project})'}")
    print("=" * 60)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
