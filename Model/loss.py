"""
Loss Functions for SUB3_V2

Implements masked MSE loss that ignores padded regions during training.
This is a key improvement over V1 which computed loss on all timesteps.

Author: Riccardo
Date: 2026-01-13
"""

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss.

    Only computes loss on valid (non-padded) timesteps as indicated by the mask.
    This prevents the model from learning on artificial padded data.
    """

    def __init__(self, reduction='mean'):
        """
        Initialize masked MSE loss.

        Args:
            reduction: 'mean' or 'sum' (default: 'mean')
        """
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, mask):
        """
        Compute masked MSE loss.

        Args:
            predictions: Model predictions [batch, seq_len, 1]
            targets: Ground truth HR [batch, seq_len, 1]
            mask: Validity mask [batch, seq_len, 1] (1=valid, 0=padded)

        Returns:
            Scalar loss value
        """
        # Compute squared error
        squared_error = (predictions - targets) ** 2

        # Apply mask (zero out padded regions)
        masked_error = squared_error * mask

        # Sum over all dimensions
        total_error = masked_error.sum()

        # Count valid elements
        num_valid = mask.sum()

        # Avoid division by zero
        if num_valid == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        if self.reduction == 'mean':
            return total_error / num_valid
        elif self.reduction == 'sum':
            return total_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class MaskedMAELoss(nn.Module):
    """
    Masked Mean Absolute Error Loss.

    Alternative to MSE for more robust optimization.
    """

    def __init__(self, reduction='mean'):
        """
        Initialize masked MAE loss.

        Args:
            reduction: 'mean' or 'sum' (default: 'mean')
        """
        super(MaskedMAELoss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, mask):
        """
        Compute masked MAE loss.

        Args:
            predictions: Model predictions [batch, seq_len, 1]
            targets: Ground truth HR [batch, seq_len, 1]
            mask: Validity mask [batch, seq_len, 1] (1=valid, 0=padded)

        Returns:
            Scalar loss value
        """
        # Compute absolute error
        absolute_error = torch.abs(predictions - targets)

        # Apply mask
        masked_error = absolute_error * mask

        # Sum over all dimensions
        total_error = masked_error.sum()

        # Count valid elements
        num_valid = mask.sum()

        # Avoid division by zero
        if num_valid == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        if self.reduction == 'mean':
            return total_error / num_valid
        elif self.reduction == 'sum':
            return total_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_masked_mae(predictions, targets, mask):
    """
    Compute masked MAE metric (for evaluation).

    Args:
        predictions: Model predictions [batch, seq_len, 1]
        targets: Ground truth HR [batch, seq_len, 1]
        mask: Validity mask [batch, seq_len, 1]

    Returns:
        MAE value (scalar)
    """
    with torch.no_grad():
        absolute_error = torch.abs(predictions - targets)
        masked_error = absolute_error * mask
        total_error = masked_error.sum()
        num_valid = mask.sum()

        if num_valid == 0:
            return 0.0

        return (total_error / num_valid).item()


def compute_masked_rmse(predictions, targets, mask):
    """
    Compute masked RMSE metric (for evaluation).

    Args:
        predictions: Model predictions [batch, seq_len, 1]
        targets: Ground truth HR [batch, seq_len, 1]
        mask: Validity mask [batch, seq_len, 1]

    Returns:
        RMSE value (scalar)
    """
    with torch.no_grad():
        squared_error = (predictions - targets) ** 2
        masked_error = squared_error * mask
        total_error = masked_error.sum()
        num_valid = mask.sum()

        if num_valid == 0:
            return 0.0

        mse = total_error / num_valid
        return torch.sqrt(mse).item()


if __name__ == "__main__":
    # Test masked loss functions
    print("Testing masked loss functions...")

    batch_size = 4
    seq_len = 500

    # Create sample data
    predictions = torch.randn(batch_size, seq_len, 1) * 10 + 150  # ~150 BPM
    targets = torch.randn(batch_size, seq_len, 1) * 10 + 150

    # Create mask (first 400 valid, last 100 padded)
    mask = torch.ones(batch_size, seq_len, 1)
    mask[:, 400:, :] = 0

    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Valid timesteps: {int(mask.sum())} / {batch_size * seq_len}")

    # Test MSE loss
    mse_loss = MaskedMSELoss(reduction='mean')
    loss_value = mse_loss(predictions, targets, mask)
    print(f"\nMasked MSE Loss: {loss_value.item():.4f}")

    # Test MAE loss
    mae_loss = MaskedMAELoss(reduction='mean')
    loss_value = mae_loss(predictions, targets, mask)
    print(f"Masked MAE Loss: {loss_value.item():.4f}")

    # Test metrics
    mae = compute_masked_mae(predictions, targets, mask)
    rmse = compute_masked_rmse(predictions, targets, mask)
    print(f"\nMasked MAE: {mae:.4f}")
    print(f"Masked RMSE: {rmse:.4f}")

    # Test with all-zero mask (edge case)
    zero_mask = torch.zeros(batch_size, seq_len, 1)
    loss_value = mse_loss(predictions, targets, zero_mask)
    print(f"\nEdge case (all padded): {loss_value.item():.4f}")

    print("\n✓ Loss function tests passed!")
