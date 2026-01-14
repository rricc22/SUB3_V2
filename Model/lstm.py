"""
LSTM Model for Heart Rate Prediction (SUB3_V2)

Architecture:
- Input: [batch, seq_len=500, num_features=14]
  Features: speed, altitude, gender, 8 temporal features, 3 workout type one-hot
- LSTM: 2 layers, configurable hidden size, dropout
- Output: [batch, seq_len=500, 1] (unnormalized HR in BPM)

Author: Riccardo
Date: 2026-01-13
Updated: 2026-01-14 - 14 input features (added workout type one-hot)
"""

import torch
import torch.nn as nn


class HeartRateLSTM_V2(nn.Module):
    """
    LSTM model for heart rate prediction from running data.

    Uses 14 input features:
    - Base (3): speed, altitude, gender
    - Temporal (8): lag, derivative, rolling, cumulative features
    - Workout type (3): is_recovery, is_steady, is_intensive (one-hot)

    Outputs heart rate predictions for each timestep.
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Initialize the LSTM model.

        Args:
            input_size: Number of input features (default: 14)
            hidden_size: Number of hidden units in LSTM (default: 128)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate between LSTM layers (default: 0.3)
            bidirectional: Use bidirectional LSTM (default: False)
        """
        super(HeartRateLSTM_V2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output projection
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)

    def forward(self, x, lengths=None):
        """
        Forward pass.

        Args:
            x: Input features [batch, seq_len, input_size]
            lengths: Original sequence lengths (optional, for packed sequences)

        Returns:
            Heart rate predictions [batch, seq_len, 1]
        """
        batch_size = x.size(0)

        # LSTM forward pass
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size * num_directions]

        # Project to output
        output = self.fc(lstm_out)
        # output: [batch, seq_len, 1]

        return output

    def get_model_info(self):
        """Get model architecture information."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


if __name__ == "__main__":
    # Test the model
    print("Testing HeartRateLSTM_V2 (14 features)...")

    # Create model
    model = HeartRateLSTM_V2(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False
    )

    print(f"\nModel architecture:")
    print(model)

    # Model info
    info = model.get_model_info()
    print(f"\nModel info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    batch_size = 4
    seq_len = 500
    input_size = 14

    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")

    # Check output range (should be reasonable HR values after training)
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    print("\n✓ Model test passed!")
