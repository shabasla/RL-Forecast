"""Neural network architectures for the SAC agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMNetwork(nn.Module):
    """CNN-LSTM network for processing sequential market data.
    
    Combines convolutional layers for local pattern extraction with
    LSTM layers for temporal sequence modeling.
    
    Args:
        state_dim: Dimension of input features
        action_dim: Dimension of output actions
        sequence_length: Length of input sequences
        hidden_dim: Hidden dimension for LSTM and FC layers
    """
    
    def __init__(self, state_dim: int, action_dim: int, sequence_length: int = 15, hidden_dim: int = 128):
        super(CNNLSTMNetwork, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(state_dim, 32, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, sequence_length])
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, sequence_length])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch, sequence, features]
            
        Returns:
            Output tensor of shape [batch, action_dim]
        """
        # x shape: [batch, sequence, features]
        x = x.permute(0, 2, 1)  # [batch, features, sequence]
        
        # Convolutional feature extraction
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        
        x = x.permute(0, 2, 1)  # [batch, sequence, channels]
        
        # LSTM temporal modeling
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last LSTM output
        
        # Final fully connected layers
        x = F.relu(self.ln3(self.fc1(x)))
        return self.fc2(x)