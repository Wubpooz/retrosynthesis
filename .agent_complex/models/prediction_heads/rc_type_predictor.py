"""Reaction center type prediction: atom vs bond."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCTypePredictor(nn.Module):
    """
    Binary classifier to predict reaction center type.
    0 = atom center, 1 = bond center
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)
        )
    
    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_features: [batch_size, hidden_dim]
            
        Returns:
            logits: [batch_size, 2]
        """
        return self.classifier(graph_features)
