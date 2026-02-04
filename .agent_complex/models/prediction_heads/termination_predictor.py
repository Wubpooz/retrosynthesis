"""Termination prediction for multi-step retrosynthesis."""

import torch
import torch.nn as nn


class TerminationPredictor(nn.Module):
    """
    Predicts whether the retrosynthesis process should terminate.
    Binary classification: 0 = continue, 1 = stop
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
            logits: [batch_size, 2] (continue vs stop)
        """
        return self.classifier(graph_features)
