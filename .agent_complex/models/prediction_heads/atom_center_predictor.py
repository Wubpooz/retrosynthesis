"""Atom-level reaction center localization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomCenterPredictor(nn.Module):
    """
    Predicts which atom(s) are reaction centers.
    Returns per-atom logits for multi-label classification.
    """
    
    def __init__(self, atom_dim: int):
        super().__init__()
        
        self.scorer = nn.Sequential(
            nn.Linear(atom_dim, atom_dim // 2),
            nn.GELU(),
            nn.Linear(atom_dim // 2, 1)
        )
    
    def forward(self, atom_features: torch.Tensor, batch: torch.Tensor) -> list:
        """
        Args:
            atom_features: [num_total_atoms, atom_dim]
            batch: [num_total_atoms] batch assignment
            
        Returns:
            List of [num_atoms_i] logit tensors, one per graph
        """
        # Compute per-atom scores
        scores = self.scorer(atom_features).squeeze(-1)  # [num_total_atoms]
        
        # Split by batch
        unique_batches = torch.unique(batch, sorted=True)
        outputs = []
        
        for b in unique_batches:
            mask = batch == b
            outputs.append(scores[mask])
        
        return outputs
    
    def predict_top_k(self, atom_features: torch.Tensor, batch: torch.Tensor, k: int = 5) -> list:
        """
        Predict top-k reaction center atoms.
        
        Args:
            atom_features: [num_total_atoms, atom_dim]
            batch: [num_total_atoms]
            k: number of atoms to select
            
        Returns:
            List of [k] index tensors
        """
        scores_list = self.forward(atom_features, batch)
        top_k_indices = []
        
        for scores in scores_list:
            k_actual = min(k, len(scores))
            _, indices = torch.topk(scores, k_actual)
            top_k_indices.append(indices)
        
        return top_k_indices
