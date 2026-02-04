"""Bond-level reaction center localization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BondCenterPredictor(nn.Module):
    """
    Predicts which bond(s) are reaction centers.
    Works on atom pairs connected by bonds.
    """
    
    def __init__(self, atom_dim: int):
        super().__init__()
        
        # Score bonds based on concatenated endpoint atom features
        self.scorer = nn.Sequential(
            nn.Linear(atom_dim * 2, atom_dim),
            nn.GELU(),
            nn.Linear(atom_dim, atom_dim // 2),
            nn.GELU(),
            nn.Linear(atom_dim // 2, 1)
        )
    
    def forward(self, atom_features: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> list:
        """
        Args:
            atom_features: [num_total_atoms, atom_dim]
            edge_index: [2, num_edges]
            batch: [num_total_atoms] batch assignment
            
        Returns:
            List of [num_bonds_i] logit tensors, one per graph
        """
        # Get features for bond endpoints
        src_feats = atom_features[edge_index[0]]  # [num_edges, atom_dim]
        dst_feats = atom_features[edge_index[1]]  # [num_edges, atom_dim]
        
        # Concatenate endpoint features
        bond_feats = torch.cat([src_feats, dst_feats], dim=-1)  # [num_edges, atom_dim * 2]
        
        # Score bonds
        scores = self.scorer(bond_feats).squeeze(-1)  # [num_edges]
        
        # Split by batch (use source atom batch)
        edge_batch = batch[edge_index[0]]
        unique_batches = torch.unique(edge_batch, sorted=True)
        outputs = []
        
        for b in unique_batches:
            mask = edge_batch == b
            outputs.append(scores[mask])
        
        return outputs
    
    def predict_top_k(self, atom_features: torch.Tensor, edge_index: torch.Tensor,
                     batch: torch.Tensor, k: int = 5) -> list:
        """
        Predict top-k reaction center bonds.
        
        Returns:
            List of [k, 2] edge index tensors
        """
        scores_list = self.forward(atom_features, edge_index, batch)
        edge_batch = batch[edge_index[0]]
        unique_batches = torch.unique(edge_batch, sorted=True)
        
        top_k_edges = []
        offset = 0
        
        for b_idx, b in enumerate(unique_batches):
            mask = edge_batch == b
            graph_edges = edge_index[:, mask]  # [2, num_edges_i]
            scores = scores_list[b_idx]
            
            k_actual = min(k, len(scores))
            _, indices = torch.topk(scores, k_actual)
            
            # Get corresponding edges
            selected_edges = graph_edges[:, indices].t()  # [k, 2]
            top_k_edges.append(selected_edges)
        
        return top_k_edges
