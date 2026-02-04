"""
Base encoder abstract class for molecular representation learning.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch


class MolecularEncoder(nn.Module, ABC):
    """
    Abstract base class for molecular encoders.
    
    All encoders must implement forward() to return:
    - atom_features: [batch*num_atoms, atom_dim]
    - pair_features: [batch*num_atoms, num_atoms, pair_dim] or None
    - graph_features: [batch, hidden_dim]
    - batch_info: dict with batch indices and sizes
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
    
    @abstractmethod
    def forward(self, graph: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], 
                                             torch.Tensor, dict]:
        """
        Encode molecular graph.
        
        Args:
            graph: PyTorch Geometric Batch object
            
        Returns:
            atom_features: [num_total_atoms, atom_dim]
            pair_features: [num_total_atoms, max_atoms, pair_dim] or None
            graph_features: [batch_size, hidden_dim]
            batch_info: {
                'batch': batch assignment for atoms,
                'num_atoms': list of atom counts per graph,
                'atom_offsets': cumulative sum for indexing
            }
        """
        pass
    
    def get_atom_features_by_batch(self, atom_features: torch.Tensor, 
                                   batch_info: dict) -> list:
        """
        Split concatenated atom features back into per-graph tensors.
        
        Args:
            atom_features: [num_total_atoms, dim]
            batch_info: dict from forward()
            
        Returns:
            List of [num_atoms_i, dim] tensors
        """
        features_list = []
        offsets = batch_info['atom_offsets']
        
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            features_list.append(atom_features[start:end])
        
        return features_list
    
    def create_batch_info(self, batch: Batch) -> dict:
        """Create batch info dict from PyG Batch."""
        # Get number of atoms per graph
        unique, counts = torch.unique(batch.batch, return_counts=True)
        num_atoms = counts.tolist()
        
        # Create cumulative offsets
        atom_offsets = [0] + torch.cumsum(counts, dim=0).tolist()
        
        return {
            'batch': batch.batch,
            'num_atoms': num_atoms,
            'atom_offsets': atom_offsets,
            'num_graphs': len(num_atoms)
        }


class EmbeddingLayer(nn.Module):
    """Embedding layer for atomic features."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 continuous_features: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.continuous_features = continuous_features
        
        if continuous_features > 0:
            self.continuous_proj = nn.Linear(continuous_features, embedding_dim)
    
    def forward(self, categorical_feats: torch.Tensor, 
               continuous_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            categorical_feats: [batch, num_features] integer indices
            continuous_feats: [batch, num_continuous] float values
            
        Returns:
            [batch, embedding_dim]
        """
        # Sum embeddings for categorical features
        embedded = self.embedding(categorical_feats).sum(dim=1)
        
        # Add continuous features if provided
        if continuous_feats is not None and self.continuous_features > 0:
            embedded = embedded + self.continuous_proj(continuous_feats)
        
        return embedded


class AtomEmbedding(nn.Module):
    """Atom feature embedding."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        
        # Embeddings for different atom properties
        self.atomic_num_embedding = nn.Embedding(120, dim)  # Up to element 120
        self.formal_charge_embedding = nn.Embedding(11, dim)  # -5 to +5
        self.chirality_embedding = nn.Embedding(5, dim)  # CHI_UNSPECIFIED, etc.
        self.hybridization_embedding = nn.Embedding(10, dim)  # SP, SP2, SP3, etc.
        
        # Continuous features projection
        self.continuous_proj = nn.Linear(3, dim)  # degree, num_hs, aromatic
        
    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_features: [num_atoms, 7] containing:
                [atomic_num, formal_charge, num_hs, is_aromatic, chirality, degree, hybridization]
                
        Returns:
            [num_atoms, dim]
        """
        # Extract features
        atomic_num = atom_features[:, 0].long()
        formal_charge = (atom_features[:, 1] + 5).long().clamp(0, 10)  # Shift to 0-10
        num_hs = atom_features[:, 2]
        is_aromatic = atom_features[:, 3]
        chirality = atom_features[:, 4].long().clamp(0, 4)
        degree = atom_features[:, 5]
        hybridization = atom_features[:, 6].long().clamp(0, 9)
        
        # Embed categorical features
        embedded = (
            self.atomic_num_embedding(atomic_num) +
            self.formal_charge_embedding(formal_charge) +
            self.chirality_embedding(chirality) +
            self.hybridization_embedding(hybridization)
        )
        
        # Add continuous features
        continuous = torch.stack([degree, num_hs, is_aromatic], dim=-1)
        embedded = embedded + self.continuous_proj(continuous)
        
        return embedded


class BondEmbedding(nn.Module):
    """Bond feature embedding."""
    
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        
        # Bond type: SINGLE, DOUBLE, TRIPLE, AROMATIC
        self.bond_type_embedding = nn.Embedding(10, dim)
        
        # Stereochemistry
        self.stereo_embedding = nn.Embedding(7, dim)
        
        # Binary features
        self.binary_proj = nn.Linear(2, dim)  # conjugated, in_ring
        
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: [num_edges, 4] containing:
                [bond_type_double, is_conjugated, is_in_ring, stereo]
                
        Returns:
            [num_edges, dim]
        """
        # Extract features
        bond_type = (edge_features[:, 0] * 2).long().clamp(0, 9)  # Convert to integer type
        is_conjugated = edge_features[:, 1]
        is_in_ring = edge_features[:, 2]
        stereo = edge_features[:, 3].long().clamp(0, 6)
        
        # Embed
        embedded = (
            self.bond_type_embedding(bond_type) +
            self.stereo_embedding(stereo)
        )
        
        # Add binary features
        binary = torch.stack([is_conjugated, is_in_ring], dim=-1)
        embedded = embedded + self.binary_proj(binary)
        
        return embedded
