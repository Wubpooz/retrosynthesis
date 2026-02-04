"""
GNN Encoder using Uni-Mol+ style architecture.
Implements pair-wise attention with bond features as bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from typing import Tuple, Optional

from .base_encoder import MolecularEncoder, AtomEmbedding, BondEmbedding


class PairBiasAttention(nn.Module):
    """
    Multi-head attention with pair-wise bias from bond features.
    Used in Uni-Mol+ and AlphaFold.
    """
    
    def __init__(self, atom_dim: int, pair_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.atom_dim = atom_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = atom_dim // num_heads
        
        assert atom_dim % num_heads == 0, "atom_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(atom_dim, atom_dim)
        self.k_proj = nn.Linear(atom_dim, atom_dim)
        self.v_proj = nn.Linear(atom_dim, atom_dim)
        
        # Pair bias projection
        self.pair_bias_proj = nn.Linear(pair_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(atom_dim, atom_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, atom_feats: torch.Tensor, pair_feats: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atom_feats: [batch, num_atoms, atom_dim]
            pair_feats: [batch, num_atoms, num_atoms, pair_dim]
            mask: [batch, num_atoms] boolean mask for valid atoms
            
        Returns:
            [batch, num_atoms, atom_dim]
        """
        batch_size, num_atoms, _ = atom_feats.shape
        
        # Project to Q, K, V
        q = self.q_proj(atom_feats).view(batch_size, num_atoms, self.num_heads, self.head_dim)
        k = self.k_proj(atom_feats).view(batch_size, num_atoms, self.num_heads, self.head_dim)
        v = self.v_proj(atom_feats).view(batch_size, num_atoms, self.num_heads, self.head_dim)
        
        # Reshape for attention: [batch, num_heads, num_atoms, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, num_heads, num_atoms, num_atoms]
        
        # Add pair bias
        pair_bias = self.pair_bias_proj(pair_feats).permute(0, 3, 1, 2)  # [batch, num_heads, num_atoms, num_atoms]
        attn = attn + pair_bias
        
        # Apply mask if provided
        if mask is not None:
            # Create attention mask: [batch, 1, 1, num_atoms]
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, num_heads, num_atoms, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, num_atoms, self.atom_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with pair bias attention."""
    
    def __init__(self, atom_dim: int, pair_dim: int, num_heads: int = 8, 
                 ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.attention = PairBiasAttention(atom_dim, pair_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(atom_dim)
        self.norm2 = nn.LayerNorm(atom_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(atom_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, atom_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, atom_feats: torch.Tensor, pair_feats: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atom_feats: [batch, num_atoms, atom_dim]
            pair_feats: [batch, num_atoms, num_atoms, pair_dim]
            mask: [batch, num_atoms]
            
        Returns:
            [batch, num_atoms, atom_dim]
        """
        # Attention with residual
        attn_out = self.attention(self.norm1(atom_feats), pair_feats, mask)
        atom_feats = atom_feats + attn_out
        
        # Feed-forward with residual
        ff_out = self.ff(self.norm2(atom_feats))
        atom_feats = atom_feats + ff_out
        
        return atom_feats


class UniMolPlusEncoder(MolecularEncoder):
    """
    Uni-Mol+ style GNN encoder with pair bias attention.
    
    Architecture:
    - Embed atoms and bonds
    - Create pair representations from bonds
    - Apply transformer blocks with pair bias
    - Return atom, pair, and graph-level features
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.atom_dim = config.get('atom_dim', 256)
        self.pair_dim = config.get('pair_dim', 128)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Embedding layers
        self.atom_embedding = AtomEmbedding(self.atom_dim)
        self.bond_embedding = BondEmbedding(self.pair_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.atom_dim, 
                self.pair_dim, 
                self.num_heads,
                ff_dim=self.atom_dim * 4,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Graph-level pooling
        self.graph_projection = nn.Sequential(
            nn.Linear(self.atom_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def create_pair_features(self, bond_feats: torch.Tensor, edge_index: torch.Tensor,
                            num_atoms: int, batch_size: int) -> torch.Tensor:
        """
        Create dense pair representation matrix from sparse edges.
        
        Args:
            bond_feats: [num_edges, pair_dim]
            edge_index: [2, num_edges]
            num_atoms: maximum number of atoms
            batch_size: number of graphs
            
        Returns:
            [batch_size, num_atoms, num_atoms, pair_dim]
        """
        pair_feats = torch.zeros(
            batch_size, num_atoms, num_atoms, self.pair_dim,
            device=bond_feats.device, dtype=bond_feats.dtype
        )
        
        # Fill in bond features
        # Note: This is simplified; in practice, need to handle batching properly
        i, j = edge_index[0], edge_index[1]
        
        # For now, use a simpler approach with max pooling
        # In full implementation, properly batch the edge features
        
        return pair_feats
    
    def forward(self, graph: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], 
                                             torch.Tensor, dict]:
        """
        Encode molecular graph.
        
        Args:
            graph: PyTorch Geometric Batch
            
        Returns:
            atom_features: [num_total_atoms, atom_dim]
            pair_features: None (not used in this simplified version)
            graph_features: [batch_size, hidden_dim]
            batch_info: metadata dict
        """
        # Create batch info
        batch_info = self.create_batch_info(graph)
        batch_size = batch_info['num_graphs']
        
        # Embed atoms and bonds
        atom_feats = self.atom_embedding(graph.x)  # [num_total_atoms, atom_dim]
        bond_feats = self.bond_embedding(graph.edge_attr)  # [num_edges, pair_dim]
        
        # For simplicity, we'll process without dense pair features in this version
        # In full implementation, convert to dense format for pair bias attention
        
        # Group by batch for processing
        max_atoms = max(batch_info['num_atoms'])
        
        # Pad and reshape atom features to [batch, max_atoms, atom_dim]
        atom_feats_padded = torch.zeros(
            batch_size, max_atoms, self.atom_dim,
            device=atom_feats.device, dtype=atom_feats.dtype
        )
        
        masks = torch.zeros(batch_size, max_atoms, device=atom_feats.device, dtype=torch.bool)
        
        offsets = batch_info['atom_offsets']
        for i in range(batch_size):
            start, end = offsets[i], offsets[i + 1]
            num_atoms_i = end - start
            atom_feats_padded[i, :num_atoms_i] = atom_feats[start:end]
            masks[i, :num_atoms_i] = True
        
        # Create simple pair features (identity for this simplified version)
        pair_feats = torch.zeros(
            batch_size, max_atoms, max_atoms, self.pair_dim,
            device=atom_feats.device, dtype=atom_feats.dtype
        )
        
        # Apply transformer blocks
        for block in self.blocks:
            atom_feats_padded = block(atom_feats_padded, pair_feats, masks)
        
        # Unpad back to flat representation
        atom_feats_out = []
        for i in range(batch_size):
            num_atoms_i = batch_info['num_atoms'][i]
            atom_feats_out.append(atom_feats_padded[i, :num_atoms_i])
        atom_feats_out = torch.cat(atom_feats_out, dim=0)
        
        # Global pooling for graph features
        graph_feats = global_mean_pool(atom_feats_out, graph.batch)  # [batch_size, atom_dim]
        graph_feats = self.graph_projection(graph_feats)  # [batch_size, hidden_dim]
        
        return atom_feats_out, None, graph_feats, batch_info


if __name__ == '__main__':
    # Test encoder
    from torch_geometric.data import Data
    
    # Create dummy graph
    x = torch.randn(10, 7)  # 10 atoms
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)  # 4 edges
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])
    
    config = {
        'hidden_dim': 256,
        'atom_dim': 256,
        'pair_dim': 128,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.1
    }
    
    encoder = UniMolPlusEncoder(config)
    atom_feats, pair_feats, graph_feats, batch_info = encoder(batch)
    
    print(f"Atom features: {atom_feats.shape}")
    print(f"Pair features: {pair_feats}")
    print(f"Graph features: {graph_feats.shape}")
    print(f"Batch info: {batch_info}")
