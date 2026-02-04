"""
Mamba-based encoder for molecular graphs.
Uses state-space models for efficient sequential modeling.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from typing import Tuple, Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")

from .base_encoder import MolecularEncoder, AtomEmbedding
from .graph_linearizer import GraphLinearizer


class MambaBlock(nn.Module):
    """Mamba SSM block with residual connections."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required for MambaBlock")
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model]
        """
        return x + self.mamba(self.norm(x))


class MambaEncoder(MolecularEncoder):
    """
    Mamba-based molecular encoder.
    
    Converts molecular graphs to sequences using DFS/BFS linearization,
    then applies Mamba blocks for efficient O(n) sequential modeling.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required. Install with: pip install mamba-ssm")
        
        self.hidden_dim = config.get('hidden_dim', 768)
        self.num_layers = config.get('num_layers', 12)
        self.dropout = config.get('dropout', 0.1)
        
        # Mamba config
        mamba_config = config.get('mamba', {})
        self.d_state = mamba_config.get('d_state', 16)
        self.d_conv = mamba_config.get('d_conv', 4)
        self.expand = mamba_config.get('expand', 2)
        
        # Linearization config
        lin_config = config.get('linearization', {})
        self.strategy = lin_config.get('strategy', 'dfs')
        self.add_backtrack = lin_config.get('add_backtrack_tokens', True)
        
        self.linearizer = GraphLinearizer(self.strategy, self.add_backtrack)
        
        # Atom embedding
        self.atom_embedding = AtomEmbedding(self.hidden_dim)
        
        # Special token for backtracking (if enabled)
        if self.add_backtrack:
            self.backtrack_token = nn.Parameter(torch.randn(1, self.hidden_dim))
        
        # Position encoding
        self.position_encoding = nn.Embedding(512, self.hidden_dim)  # Max seq length 512
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.hidden_dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projections
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.graph_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def forward(self, graph: Batch) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                             torch.Tensor, dict]:
        """
        Encode molecular graph using Mamba.
        
        Args:
            graph: PyTorch Geometric Batch
            
        Returns:
            atom_features: [num_total_atoms, hidden_dim]
            pair_features: None (not computed)
            graph_features: [batch_size, hidden_dim]
            batch_info: metadata dict
        """
        # Create batch info
        batch_info = self.create_batch_info(graph)
        batch_size = batch_info['num_graphs']
        
        # Embed atoms
        atom_feats = self.atom_embedding(graph.x)  # [num_total_atoms, hidden_dim]
        
        # Linearize graphs
        sequence_indices, graph_sizes = self.linearizer.linearize_batch(
            graph.edge_index, graph.batch
        )
        
        # Create sequential representation
        max_seq_len = max(graph_sizes)
        sequences = torch.zeros(
            batch_size, max_seq_len, self.hidden_dim,
            device=atom_feats.device, dtype=atom_feats.dtype
        )
        
        offset = 0
        for i, seq_len in enumerate(graph_sizes):
            seq_indices = sequence_indices[offset:offset + seq_len]
            
            # Handle backtrack tokens
            for j, idx in enumerate(seq_indices):
                if idx >= 0:
                    sequences[i, j] = atom_feats[idx]
                elif self.add_backtrack:
                    sequences[i, j] = self.backtrack_token.squeeze(0)
            
            offset += seq_len
        
        # Add positional encoding
        positions = torch.arange(max_seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        sequences = sequences + self.position_encoding(positions)
        sequences = self.dropout_layer(sequences)
        
        # Apply Mamba blocks
        for block in self.blocks:
            sequences = block(sequences)
        
        # Map back to atom features
        atom_feats_out = []
        offset = 0
        for i in range(batch_size):
            num_atoms = batch_info['num_atoms'][i]
            seq_len = graph_sizes[i]
            
            # Extract features for actual atoms (not backtrack tokens)
            seq_indices = sequence_indices[offset:offset + seq_len]
            atom_indices = [idx for idx in seq_indices if idx >= 0][:num_atoms]
            
            # Average features for each atom across sequence
            atom_feats_i = torch.zeros(num_atoms, self.hidden_dim, device=sequences.device)
            for j, atom_idx in enumerate(atom_indices):
                # Find positions in sequence where this atom appears
                positions = (seq_indices == atom_idx).nonzero(as_tuple=True)[0]
                atom_feats_i[j] = sequences[i, positions].mean(dim=0)
            
            atom_feats_out.append(atom_feats_i)
            offset += seq_len
        
        atom_feats_out = torch.cat(atom_feats_out, dim=0)
        
        # Global pooling for graph features
        graph_feats = global_mean_pool(atom_feats_out, graph.batch)
        graph_feats = self.graph_projection(graph_feats)
        
        return atom_feats_out, None, graph_feats, batch_info


if __name__ == '__main__':
    if MAMBA_AVAILABLE:
        from torch_geometric.data import Data
        
        # Create dummy graph
        x = torch.randn(10, 7)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        edge_attr = torch.randn(6, 4)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = Batch.from_data_list([data])
        
        config = {
            'hidden_dim': 768,
            'num_layers': 2,
            'dropout': 0.1,
            'mamba': {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            },
            'linearization': {
                'strategy': 'dfs',
                'add_backtrack_tokens': True
            }
        }
        
        encoder = MambaEncoder(config)
        atom_feats, pair_feats, graph_feats, batch_info = encoder(batch)
        
        print(f"Atom features: {atom_feats.shape}")
        print(f"Graph features: {graph_feats.shape}")
    else:
        print("Mamba not available - install mamba-ssm to test")
