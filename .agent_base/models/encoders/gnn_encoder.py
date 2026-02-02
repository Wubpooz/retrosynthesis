import torch
import torch.nn as nn

class MolecularEncoder(nn.Module):
    """Abstract base for GNN and Mamba encoders"""
    def forward(self, mol_graph):
        """
        Forward pass for molecular encoder.
        Args:
            mol_graph: Input molecular graph.
        Returns:
            Tuple of (atom_features, pair_features, graph_features)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class UniMolPlusEncoder(MolecularEncoder):
    """
    GNN Encoder (Uni-Mol+ Style)
    Components:
    - Atom embedding: atomic number, formal charge, chirality
    - Bond embedding: bond type, aromaticity, ring membership
    - Transformer blocks with pair bias attention
    - 6 layers, hidden_dim=256, pair_dim=128
    """
    
    def __init__(self, config):
        super(UniMolPlusEncoder, self).__init__()
        self.atom_embedding = nn.Embedding(config['num_atom_types'], config['hidden_dim'])
        self.bond_embedding = nn.Embedding(config['num_bond_types'], config['pair_dim'])
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_dim'],
                nhead=config['num_heads'],
                dim_feedforward=config['ffn_dim']
            ) for _ in range(config['num_layers'])
        ])

    def forward(self, batch):
        """
        Forward pass for GNN encoder.
        Args:
            batch: Batched molecular graph data.
        Returns:
            Tuple of (atom_features, pair_features, graph_features)
        """
        # Embed atoms and bonds
        atom_features = self.atom_embedding(batch['atom_types'])
        bond_features = self.bond_embedding(batch['bond_types'])

        # Apply transformer blocks
        for block in self.transformer_blocks:
            atom_features = block(atom_features)

        # Compute graph-level features (e.g., mean pooling)
        graph_features = torch.mean(atom_features, dim=1)

        return atom_features, bond_features, graph_features