import torch
import torch.nn as nn

class RCTypePredictor(nn.Module):
    """Binary classifier: atom vs bond center"""
    def __init__(self, hidden_dim):
        super(RCTypePredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, graph_features):
        """
        Forward pass for reaction center type prediction.
        Args:
            graph_features: Graph-level features from encoder.
        Returns:
            Binary classification logits.
        """
        return self.classifier(graph_features)

class AtomCenterPredictor(nn.Module):
    """Predict which atom is the reaction center"""
    def __init__(self, hidden_dim, num_atoms):
        super(AtomCenterPredictor, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_atoms)

    def forward(self, graph_features):
        """
        Forward pass for atom center localization.
        Args:
            graph_features: Graph-level features from encoder.
        Returns:
            Atom-level logits.
        """
        return self.classifier(graph_features)

class BondCenterPredictor(nn.Module):
    """Predict which bond is the reaction center"""
    def __init__(self, hidden_dim, num_bonds):
        super(BondCenterPredictor, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_bonds)

    def forward(self, graph_features):
        """
        Forward pass for bond center localization.
        Args:
            graph_features: Graph-level features from encoder.
        Returns:
            Bond-level logits.
        """
        return self.classifier(graph_features)

class AtomActionPredictor(nn.Module):
    """Predict atom-level edits (H-count, chirality, LG)"""
    def __init__(self, hidden_dim, num_actions):
        super(AtomActionPredictor, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_actions)

    def forward(self, atom_features):
        """
        Forward pass for atom action prediction.
        Args:
            atom_features: Atom-level features from encoder.
        Returns:
            Action logits.
        """
        return self.classifier(atom_features)

class BondActionPredictor(nn.Module):
    """Predict bond-level edits (type change, deletion)"""
    def __init__(self, hidden_dim, num_actions):
        super(BondActionPredictor, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_actions)

    def forward(self, bond_features):
        """
        Forward pass for bond action prediction.
        Args:
            bond_features: Bond-level features from encoder.
        Returns:
            Action logits.
        """
        return self.classifier(bond_features)

class TerminationPredictor(nn.Module):
    """Decide if retrosynthesis process should stop"""
    def __init__(self, hidden_dim):
        super(TerminationPredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, graph_features):
        """
        Forward pass for termination prediction.
        Args:
            graph_features: Graph-level features from encoder.
        Returns:
            Binary classification logits.
        """
        return self.classifier(graph_features)