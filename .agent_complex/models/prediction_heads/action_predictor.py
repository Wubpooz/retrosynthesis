"""Action prediction for atoms and bonds."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomActionPredictor(nn.Module):
    """
    Predicts atom-level actions (H-count changes, leaving groups, etc).
    Multi-class classification over action vocabulary.
    """
    
    def __init__(self, atom_dim: int, action_vocab_size: int):
        super().__init__()
        self.action_vocab_size = action_vocab_size
        
        self.action_classifier = nn.Sequential(
            nn.Linear(atom_dim, atom_dim // 2),
            nn.GELU(),
            nn.Linear(atom_dim // 2, action_vocab_size)
        )
    
    def forward(self, atom_features: torch.Tensor, rc_indices: list) -> list:
        """
        Predict actions for reaction center atoms.
        
        Args:
            atom_features: [num_total_atoms, atom_dim]
            rc_indices: List of [num_rc_i] tensors with RC atom indices per graph
            
        Returns:
            List of [num_rc_i, action_vocab_size] logit tensors
        """
        outputs = []
        offset = 0
        
        for indices in rc_indices:
            if len(indices) == 0:
                # No reaction centers
                outputs.append(torch.empty(0, self.action_vocab_size, 
                                          device=atom_features.device))
                continue
            
            # Get features for RC atoms
            rc_feats = atom_features[indices + offset]
            
            # Predict actions
            action_logits = self.action_classifier(rc_feats)
            outputs.append(action_logits)
            
            # Update offset for next graph
            # Note: This assumes indices are relative to each graph
        
        return outputs


class BondActionPredictor(nn.Module):
    """
    Predicts bond-level actions (bond type changes, formation, breaking).
    Includes atom actions for bond endpoints.
    """
    
    def __init__(self, atom_dim: int, bond_action_vocab_size: int, 
                 atom_action_vocab_size: int):
        super().__init__()
        self.bond_action_vocab_size = bond_action_vocab_size
        self.atom_action_vocab_size = atom_action_vocab_size
        
        # Bond action classifier (uses both endpoint features)
        self.bond_classifier = nn.Sequential(
            nn.Linear(atom_dim * 2, atom_dim),
            nn.GELU(),
            nn.Linear(atom_dim, bond_action_vocab_size)
        )
        
        # Atom action classifier for endpoints
        self.atom_classifier = nn.Sequential(
            nn.Linear(atom_dim, atom_dim // 2),
            nn.GELU(),
            nn.Linear(atom_dim // 2, atom_action_vocab_size)
        )
    
    def forward(self, atom_features: torch.Tensor, rc_bond_indices: list) -> tuple:
        """
        Predict actions for reaction center bonds and their endpoint atoms.
        
        Args:
            atom_features: [num_total_atoms, atom_dim]
            rc_bond_indices: List of [num_rc_bonds_i, 2] tensors with RC bond endpoints
            
        Returns:
            bond_actions: List of [num_rc_bonds_i, bond_action_vocab_size] tensors
            atom_actions: List of [num_rc_bonds_i * 2, atom_action_vocab_size] tensors
        """
        bond_outputs = []
        atom_outputs = []
        
        for bond_indices in rc_bond_indices:
            if len(bond_indices) == 0:
                bond_outputs.append(torch.empty(0, self.bond_action_vocab_size,
                                               device=atom_features.device))
                atom_outputs.append(torch.empty(0, self.atom_action_vocab_size,
                                               device=atom_features.device))
                continue
            
            # Get features for bond endpoints
            src_feats = atom_features[bond_indices[:, 0]]  # [num_bonds, atom_dim]
            dst_feats = atom_features[bond_indices[:, 1]]  # [num_bonds, atom_dim]
            
            # Predict bond actions
            bond_feats = torch.cat([src_feats, dst_feats], dim=-1)
            bond_logits = self.bond_classifier(bond_feats)
            bond_outputs.append(bond_logits)
            
            # Predict atom actions for both endpoints
            endpoint_feats = torch.cat([src_feats, dst_feats], dim=0)  # [num_bonds * 2, atom_dim]
            atom_logits = self.atom_classifier(endpoint_feats)
            atom_outputs.append(atom_logits)
        
        return bond_outputs, atom_outputs
