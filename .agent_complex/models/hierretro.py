"""
HierRetro: Hierarchical Retrosynthesis Prediction Model
Combines GNN or Mamba encoder with hierarchical prediction heads.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from torch_geometric.data import Batch

from .encoders import MolecularEncoder, UniMolPlusEncoder, MambaEncoder
from .prediction_heads import (
    RCTypePredictor,
    AtomCenterPredictor,
    BondCenterPredictor,
    AtomActionPredictor,
    BondActionPredictor,
    TerminationPredictor
)


class HierRetro(nn.Module):
    """
    Hierarchical Retrosynthesis Model.
    
    Pipeline:
    1. Encode product molecule (GNN or Mamba)
    2. Predict RC type (atom vs bond)
    3. Localize reaction center (atom or bond)
    4. Predict actions at reaction center
    5. Predict termination
    """
    
    def __init__(self, encoder_type: str = 'gnn', config: Optional[Dict] = None):
        super().__init__()
        
        if config is None:
            config = self._default_config(encoder_type)
        
        self.encoder_type = encoder_type
        self.config = config
        
        # Choose encoder
        encoder_config = config.get('encoder', {})
        encoder_config['hidden_dim'] = config.get('hidden_dim', 256)
        
        if encoder_type == 'gnn':
            self.encoder = UniMolPlusEncoder(encoder_config)
        elif encoder_type == 'mamba':
            self.encoder = MambaEncoder(encoder_config)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        hidden_dim = self.encoder.hidden_dim
        atom_dim = encoder_config.get('atom_dim', hidden_dim)
        
        # Prediction heads
        pred_config = config.get('prediction_heads', {})
        
        self.rc_type_pred = RCTypePredictor(
            hidden_dim=hidden_dim,
            dropout=config.get('dropout', 0.1)
        )
        
        self.atom_center_pred = AtomCenterPredictor(atom_dim=atom_dim)
        
        self.bond_center_pred = BondCenterPredictor(atom_dim=atom_dim)
        
        # Action predictors
        atom_action_vocab_size = pred_config.get('atom_action_vocab_size', 50)
        bond_action_vocab_size = pred_config.get('bond_action_vocab_size', 20)
        
        self.atom_action_pred = AtomActionPredictor(
            atom_dim=atom_dim,
            action_vocab_size=atom_action_vocab_size
        )
        
        self.bond_action_pred = BondActionPredictor(
            atom_dim=atom_dim,
            bond_action_vocab_size=bond_action_vocab_size,
            atom_action_vocab_size=atom_action_vocab_size
        )
        
        self.termination_pred = TerminationPredictor(
            hidden_dim=hidden_dim,
            dropout=config.get('dropout', 0.1)
        )
    
    def _default_config(self, encoder_type: str) -> Dict:
        """Get default configuration."""
        if encoder_type == 'gnn':
            return {
                'hidden_dim': 256,
                'dropout': 0.1,
                'encoder': {
                    'atom_dim': 256,
                    'pair_dim': 128,
                    'num_layers': 6,
                    'num_heads': 8
                },
                'prediction_heads': {
                    'atom_action_vocab_size': 50,
                    'bond_action_vocab_size': 20
                }
            }
        else:  # mamba
            return {
                'hidden_dim': 768,
                'dropout': 0.1,
                'encoder': {
                    'num_layers': 12,
                    'mamba': {
                        'd_state': 16,
                        'd_conv': 4,
                        'expand': 2
                    }
                },
                'prediction_heads': {
                    'atom_action_vocab_size': 50,
                    'bond_action_vocab_size': 20
                }
            }
    
    def forward(self, graph: Batch, history: Optional[Dict] = None) -> Dict:
        """
        Forward pass through hierarchical prediction.
        
        Args:
            graph: PyTorch Geometric batch of product molecules
            history: Optional dict with previous step information
            
        Returns:
            Dict with:
            - atom_features: [num_atoms, dim]
            - graph_features: [batch, hidden_dim]
            - rc_type_logits: [batch, 2]
            - atom_center_logits: List of [num_atoms_i] tensors
            - bond_center_logits: List of [num_bonds_i] tensors
            - termination_logits: [batch, 2]
            - batch_info: metadata
        """
        # Encode molecule
        atom_feats, pair_feats, graph_feats, batch_info = self.encoder(graph)
        
        # Predict RC type
        rc_type_logits = self.rc_type_pred(graph_feats)
        
        # Predict reaction centers (both atom and bond, select based on type later)
        atom_center_logits = self.atom_center_pred(atom_feats, graph.batch)
        bond_center_logits = self.bond_center_pred(atom_feats, graph.edge_index, graph.batch)
        
        # Predict termination
        termination_logits = self.termination_pred(graph_feats)
        
        return {
            'atom_features': atom_feats,
            'graph_features': graph_feats,
            'rc_type_logits': rc_type_logits,
            'atom_center_logits': atom_center_logits,
            'bond_center_logits': bond_center_logits,
            'termination_logits': termination_logits,
            'batch_info': batch_info
        }
    
    def predict_actions(self, outputs: Dict, graph: Batch, use_predicted_rc: bool = True) -> Dict:
        """
        Predict actions given reaction centers.
        
        Args:
            outputs: Output from forward()
            graph: Original graph batch
            use_predicted_rc: If True, use model's RC predictions; else use ground truth
            
        Returns:
            Updated outputs dict with action predictions
        """
        atom_feats = outputs['atom_features']
        rc_type_probs = torch.softmax(outputs['rc_type_logits'], dim=-1)
        
        # Decide on RC type (simplified: use argmax)
        rc_types = rc_type_probs.argmax(dim=-1)  # [batch]
        
        # Get top-k atom and bond centers
        top_atom_centers = self.atom_center_pred.predict_top_k(
            atom_feats, graph.batch, k=5
        )
        top_bond_centers = self.bond_center_pred.predict_top_k(
            atom_feats, graph.edge_index, graph.batch, k=5
        )
        
        # Predict actions for atoms
        atom_action_logits = self.atom_action_pred(atom_feats, top_atom_centers)
        
        # Predict actions for bonds
        bond_action_logits, bond_endpoint_actions = self.bond_action_pred(
            atom_feats, top_bond_centers
        )
        
        outputs['rc_types'] = rc_types
        outputs['top_atom_centers'] = top_atom_centers
        outputs['top_bond_centers'] = top_bond_centers
        outputs['atom_action_logits'] = atom_action_logits
        outputs['bond_action_logits'] = bond_action_logits
        outputs['bond_endpoint_actions'] = bond_endpoint_actions
        
        return outputs


if __name__ == '__main__':
    # Test model
    from torch_geometric.data import Data
    
    # Create dummy data
    x = torch.randn(10, 7)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])
    
    print("Testing GNN encoder:")
    model_gnn = HierRetro(encoder_type='gnn')
    outputs = model_gnn(batch)
    print(f"  RC type logits: {outputs['rc_type_logits'].shape}")
    print(f"  Graph features: {outputs['graph_features'].shape}")
    print(f"  Atom center logits: {len(outputs['atom_center_logits'])} graphs")
    
    # Test with actions
    outputs = model_gnn.predict_actions(outputs, batch)
    print(f"  Atom action logits: {len(outputs['atom_action_logits'])} graphs")
    
    print("\nTesting Mamba encoder:")
    try:
        model_mamba = HierRetro(encoder_type='mamba')
        outputs = model_mamba(batch)
        print(f"  RC type logits: {outputs['rc_type_logits'].shape}")
        print(f"  Graph features: {outputs['graph_features'].shape}")
    except ImportError:
        print("  Mamba not available (install mamba-ssm)")
