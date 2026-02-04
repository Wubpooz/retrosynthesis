"""
Beam search inference for retrosynthesis.
"""

import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
from torch_geometric.data import Data, Batch
from rdkit import Chem

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import HierRetro


@dataclass
class BeamState:
    """State in beam search."""
    smiles: str
    score: float
    history: List[Dict]
    terminated: bool


class BeamSearchRetrosynthesis:
    """
    Beam search for hierarchical retrosynthesis prediction.
    """
    
    def __init__(self, model: HierRetro, beam_width: int = 10, max_steps: int = 5):
        """
        Args:
            model: Trained HierRetro model
            beam_width: Number of beams to maintain
            max_steps: Maximum retrosynthesis steps
        """
        self.model = model
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.device = next(model.parameters()).device
        
    def smiles_to_graph(self, smiles: str) -> Data:
        """Convert SMILES to PyG graph."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Atom features (simplified)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic()),
                int(atom.GetChiralTag()),
                atom.GetDegree(),
                atom.GetHybridization().real,
            ]
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edges
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            
            bond_feats = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo()),
            ]
            edge_features.extend([bond_feats, bond_feats])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    @torch.no_grad()
    def predict(self, product_smiles: str, return_all: bool = False) -> List[Dict]:
        """
        Predict reactants for a product molecule.
        
        Args:
            product_smiles: Product SMILES string
            return_all: If True, return all beams; else return best
            
        Returns:
            List of prediction dicts with 'reactants', 'score', 'steps'
        """
        self.model.eval()
        
        # Initialize beam
        initial_state = BeamState(
            smiles=product_smiles,
            score=0.0,
            history=[],
            terminated=False
        )
        
        beams = [initial_state]
        
        # Beam search
        for step in range(self.max_steps):
            new_beams = []
            
            for beam in beams:
                if beam.terminated:
                    new_beams.append(beam)
                    continue
                
                # Convert to graph
                try:
                    graph = self.smiles_to_graph(beam.smiles)
                    graph_batch = Batch.from_data_list([graph]).to(self.device)
                except:
                    # Invalid SMILES - terminate this beam
                    beam.terminated = True
                    new_beams.append(beam)
                    continue
                
                # Forward pass
                outputs = self.model(graph_batch)
                outputs = self.model.predict_actions(outputs, graph_batch)
                
                # Check termination
                term_probs = torch.softmax(outputs['termination_logits'], dim=-1)
                if term_probs[0, 1] > 0.5:  # Predict terminate
                    beam.terminated = True
                    new_beams.append(beam)
                    continue
                
                # Get top predictions (simplified - just take top atom centers)
                rc_type = outputs['rc_types'][0].item()
                
                if rc_type == 0:  # Atom center
                    top_centers = outputs['top_atom_centers'][0][:self.beam_width]
                    top_actions = outputs['atom_action_logits'][0][:self.beam_width]
                else:  # Bond center
                    top_centers = outputs['top_bond_centers'][0][:self.beam_width]
                    top_actions = outputs['bond_action_logits'][0][:self.beam_width]
                
                # Generate new beam states (simplified)
                for k in range(min(len(top_centers), self.beam_width)):
                    # In full implementation, apply action to generate reactant SMILES
                    # For now, just record the prediction
                    new_state = BeamState(
                        smiles=beam.smiles,  # Placeholder - should be modified SMILES
                        score=beam.score - torch.log_softmax(top_actions[k], dim=-1).max().item(),
                        history=beam.history + [{
                            'step': step,
                            'rc_type': rc_type,
                            'rc_index': k,
                            'action': top_actions[k].argmax().item()
                        }],
                        terminated=False
                    )
                    new_beams.append(new_state)
            
            # Keep top beams
            new_beams.sort(key=lambda b: b.score)
            beams = new_beams[:self.beam_width]
            
            # Check if all terminated
            if all(b.terminated for b in beams):
                break
        
        # Format results
        results = []
        for beam in beams:
            results.append({
                'reactants': [beam.smiles],  # Placeholder
                'score': beam.score,
                'steps': beam.history
            })
        
        return results if return_all else results[:1]


if __name__ == '__main__':
    # Test beam search
    print("Beam search inference module ready")
    print("Note: Full implementation requires reactant generation logic")
