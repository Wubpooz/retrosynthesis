"""
PyTorch dataset for retrosynthesis.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


class RetrosynthesisDataset(Dataset):
    """
    Dataset for hierarchical retrosynthesis prediction.
    
    Each sample contains:
    - Product molecular graph
    - Reaction center information
    - Action labels
    - Termination label
    """
    
    def __init__(self, data_path: str, split: str = 'train', 
                 vocab_path: Optional[str] = None):
        """
        Args:
            data_path: Path to processed data pickle
            split: 'train', 'val', or 'test'
            vocab_path: Path to action vocabulary JSON
        """
        self.split = split
        
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.reactions = data[split]
        
        # Load vocabulary if provided
        self.vocab = None
        if vocab_path:
            import json
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        
        print(f"Loaded {len(self.reactions)} {split} reactions")
    
    def __len__(self) -> int:
        return len(self.reactions)
    
    def mol_to_graph(self, smiles: str) -> Data:
        """
        Convert SMILES to PyTorch Geometric graph.
        
        Returns:
            PyG Data object with:
            - x: atom features [num_atoms, atom_dim]
            - edge_index: [2, num_edges]
            - edge_attr: bond features [num_edges, bond_dim]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Atom features
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
        
        # Edge index and features
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions
            edge_index.extend([[i, j], [j, i]])
            
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo()),
            ]
            
            # Add for both directions
            edge_features.extend([bond_features, bond_features])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample.
        
        Returns:
            Dict with:
            - graph: PyG Data object for product
            - rc_type: 0 for atom, 1 for bond
            - rc_atoms: indices of reaction center atoms
            - rc_bonds: indices of reaction center bonds
            - atom_actions: action indices for atoms
            - bond_actions: action indices for bonds
            - terminate: whether this is final step
        """
        rxn = self.reactions[idx]
        
        # Convert product to graph
        graph = self.mol_to_graph(rxn['product'])
        
        # Extract reaction center info
        rc_info = rxn['rc_info']
        
        # Determine RC type (simplified: atom if only atoms changed, else bond)
        rc_type = 0 if rc_info['bonds_changed'] == [] else 1
        
        # Get RC indices
        rc_atoms = torch.tensor(rc_info['atoms_changed'], dtype=torch.long)
        rc_bonds = torch.tensor([
            tuple(sorted(bond)) for bond in rc_info['bonds_changed']
        ], dtype=torch.long) if rc_info['bonds_changed'] else torch.empty(0, 2, dtype=torch.long)
        
        # Map actions to vocabulary indices (if vocab provided)
        atom_action_indices = []
        if self.vocab:
            import json
            for action in rc_info['atom_actions']:
                action_str = json.dumps(action, sort_keys=True)
                idx = self.vocab['atom_actions'].get(action_str, 
                                                     self.vocab['atom_actions']['<UNK>'])
                atom_action_indices.append(idx)
        
        atom_actions = torch.tensor(atom_action_indices, dtype=torch.long) if atom_action_indices else torch.empty(0, dtype=torch.long)
        
        bond_action_indices = []
        if self.vocab:
            import json
            for action in rc_info['bond_actions']:
                action_str = json.dumps(action, sort_keys=True)
                idx = self.vocab['bond_actions'].get(action_str,
                                                     self.vocab['bond_actions']['<UNK>'])
                bond_action_indices.append(idx)
        
        bond_actions = torch.tensor(bond_action_indices, dtype=torch.long) if bond_action_indices else torch.empty(0, dtype=torch.long)
        
        # Termination (True if single-step reaction)
        terminate = len(rc_info['atoms_changed']) + len(rc_info['bonds_changed']) <= 3
        
        return {
            'graph': graph,
            'rc_type': torch.tensor(rc_type, dtype=torch.long),
            'rc_atoms': rc_atoms,
            'rc_bonds': rc_bonds,
            'atom_actions': atom_actions,
            'bond_actions': bond_actions,
            'terminate': torch.tensor(int(terminate), dtype=torch.long),
            'rxn_class': torch.tensor(rxn['rxn_class'], dtype=torch.long),
            'product_smiles': rxn['product'],
            'reactants_smiles': rxn['reactants']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching heterogeneous graphs.
    """
    from torch_geometric.data import Batch
    
    # Batch graphs
    graphs = Batch.from_data_list([item['graph'] for item in batch])
    
    # Stack scalar targets
    rc_types = torch.stack([item['rc_type'] for item in batch])
    terminates = torch.stack([item['terminate'] for item in batch])
    rxn_classes = torch.stack([item['rxn_class'] for item in batch])
    
    # Keep variable-length tensors as lists
    rc_atoms = [item['rc_atoms'] for item in batch]
    rc_bonds = [item['rc_bonds'] for item in batch]
    atom_actions = [item['atom_actions'] for item in batch]
    bond_actions = [item['bond_actions'] for item in batch]
    
    # Keep SMILES strings
    product_smiles = [item['product_smiles'] for item in batch]
    reactants_smiles = [item['reactants_smiles'] for item in batch]
    
    return {
        'graph': graphs,
        'rc_type': rc_types,
        'rc_atoms': rc_atoms,
        'rc_bonds': rc_bonds,
        'atom_actions': atom_actions,
        'bond_actions': bond_actions,
        'terminate': terminates,
        'rxn_class': rxn_classes,
        'product_smiles': product_smiles,
        'reactants_smiles': reactants_smiles
    }


if __name__ == '__main__':
    # Test dataset
    dataset = RetrosynthesisDataset(
        data_path='data/processed/uspto_50k_processed.pkl',
        split='train',
        vocab_path='data/processed/action_vocab.json'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
