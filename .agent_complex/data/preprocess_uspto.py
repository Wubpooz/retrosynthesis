"""
Data preprocessing utilities for USPTO-50K dataset.
Handles SMILES canonicalization, reaction center extraction, and action vocabulary building.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from tqdm import tqdm


class USPTO50KPreprocessor:
    """
    Preprocesses USPTO-50K dataset for hierarchical retrosynthesis.
    
    Tasks:
    1. Canonicalize SMILES
    2. Randomize atom mapping to prevent leakage
    3. Extract reaction centers (atom/bond changes)
    4. Build action vocabularies
    5. Create train/val/test splits
    """
    
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vocabularies
        self.atom_action_vocab = {}
        self.bond_action_vocab = {}
        self.leaving_group_vocab = {}
        
        # Statistics
        self.stats = defaultdict(int)
        
    def load_raw_data(self) -> List[Dict]:
        """Load raw USPTO-50K reactions."""
        reactions = []
        
        # Assuming data is in format: product>>reactants
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rxn_smiles = parts[0]
                    rxn_class = int(parts[1]) if len(parts) > 1 else -1
                    
                    reactions.append({
                        'rxn_smiles': rxn_smiles,
                        'rxn_class': rxn_class
                    })
        
        print(f"Loaded {len(reactions)} reactions")
        return reactions
    
    def canonicalize_smiles(self, smiles: str, randomize_atoms: bool = False) -> Optional[str]:
        """Canonicalize SMILES string with optional atom mapping randomization."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if randomize_atoms:
                # Randomize atom order
                atom_order = list(range(mol.GetNumAtoms()))
                np.random.shuffle(atom_order)
                mol = Chem.RenumberAtoms(mol, atom_order)
            
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def extract_reaction_center(self, product_mol: Chem.Mol, 
                               reactant_mols: List[Chem.Mol]) -> Dict:
        """
        Extract reaction center by comparing product and reactants.
        
        Returns:
            Dict with 'atoms_changed', 'bonds_changed', 'atom_actions', 'bond_actions'
        """
        rc_info = {
            'atoms_changed': [],
            'bonds_changed': [],
            'atom_actions': [],
            'bond_actions': []
        }
        
        # Combine all reactants
        reactant_mol = reactant_mols[0]
        for mol in reactant_mols[1:]:
            reactant_mol = Chem.CombineMols(reactant_mol, mol)
        
        # Find atom mapping
        product_atoms = {atom.GetAtomMapNum(): atom for atom in product_mol.GetAtoms() 
                        if atom.GetAtomMapNum() > 0}
        reactant_atoms = {atom.GetAtomMapNum(): atom for atom in reactant_mol.GetAtoms()
                         if atom.GetAtomMapNum() > 0}
        
        # Compare atoms
        for map_num in product_atoms:
            if map_num in reactant_atoms:
                p_atom = product_atoms[map_num]
                r_atom = reactant_atoms[map_num]
                
                # Check for changes
                if (p_atom.GetTotalNumHs() != r_atom.GetTotalNumHs() or
                    p_atom.GetFormalCharge() != r_atom.GetFormalCharge() or
                    p_atom.GetChiralTag() != r_atom.GetChiralTag()):
                    
                    rc_info['atoms_changed'].append(map_num)
                    
                    action = {
                        'type': 'atom',
                        'h_change': p_atom.GetTotalNumHs() - r_atom.GetTotalNumHs(),
                        'charge_change': p_atom.GetFormalCharge() - r_atom.GetFormalCharge(),
                        'chirality_change': str(p_atom.GetChiralTag()) != str(r_atom.GetChiralTag())
                    }
                    rc_info['atom_actions'].append(action)
        
        # Compare bonds
        product_bonds = {}
        for bond in product_mol.GetBonds():
            begin = bond.GetBeginAtom().GetAtomMapNum()
            end = bond.GetEndAtom().GetAtomMapNum()
            if begin > 0 and end > 0:
                key = tuple(sorted([begin, end]))
                product_bonds[key] = bond
        
        reactant_bonds = {}
        for bond in reactant_mol.GetBonds():
            begin = bond.GetBeginAtom().GetAtomMapNum()
            end = bond.GetEndAtom().GetAtomMapNum()
            if begin > 0 and end > 0:
                key = tuple(sorted([begin, end]))
                reactant_bonds[key] = bond
        
        # Find changed/deleted bonds
        for bond_key in product_bonds:
            if bond_key in reactant_bonds:
                p_bond = product_bonds[bond_key]
                r_bond = reactant_bonds[bond_key]
                
                if p_bond.GetBondType() != r_bond.GetBondType():
                    rc_info['bonds_changed'].append(bond_key)
                    
                    action = {
                        'type': 'bond_change',
                        'from': str(r_bond.GetBondType()),
                        'to': str(p_bond.GetBondType())
                    }
                    rc_info['bond_actions'].append(action)
            else:
                # Bond deleted
                rc_info['bonds_changed'].append(bond_key)
                rc_info['bond_actions'].append({
                    'type': 'bond_delete',
                    'bond_type': str(product_bonds[bond_key].GetBondType())
                })
        
        # Find new bonds
        for bond_key in reactant_bonds:
            if bond_key not in product_bonds:
                rc_info['bonds_changed'].append(bond_key)
                rc_info['bond_actions'].append({
                    'type': 'bond_form',
                    'bond_type': str(reactant_bonds[bond_key].GetBondType())
                })
        
        return rc_info
    
    def build_action_vocabularies(self, processed_reactions: List[Dict]):
        """Build vocabularies from extracted reaction centers."""
        atom_actions = Counter()
        bond_actions = Counter()
        
        for rxn in processed_reactions:
            rc_info = rxn.get('rc_info', {})
            
            for action in rc_info.get('atom_actions', []):
                action_str = json.dumps(action, sort_keys=True)
                atom_actions[action_str] += 1
            
            for action in rc_info.get('bond_actions', []):
                action_str = json.dumps(action, sort_keys=True)
                bond_actions[action_str] += 1
        
        # Create vocabularies (keep actions appearing at least 5 times)
        self.atom_action_vocab = {
            action: idx for idx, (action, count) in 
            enumerate(atom_actions.most_common())
            if count >= 5
        }
        
        self.bond_action_vocab = {
            action: idx for idx, (action, count) in 
            enumerate(bond_actions.most_common())
            if count >= 5
        }
        
        # Add special tokens
        self.atom_action_vocab['<UNK>'] = len(self.atom_action_vocab)
        self.bond_action_vocab['<UNK>'] = len(self.bond_action_vocab)
        
        print(f"Atom action vocabulary size: {len(self.atom_action_vocab)}")
        print(f"Bond action vocabulary size: {len(self.bond_action_vocab)}")
    
    def process_dataset(self, train_ratio: float = 0.8, 
                       val_ratio: float = 0.1) -> Dict:
        """
        Main processing pipeline.
        
        Returns:
            Dict with processed data and statistics
        """
        print("Loading raw data...")
        reactions = self.load_raw_data()
        
        print("Processing reactions...")
        processed_reactions = []
        
        for rxn in tqdm(reactions):
            try:
                # Parse reaction
                parts = rxn['rxn_smiles'].split('>>')
                if len(parts) != 2:
                    continue
                
                product_smiles = parts[0]
                reactants_smiles = parts[1]
                
                # Canonicalize
                product_smiles = self.canonicalize_smiles(product_smiles)
                reactants = [self.canonicalize_smiles(r) for r in reactants_smiles.split('.')]
                
                if product_smiles is None or None in reactants:
                    continue
                
                # Create molecules
                product_mol = Chem.MolFromSmiles(product_smiles)
                reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
                
                if product_mol is None or None in reactant_mols:
                    continue
                
                # Extract reaction center
                rc_info = self.extract_reaction_center(product_mol, reactant_mols)
                
                processed_reactions.append({
                    'product': product_smiles,
                    'reactants': reactants,
                    'rxn_class': rxn['rxn_class'],
                    'rc_info': rc_info
                })
                
                # Update statistics
                self.stats['total_reactions'] += 1
                self.stats[f'rxn_class_{rxn["rxn_class"]}'] += 1
                self.stats[f'rc_count_{len(rc_info["atoms_changed"])}'] += 1
                
            except Exception as e:
                print(f"Error processing reaction: {e}")
                continue
        
        print(f"Successfully processed {len(processed_reactions)} reactions")
        
        # Build vocabularies
        print("Building action vocabularies...")
        self.build_action_vocabularies(processed_reactions)
        
        # Split data
        np.random.shuffle(processed_reactions)
        n_train = int(len(processed_reactions) * train_ratio)
        n_val = int(len(processed_reactions) * val_ratio)
        
        splits = {
            'train': processed_reactions[:n_train],
            'val': processed_reactions[n_train:n_train + n_val],
            'test': processed_reactions[n_train + n_val:]
        }
        
        # Save processed data
        print("Saving processed data...")
        with open(self.output_dir / 'uspto_50k_processed.pkl', 'wb') as f:
            pickle.dump(splits, f)
        
        with open(self.output_dir / 'action_vocab.json', 'w') as f:
            json.dump({
                'atom_actions': self.atom_action_vocab,
                'bond_actions': self.bond_action_vocab
            }, f, indent=2)
        
        with open(self.output_dir / 'stats.json', 'w') as f:
            json.dump(dict(self.stats), f, indent=2)
        
        print("Preprocessing complete!")
        return {
            'splits': splits,
            'stats': dict(self.stats),
            'vocabularies': {
                'atom_actions': self.atom_action_vocab,
                'bond_actions': self.bond_action_vocab
            }
        }


if __name__ == '__main__':
    # Example usage
    preprocessor = USPTO50KPreprocessor(
        data_path='data/raw/uspto_50k.txt',
        output_dir='data/processed'
    )
    
    results = preprocessor.process_dataset()
    print("\nDataset Statistics:")
    print(f"Train: {len(results['splits']['train'])}")
    print(f"Val: {len(results['splits']['val'])}")
    print(f"Test: {len(results['splits']['test'])}")
