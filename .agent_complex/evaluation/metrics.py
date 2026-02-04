"""
Evaluation metrics for retrosynthesis.
"""

import torch
from typing import List, Dict
from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None


def compute_exact_match_accuracy(predictions: List[List[str]], 
                                 ground_truth: List[List[str]], 
                                 k: int = 1) -> float:
    """
    Top-k exact match accuracy.
    
    Args:
        predictions: List of [k predictions] for each sample
        ground_truth: List of [reactants] for each sample
        k: Top-k accuracy
        
    Returns:
        Accuracy (0-1)
    """
    correct = 0
    
    for pred_list, gt_list in zip(predictions, ground_truth):
        # Canonicalize
        gt_canon = set(canonicalize_smiles(s) for s in gt_list)
        gt_canon.discard(None)
        
        # Check if any prediction matches
        for pred in pred_list[:k]:
            pred_canon = set(canonicalize_smiles(s) for s in pred) if isinstance(pred, list) else {canonicalize_smiles(pred)}
            pred_canon.discard(None)
            
            if pred_canon == gt_canon:
                correct += 1
                break
    
    return correct / len(predictions)


def compute_rc_identification_accuracy(pred_rcs: List[List[int]], 
                                       true_rcs: List[List[int]]) -> float:
    """
    Reaction center identification accuracy.
    
    Args:
        pred_rcs: List of predicted RC atom/bond indices
        true_rcs: List of ground truth RC indices
        
    Returns:
        Accuracy (0-1)
    """
    correct = 0
    
    for pred, true in zip(pred_rcs, true_rcs):
        pred_set = set(pred) if pred else set()
        true_set = set(true) if true else set()
        
        if pred_set == true_set:
            correct += 1
    
    return correct / len(pred_rcs)


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Generic top-k accuracy.
    
    Args:
        logits: [batch, num_classes]
        targets: [batch]
        k: Top-k
        
    Returns:
        Accuracy (0-1)
    """
    _, top_k_preds = logits.topk(k, dim=-1)
    targets = targets.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == targets).any(dim=-1).sum().item()
    
    return correct / logits.size(0)


class MetricsTracker:
    """Track metrics during training/evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'rc_type_acc': [],
            'rc_loc_acc': [],
            'action_acc': [],
            'term_acc': [],
            'exact_match_acc': []
        }
    
    def update(self, metric_name: str, value: float):
        """Add a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def compute_average(self, metric_name: str) -> float:
        """Compute average of a metric."""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all metric averages."""
        return {name: self.compute_average(name) for name in self.metrics.keys()}


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics:")
    
    # Exact match test
    preds = [[['CC', 'O']], [['CCO']], [['CC=O', 'N']]]
    gt = [['CC', 'O'], ['CCO'], ['CC=O', 'NH3']]
    acc = compute_exact_match_accuracy(preds, gt, k=1)
    print(f"Exact match accuracy: {acc:.2f}")
    
    # Top-k test
    logits = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))
    acc = compute_top_k_accuracy(logits, targets, k=3)
    print(f"Top-3 accuracy: {acc:.2f}")
