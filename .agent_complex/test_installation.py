"""
Quick test script to verify HierRetro installation and components.
Run this after installing dependencies to check everything works.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("HierRetro Installation Test")
print("="*60)

# Test 1: Basic imports
print("\n[1/7] Testing basic imports...")
try:
    import torch
    import torch_geometric
    from rdkit import Chem
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ PyTorch Geometric {torch_geometric.__version__}")
    print(f"  ✓ RDKit available")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model imports
print("\n[2/7] Testing model imports...")
try:
    from models import HierRetro
    from models.encoders import UniMolPlusEncoder, MambaEncoder, GraphLinearizer
    from models.prediction_heads import (
        RCTypePredictor, AtomCenterPredictor, BondCenterPredictor,
        AtomActionPredictor, BondActionPredictor, TerminationPredictor
    )
    print("  ✓ All model components imported")
except ImportError as e:
    print(f"  ✗ Model import failed: {e}")
    sys.exit(1)

# Test 3: Training imports
print("\n[3/7] Testing training imports...")
try:
    from training.damt_loss import DAMTLoss
    print("  ✓ Training components imported")
except ImportError as e:
    print(f"  ✗ Training import failed: {e}")
    sys.exit(1)

# Test 4: Data imports
print("\n[4/7] Testing data imports...")
try:
    from data.dataset import RetrosynthesisDataset, collate_fn
    print("  ✓ Data components imported")
except ImportError as e:
    print(f"  ✗ Data import failed: {e}")
    sys.exit(1)

# Test 5: Evaluation imports
print("\n[5/7] Testing evaluation imports...")
try:
    from evaluation.metrics import (
        compute_exact_match_accuracy,
        compute_rc_identification_accuracy,
        MetricsTracker
    )
    print("  ✓ Evaluation components imported")
except ImportError as e:
    print(f"  ✗ Evaluation import failed: {e}")
    sys.exit(1)

# Test 6: GNN model creation
print("\n[6/7] Testing GNN model creation...")
try:
    model_gnn = HierRetro(encoder_type='gnn')
    num_params = sum(p.numel() for p in model_gnn.parameters())
    print(f"  ✓ GNN model created: {num_params:,} parameters")
    
    # Test forward pass with dummy data
    from torch_geometric.data import Data, Batch
    x = torch.randn(5, 7)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data])
    
    with torch.no_grad():
        outputs = model_gnn(batch)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - RC type logits: {outputs['rc_type_logits'].shape}")
    print(f"    - Graph features: {outputs['graph_features'].shape}")
    
except Exception as e:
    print(f"  ✗ GNN test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Mamba model (optional)
print("\n[7/7] Testing Mamba model (optional)...")
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

if MAMBA_AVAILABLE:
    try:
        model_mamba = HierRetro(encoder_type='mamba')
        num_params = sum(p.numel() for p in model_mamba.parameters())
        print(f"  ✓ Mamba model created: {num_params:,} parameters")
    except Exception as e:
        print(f"  ✗ Mamba test failed: {e}")
else:
    print("  ⚠ Mamba not installed (optional)")
    print("    Install with: pip install mamba-ssm causal-conv1d")

# Summary
print("\n" + "="*60)
print("✅ Installation test complete!")
print("="*60)
print("\nNext steps:")
print("1. Download USPTO-50K dataset")
print("2. Run: python data/preprocess_uspto.py")
print("3. Train: python training/train.py --config configs/gnn_config.yaml")
print("\nSee README_IMPLEMENTATION.md for detailed instructions.")
