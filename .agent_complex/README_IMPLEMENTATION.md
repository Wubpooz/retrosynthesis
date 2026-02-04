# HierRetro: Hierarchical Retrosynthesis Prediction Framework

A PyTorch implementation of hierarchical retrosynthesis prediction combining GNN and Mamba (SSM) encoders with interpretable prediction heads.

## Features

- **Dual Encoder Architecture**: Choose between GNN (Uni-Mol+ style) or Mamba (State Space Model)
- **Hierarchical Prediction**: RC type → RC localization → Actions → Termination
- **Dynamic Adaptive Multi-Task Learning (DAMT)**: Automatically balances task weights during training
- **Interpretable Pipeline**: Each decision step is explicit and traceable
- **CPU-Optimized Inference**: Efficient deployment on standard hardware

## Architecture Overview

```
Product Molecule
       ↓
   Encoder (GNN/Mamba)
       ↓
  ┌────┴────┐
  │ RC Type │  (Atom vs Bond)
  └────┬────┘
       ↓
  ┌────┴────┐
  │ RC Loc. │  (Which atoms/bonds)
  └────┬────┘
       ↓
  ┌────┴────┐
  │ Actions │  (H-changes, leaving groups, etc.)
  └────┬────┘
       ↓
  ┌────┴────┐
  │  Term.  │  (Stop or continue)
  └─────────┘
```

## Installation

### 1. Create Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For Mamba encoder, install `mamba-ssm`:
```bash
pip install mamba-ssm causal-conv1d>=1.0.0
```

For GPU support, ensure PyTorch is installed with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Preprocess Data

```bash
python data/preprocess_uspto.py
```

This will:
- Load USPTO-50K dataset
- Canonicalize SMILES
- Extract reaction centers
- Build action vocabularies
- Create train/val/test splits

### 2. Train Model

**GNN Encoder:**
```bash
python training/train.py --config configs/gnn_config.yaml
```

**Mamba Encoder:**
```bash
python training/train.py --config configs/mamba_config.yaml
```

### 3. Inference

```python
from models import HierRetro
from inference.beam_search import BeamSearchRetrosynthesis
import torch

# Load model
model = HierRetro(encoder_type='gnn')
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict reactants
searcher = BeamSearchRetrosynthesis(model, beam_width=10)
results = searcher.predict('CCO')  # Product SMILES

print(f"Predicted reactants: {results[0]['reactants']}")
print(f"Score: {results[0]['score']:.4f}")
```

## Configuration

Edit `configs/gnn_config.yaml` or `configs/mamba_config.yaml`:

```yaml
model:
  encoder_type: "gnn"  # or "mamba"
  hidden_dim: 256
  num_layers: 6

training:
  batch_size: 32
  learning_rate: 0.0001
  max_epochs: 100
  
  damt_loss:
    queue_size: 50
    tau: 1.0

data:
  dataset: "uspto_50k"
  train_split: 0.8
```

## Project Structure

```
retrosynthesis/
├── data/                      # Data processing
│   ├── preprocess_uspto.py   # USPTO-50K preprocessing
│   └── dataset.py            # PyTorch Dataset
├── models/
│   ├── encoders/             # Molecular encoders
│   │   ├── base_encoder.py  # Abstract base
│   │   ├── gnn_encoder.py   # GNN (Uni-Mol+)
│   │   ├── mamba_encoder.py # Mamba SSM
│   │   └── graph_linearizer.py
│   ├── prediction_heads/     # Task-specific heads
│   │   ├── rc_type_predictor.py
│   │   ├── atom_center_predictor.py
│   │   ├── bond_center_predictor.py
│   │   ├── action_predictor.py
│   │   └── termination_predictor.py
│   └── hierretro.py          # Main model
├── training/
│   ├── train.py              # Training script
│   └── damt_loss.py          # Adaptive loss
├── inference/
│   ├── beam_search.py        # Beam search
│   └── explainability.py     # Visualization
├── evaluation/
│   └── metrics.py            # Evaluation metrics
├── configs/                   # YAML configs
│   ├── gnn_config.yaml
│   └── mamba_config.yaml
└── requirements.txt
```

## Key Features

### 1. Dynamic Adaptive Multi-Task Learning (DAMT)

Automatically balances task weights based on descent rates:
- Fast-descending tasks get lower weight
- Slow-descending tasks get higher weight
- Prevents any single task from dominating

### 2. Dual Encoder Support

**GNN (Uni-Mol+):**
- Pair-wise attention with bond features
- 6 transformer blocks
- 256-dim atom features
- ~10M parameters

**Mamba (SSM):**
- State-space sequential modeling
- O(n) complexity vs O(n²) for transformers
- Graph linearization (DFS/BFS)
- 12 layers, 768-dim
- ~20M parameters

### 3. Hierarchical Interpretability

Each prediction step is explicit:
1. RC Type: "Is this an atom or bond center?"
2. RC Localization: "Which atom/bond specifically?"
3. Action: "What chemical change occurs?"
4. Termination: "Is retrosynthesis complete?"

## Evaluation

```python
from evaluation.metrics import compute_exact_match_accuracy, MetricsTracker

tracker = MetricsTracker()

# During evaluation
tracker.update('exact_match_acc', accuracy)
tracker.update('rc_type_acc', rc_accuracy)

# Get results
results = tracker.get_all_averages()
print(f"Top-1 Accuracy: {results['exact_match_acc']:.2%}")
```

## Performance Targets

Based on HierRetro paper (USPTO-50K, unknown reaction type):

| Metric | Target | GNN | Mamba |
|--------|--------|-----|-------|
| Top-1 Accuracy | 52.3% | TBD | TBD |
| Top-5 Accuracy | 83.1% | TBD | TBD |
| Top-10 Accuracy | 89.1% | TBD | TBD |
| RC ID (Top-1) | 72.5% | TBD | TBD |
| Round-trip (Top-5) | 96.2% | TBD | TBD |

## GNN vs Mamba Comparison

| Aspect | GNN | Mamba |
|--------|-----|-------|
| Complexity | O(n²) | O(n) |
| Parameters | ~10M | ~20M |
| Training Time | Faster | Slower |
| Inference Speed (CPU) | Medium | **Fast** |
| Long Molecules (>50 atoms) | Slower | **2x Faster** |
| Interpretability | High | Medium |

## Citation

If you use this code, please cite:

```bibtex
@article{hierretro2024,
  title={HierRetro: Hierarchical Retrosynthesis Prediction},
  journal={arXiv preprint},
  year={2024}
}
```

## References

- **HierRetro**: Liu et al., 2024 (arXiv:2411.19503)
- **Uni-Mol+**: Zhou et al., 2023
- **Mamba**: Gu & Dao, 2023
- **USPTO-50K**: Coley et al., 2017

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.

---

**Status**: ✅ Core framework implemented | 🔄 Training in progress | 📊 Evaluation pending
