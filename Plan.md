# Hierarchical Retrosynthesis Framework Implementation Plan

## Project Overview
Build a hybrid retrosynthesis prediction system combining HierRetro's hierarchical decision-making with Mamba's efficient state-space modeling for explainable predictions and fast CPU inference.

---

## Phase 1: Environment Setup & Data Preparation

### 1.1 Development Environment
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse
pip install rdkit-pypi
pip install mamba-ssm causal-conv1d>=1.0.0
pip install transformers datasets
pip install wandb tensorboard
pip install pytest black flake8
```

### 1.2 USPTO-50K Data Processing
**File: `data/preprocess_uspto.py`**

```python
# Tasks:
1. Download USPTO-50K from deepchem/moleculenet
2. Canonicalize SMILES using RDKit
3. Randomize atom-mapping to prevent leakage
4. Extract reaction centers (atom/bond changes)
5. Build action vocabulary:
   - Atom actions: H-count changes, chirality, leaving groups
   - Bond actions: bond type changes, deletions
6. Create train/val/test splits (Coley et al. protocol)
7. Generate statistics: RC distribution, action frequencies
```

**Outputs:**
- `uspto_50k_processed.pkl`: Processed reactions
- `action_vocab.json`: Atom/bond action mappings
- `stats.json`: Dataset statistics

---

## Phase 2: Molecular Encoding Architecture

### 2.1 Dual Encoder System
**File: `models/encoders/base_encoder.py`**

```python
class MolecularEncoder(nn.Module):
    """Abstract base for GNN and Mamba encoders"""
    def forward(self, mol_graph):
        # Returns: (atom_features, pair_features, graph_features)
```

### 2.2 GNN Encoder (Uni-Mol+ Style)
**File: `models/encoders/gnn_encoder.py`**

```python
class UniMolPlusEncoder(MolecularEncoder):
    """
    Components:
    - Atom embedding: atomic number, formal charge, chirality
    - Bond embedding: bond type, aromaticity, ring membership
    - Transformer blocks with pair bias attention
    - 6 layers, hidden_dim=256, pair_dim=128
    """
    
    def __init__(self, config):
        self.atom_embedding = AtomEmbedding(dim=256)
        self.bond_embedding = BondEmbedding(dim=128)
        self.transformer_blocks = nn.ModuleList([
            GraphTransformerBlock(
                atom_dim=256,
                pair_dim=128,
                num_heads=8
            ) for _ in range(6)
        ])
    
    def forward(self, batch):
        # 1. Embed atoms and bonds
        # 2. Apply transformer blocks with pair bias
        # 3. Return atom/pair/graph representations
```

### 2.3 Mamba Encoder (SSM-based)
**File: `models/encoders/mamba_encoder.py`**

```python
class MambaEncoder(MolecularEncoder):
    """
    State-space sequence model for molecules
    - Convert graph to linearized sequence (DFS/BFS)
    - Apply Mamba blocks for sequential modeling
    - 12 layers (2x GNN equivalent), hidden_dim=768
    - Linear complexity O(n) vs O(n²) attention
    """
    
    def __init__(self, config):
        self.atom_embedding = AtomEmbedding(dim=768)
        self.graph_linearizer = GraphLinearizer(strategy='dfs')
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=768,
                d_state=16,
                d_conv=4,
                expand=2
            ) for _ in range(12)
        ])
        
    def forward(self, batch):
        # 1. Linearize molecular graph
        # 2. Apply Mamba blocks
        # 3. Map back to graph structure
        # 4. Return atom/pair/graph features
```

**File: `models/encoders/graph_linearizer.py`**
```python
class GraphLinearizer:
    """Convert molecular graph to sequence"""
    def dfs_linearize(self, mol_graph):
        # Depth-first traversal with backtracking tokens
    
    def bfs_linearize(self, mol_graph):
        # Breadth-first traversal
    
    def reconstruct_graph(self, sequence, positions):
        # Map sequential features back to graph nodes
```

---

## Phase 3: Hierarchical Prediction Modules

### 3.1 Reaction Center Type Prediction (RCP)
**File: `models/prediction_heads/rc_type_predictor.py`**

```python
class RCTypePredictor(nn.Module):
    """Binary classifier: atom vs bond center"""
    
    def __init__(self, hidden_dim):
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, graph_features):
        # Input: [batch, hidden_dim]
        # Output: [batch, 1] probability of bond center
        return self.classifier(graph_features)
```

### 3.2 Atom Center Localization (AC)
**File: `models/prediction_heads/atom_center_predictor.py`**

```python
class AtomCenterPredictor(nn.Module):
    """Predict which atom is the reaction center"""
    
    def __init__(self, atom_dim):
        self.scorer = nn.Linear(atom_dim, 1)
    
    def forward(self, atom_features):
        # Input: [batch, num_atoms, atom_dim]
        # Output: [batch, num_atoms] logits
        scores = self.scorer(atom_features).squeeze(-1)
        return F.log_softmax(scores, dim=-1)
```

### 3.3 Bond Center Localization (BC)
**File: `models/prediction_heads/bond_center_predictor.py`**

```python
class BondCenterPredictor(nn.Module):
    """Predict which bond is the reaction center"""
    
    def __init__(self, pair_dim):
        self.scorer = nn.Linear(pair_dim, 1)
    
    def forward(self, pair_features, edge_index):
        # Input: [batch, num_pairs, pair_dim]
        # Output: [batch, num_bonds] logits
        scores = self.scorer(pair_features).squeeze(-1)
        return F.log_softmax(scores, dim=-1)
```

### 3.4 Action Prediction
**File: `models/prediction_heads/action_predictor.py`**

```python
class AtomActionPredictor(nn.Module):
    """Predict atom-level edits (H-count, chirality, LG)"""
    
    def __init__(self, atom_dim, action_vocab_size):
        self.action_classifier = nn.Linear(atom_dim, action_vocab_size)
    
    def forward(self, atom_features, rc_indices):
        # Extract features at RC locations
        # Predict action probabilities

class BondActionPredictor(nn.Module):
    """Predict bond-level edits (type change, deletion)"""
    
    def __init__(self, pair_dim, action_vocab_size):
        self.bond_classifier = nn.Linear(pair_dim, action_vocab_size)
        self.atom_classifier = nn.Linear(atom_dim, atom_action_vocab_size)
    
    def forward(self, pair_features, atom_features, rc_bond_indices):
        # Predict bond action
        # Predict atom actions for both endpoints
```

### 3.5 Termination Predictor
**File: `models/prediction_heads/termination_predictor.py`**

```python
class TerminationPredictor(nn.Module):
    """Decide if retrosynthesis process should stop"""
    
    def __init__(self, hidden_dim):
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, graph_features):
        return self.classifier(graph_features)
```

---

## Phase 4: Unified Framework

### 4.1 HierRetro Model
**File: `models/hierretro.py`**

```python
class HierRetro(nn.Module):
    def __init__(self, encoder_type='gnn', config=None):
        # Choose encoder
        if encoder_type == 'gnn':
            self.encoder = UniMolPlusEncoder(config)
        elif encoder_type == 'mamba':
            self.encoder = MambaEncoder(config)
        
        # Prediction heads
        self.rc_type_pred = RCTypePredictor(config.hidden_dim)
        self.atom_center_pred = AtomCenterPredictor(config.atom_dim)
        self.bond_center_pred = BondCenterPredictor(config.pair_dim)
        self.atom_action_pred = AtomActionPredictor(...)
        self.bond_action_pred = BondActionPredictor(...)
        self.termination_pred = TerminationPredictor(config.hidden_dim)
    
    def forward(self, product_graph, history=None):
        # 1. Encode molecular graph
        atom_feats, pair_feats, graph_feats = self.encoder(product_graph)
        
        # Incorporate history for multi-step
        if history is not None:
            atom_feats = self.update_with_history(atom_feats, history)
        
        # 2. Hierarchical prediction
        rc_type_prob = self.rc_type_pred(graph_feats)
        
        # 3. Locate reaction center based on type
        atom_probs = self.atom_center_pred(atom_feats)
        bond_probs = self.bond_center_pred(pair_feats, edges)
        
        # Combine using rc_type_prob as weight
        rc_probs = self.combine_rc_probs(
            rc_type_prob, atom_probs, bond_probs
        )
        
        # 4. Predict actions
        actions = self.predict_actions(
            atom_feats, pair_feats, rc_probs
        )
        
        # 5. Termination decision
        terminate = self.termination_pred(graph_feats)
        
        return {
            'rc_type': rc_type_prob,
            'rc_location': rc_probs,
            'actions': actions,
            'terminate': terminate
        }
```

### 4.2 Dynamic Adaptive Multi-Task Learning (DAMT)
**File: `training/damt_loss.py`**

```python
class DAMTLoss(nn.Module):
    """Dynamic loss weighting based on descent rates"""
    
    def __init__(self, num_tasks=4, queue_size=50, tau=1.0):
        self.loss_queue = {i: [] for i in range(num_tasks)}
        self.queue_size = queue_size
        self.tau = tau
    
    def compute_weights(self, current_losses):
        # 1. Compute descent rates
        weights = []
        for i, loss in enumerate(current_losses):
            if len(self.loss_queue[i]) >= 2:
                rate = self.loss_queue[i][-1] / self.loss_queue[i][-2]
            else:
                rate = 1.0
            
            # 2. Normalize by magnitude
            avg_loss = np.mean(self.loss_queue[i][-self.queue_size:])
            alpha = len(self.loss_queue[i]) / avg_loss if avg_loss > 0 else 1.0
            
            weights.append(np.exp(rate / self.tau))
        
        # 3. Normalize weights
        weights = np.array(weights)
        weights /= weights.sum()
        
        return weights
    
    def forward(self, losses_dict):
        # losses_dict = {'rc_type': ..., 'rc_loc': ..., 'action': ..., 'term': ...}
        losses = [losses_dict[k] for k in sorted(losses_dict.keys())]
        weights = self.compute_weights(losses)
        
        # Update queue
        for i, loss in enumerate(losses):
            self.loss_queue[i].append(loss.item())
            if len(self.loss_queue[i]) > self.queue_size:
                self.loss_queue[i].pop(0)
        
        # Weighted sum
        total_loss = sum(w * l for w, l in zip(weights, losses))
        return total_loss, weights
```

---

## Phase 5: Training Pipeline

### 5.1 Contrastive Pre-training (Optional)
**File: `training/pretrain_contrastive.py`**

```python
class ContrastiveLearning:
    """Pre-train encoder on 2D-3D alignment (3D-InfoMax style)"""
    
    def __init__(self, encoder_2d, encoder_3d):
        self.encoder_2d = encoder_2d  # Uses only SMILES/graph
        self.encoder_3d = encoder_3d  # Uses 3D conformers
    
    def contrastive_loss(self, z_2d, z_3d):
        # InfoNCE loss: maximize agreement between 2D and 3D views
        similarity = F.cosine_similarity(z_2d, z_3d)
        return -torch.log(similarity).mean()
    
    def train_step(self, batch):
        # Generate conformers with RDKit MMFF
        z_2d = self.encoder_2d(batch.graph_2d)
        z_3d = self.encoder_3d(batch.conformer_3d)
        loss = self.contrastive_loss(z_2d, z_3d)
        return loss
```

### 5.2 Main Training Loop
**File: `training/train.py`**

```python
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        self.scheduler = PolynomialDecayLR(optimizer, warmup_steps=1000)
        self.damt_loss = DAMTLoss(num_tasks=4)
    
    def train_epoch(self, dataloader):
        for batch in dataloader:
            # Forward pass
            outputs = self.model(batch.product_graph, batch.history)
            
            # Compute individual losses
            loss_rc_type = F.binary_cross_entropy(
                outputs['rc_type'], batch.rc_type_label
            )
            loss_rc_loc = F.nll_loss(
                outputs['rc_location'], batch.rc_location_label
            )
            loss_action = self.compute_action_loss(
                outputs['actions'], batch.actions
            )
            loss_term = F.binary_cross_entropy(
                outputs['terminate'], batch.terminate_label
            )
            
            # DAMT weighting
            losses = {
                'rc_type': loss_rc_type,
                'rc_loc': loss_rc_loc,
                'action': loss_action,
                'term': loss_term
            }
            total_loss, weights = self.damt_loss(losses)
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
```

### 5.3 Multi-Step Data Augmentation
**File: `data/augmentation.py`**

```python
class MultiRCPermutationAugmenter:
    """Augment reactions with multiple RCs by permuting order"""
    
    def augment(self, reaction, max_permutations=24):
        if reaction.num_rcs <= 1:
            return [reaction]
        
        # Generate all permutations (limited to max_permutations)
        rc_orders = list(permutations(reaction.rcs))[:max_permutations]
        
        augmented = []
        for order in rc_orders:
            # Apply RCs in this order
            augmented.append(self.apply_rc_sequence(reaction, order))
        
        return augmented
```

---

## Phase 6: Inference & Explainability

### 6.1 Beam Search Inference
**File: `inference/beam_search.py`**

```python
class BeamSearchRetrosynthesis:
    def __init__(self, model, beam_width=10):
        self.model = model
        self.beam_width = beam_width
    
    def search(self, product_smiles):
        # Initialize beam with product
        beam = [{'mol': product_smiles, 'score': 0.0, 'history': []}]
        
        while not self.all_terminated(beam):
            candidates = []
            
            for state in beam:
                if state['terminated']:
                    candidates.append(state)
                    continue
                
                # Predict next step
                outputs = self.model(state['mol'], state['history'])
                
                # Beam search at each decision level
                rc_candidates = self.beam_search_rc(outputs, k=beam_width)
                
                for rc_cand in rc_candidates:
                    action_candidates = self.beam_search_action(
                        outputs, rc_cand, k=beam_width
                    )
                    
                    for action_cand in action_candidates:
                        # Apply action to molecule
                        new_mol = self.apply_action(
                            state['mol'], rc_cand, action_cand
                        )
                        
                        new_state = {
                            'mol': new_mol,
                            'score': state['score'] + action_cand['score'],
                            'history': state['history'] + [action_cand],
                            'terminated': outputs['terminate'] > 0.5
                        }
                        candidates.append(new_state)
            
            # Keep top-k by score
            beam = sorted(candidates, key=lambda x: x['score'], reverse=True)[:self.beam_width]
        
        return beam
```

### 6.2 Attention Visualization
**File: `inference/explainability.py`**

```python
class AttentionVisualizer:
    """Visualize which atoms/bonds the model focuses on"""
    
    def get_attention_scores(self, model, product_graph):
        # Extract attention weights from encoder
        with torch.no_grad():
            atom_feats, pair_feats, _ = model.encoder(product_graph)
            
            # For GNN: extract attention from graph transformer
            if hasattr(model.encoder, 'transformer_blocks'):
                attn_weights = [
                    block.attention.last_attn_weights 
                    for block in model.encoder.transformer_blocks
                ]
            
            # For Mamba: use state activations as proxy
            elif hasattr(model.encoder, 'mamba_blocks'):
                attn_weights = [
                    block.get_state_importance()
                    for block in model.encoder.mamba_blocks
                ]
        
        return attn_weights
    
    def visualize_on_molecule(self, mol, attn_scores):
        # Color atoms by attention intensity
        from rdkit.Chem import Draw
        
        atom_colors = {}
        for i, score in enumerate(attn_scores):
            # Red = high attention, white = low
            atom_colors[i] = (1.0, 1.0-score, 1.0-score)
        
        img = Draw.MolToImage(mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                              highlightAtomColors=atom_colors)
        return img
```

### 6.3 Decision Path Explanation
**File: `inference/decision_path.py`**

```python
class DecisionPathExplainer:
    """Generate human-readable explanation of model decisions"""
    
    def explain(self, prediction_outputs, product_mol):
        explanation = {
            'product': Chem.MolToSmiles(product_mol),
            'steps': []
        }
        
        # RC type decision
        rc_type = 'bond' if prediction_outputs['rc_type'] > 0.5 else 'atom'
        explanation['steps'].append({
            'decision': 'RC Type',
            'choice': rc_type,
            'confidence': float(prediction_outputs['rc_type'])
        })
        
        # RC location
        rc_idx = prediction_outputs['rc_location'].argmax()
        if rc_type == 'atom':
            rc_atom = product_mol.GetAtomWithIdx(int(rc_idx))
            explanation['steps'].append({
                'decision': 'RC Location',
                'atom_idx': int(rc_idx),
                'atom_symbol': rc_atom.GetSymbol(),
                'confidence': float(prediction_outputs['rc_location'][rc_idx])
            })
        else:
            # Bond indices
            explanation['steps'].append({
                'decision': 'RC Location',
                'bond_idx': int(rc_idx),
                'confidence': float(prediction_outputs['rc_location'][rc_idx])
            })
        
        # Actions
        action = prediction_outputs['actions']
        explanation['steps'].append({
            'decision': 'Chemical Action',
            'action_type': action['type'],
            'details': action['params']
        })
        
        # Termination
        explanation['terminate'] = bool(prediction_outputs['terminate'] > 0.5)
        
        return explanation
```

---

## Phase 7: Mamba vs GNN Comparison

### 7.1 Benchmark Suite
**File: `evaluation/benchmark.py`**

```python
class RetrosynthesisBenchmark:
    def __init__(self, test_data):
        self.test_data = test_data
        self.metrics = {
            'top_k_accuracy': [1, 3, 5, 10],
            'rc_identification_acc': True,
            'inference_time': True,
            'memory_usage': True
        }
    
    def evaluate_model(self, model, device='cpu'):
        results = {}
        
        # Top-k accuracy
        for k in self.metrics['top_k_accuracy']:
            acc = self.compute_topk_accuracy(model, k, device)
            results[f'top_{k}_acc'] = acc
        
        # RC identification
        if self.metrics['rc_identification_acc']:
            results['rc_acc'] = self.compute_rc_accuracy(model, device)
        
        # Inference speed
        if self.metrics['inference_time']:
            import time
            times = []
            for batch in self.test_data:
                batch = batch.to(device)
                start = time.time()
                with torch.no_grad():
                    _ = model(batch.product_graph)
                times.append(time.time() - start)
            
            results['avg_inference_time'] = np.mean(times)
            results['throughput'] = len(self.test_data) / sum(times)
        
        # Memory
        if self.metrics['memory_usage']:
            if device == 'cpu':
                import psutil
                process = psutil.Process()
                results['memory_mb'] = process.memory_info().rss / 1024**2
            else:
                results['memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        return results
    
    def compare_models(self, gnn_model, mamba_model):
        print("Evaluating GNN model...")
        gnn_results = self.evaluate_model(gnn_model, device='cpu')
        
        print("Evaluating Mamba model...")
        mamba_results = self.evaluate_model(mamba_model, device='cpu')
        
        # Print comparison table
        self.print_comparison(gnn_results, mamba_results)
```

### 7.2 CPU Optimization
**File: `inference/cpu_optimize.py`**

```python
class CPUOptimizer:
    """Optimize models for CPU inference"""
    
    @staticmethod
    def quantize_model(model):
        """Apply dynamic quantization"""
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    
    @staticmethod
    def convert_to_torchscript(model, example_input):
        """JIT compile for faster inference"""
        model.eval()
        traced = torch.jit.trace(model, example_input)
        return torch.jit.freeze(traced)
    
    @staticmethod
    def enable_mkldnn(model):
        """Use Intel MKL-DNN backend"""
        import torch.backends.mkldnn
        if torch.backends.mkldnn.is_available():
            model = torch.utils.mkldnn.to_mkldnn(model)
        return model
```

---

## Phase 8: Evaluation & Analysis

### 8.1 Standard Metrics
**File: `evaluation/metrics.py`**

```python
def compute_exact_match_accuracy(predictions, ground_truth, k=1):
    """Top-k exact match on reactant SMILES"""
    correct = 0
    for pred_list, gt in zip(predictions, ground_truth):
        gt_canon = canonicalize_smiles(gt)
        pred_canon = [canonicalize_smiles(p) for p in pred_list[:k]]
        if gt_canon in pred_canon:
            correct += 1
    return correct / len(predictions)

def compute_round_trip_accuracy(predictions, products, forward_model):
    """Use forward model to verify reactant validity"""
    correct = 0
    for reactants, product in zip(predictions, products):
        predicted_product = forward_model(reactants)
        if canonicalize_smiles(predicted_product) == canonicalize_smiles(product):
            correct += 1
    return correct / len(predictions)

def compute_rc_identification_accuracy(pred_rcs, true_rcs):
    """Accuracy of identifying correct reaction center"""
    correct = sum([
        set(pred) == set(true) 
        for pred, true in zip(pred_rcs, true_rcs)
    ])
    return correct / len(pred_rcs)
```

### 8.2 Ablation Studies
**File: `evaluation/ablation.py`**

```python
class AblationStudy:
    """Systematic component removal analysis"""
    
    def __init__(self, full_model, test_data):
        self.full_model = full_model
        self.test_data = test_data
    
    def ablate_component(self, component_name):
        """Remove or replace a component"""
        ablated_model = copy.deepcopy(self.full_model)
        
        if component_name == 'rc_type_classifier':
            # Replace with random baseline
            ablated_model.rc_type_pred = RandomClassifier()
        
        elif component_name == 'damt_loss':
            # Use uniform weights
            ablated_model.use_damt = False
        
        elif component_name == 'contrastive_pretrain':
            # Reset encoder weights
            ablated_model.encoder.reset_parameters()
        
        elif component_name == 'history_update':
            # Disable multi-step history
            ablated_model.use_history = False
        
        return ablated_model
    
    def run_study(self):
        components = [
            'rc_type_classifier',
            'damt_loss',
            'contrastive_pretrain',
            'history_update'
        ]
        
        results = {'full_model': self.evaluate(self.full_model)}
        
        for comp in components:
            print(f"Ablating {comp}...")
            ablated = self.ablate_component(comp)
            results[f'without_{comp}'] = self.evaluate(ablated)
        
        return results
```

---

## Phase 9: Deployment & Utilities

### 9.1 Inference API
**File: `api/inference_server.py`**

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load optimized model
model = load_optimized_model('checkpoints/best_model.pt')
model.eval()

@app.route('/predict', methods=['POST'])
def predict_retrosynthesis():
    data = request.json
    product_smiles = data['product']
    beam_width = data.get('beam_width', 10)
    
    # Run inference
    searcher = BeamSearchRetrosynthesis(model, beam_width)
    results = searcher.search(product_smiles)
    
    # Format results
    output = {
        'product': product_smiles,
        'predictions': [
            {
                'reactants': r['mol'],
                'score': r['score'],
                'pathway': r['history']
            }
            for r in results[:10]
        ]
    }
    
    return jsonify(output)

@app.route('/explain', methods=['POST'])
def explain_prediction():
    data = request.json
    product_smiles = data['product']
    
    # Get prediction with attention
    mol_graph = smiles_to_graph(product_smiles)
    outputs = model(mol_graph)
    
    # Generate explanation
    explainer = DecisionPathExplainer()
    explanation = explainer.explain(outputs, Chem.MolFromSmiles(product_smiles))
    
    # Visualize attention
    visualizer = AttentionVisualizer()
    attn_scores = visualizer.get_attention_scores(model, mol_graph)
    attn_img = visualizer.visualize_on_molecule(
        Chem.MolFromSmiles(product_smiles), 
        attn_scores[-1].mean(0)  # Last layer, averaged across heads
    )
    
    return jsonify({
        'explanation': explanation,
        'attention_map': encode_image_base64(attn_img)
    })
```

### 9.2 Model Serialization
**File: `utils/model_io.py`**

```python
def save_model(model, path, metadata=None):
    """Save model with metadata"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'encoder_type': model.encoder_type,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)

def load_model(path, device='cpu'):
    """Load model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    model = HierRetro(
        encoder_type=checkpoint['encoder_type'],
        config=checkpoint['config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model

def export_to_onnx(model, example_input, path):
    """Export for cross-platform deployment"""
    torch.onnx.export(
        model,
        example_input,
        path,
        opset_version=14,
        input_names=['product_graph'],
        output_names=['predictions'],
        dynamic_axes={
            'product_graph': {0: 'batch_size', 1: 'num_atoms'},
            'predictions': {0: 'batch_size'}
        }
    )
```

---

## Phase 10: Testing & Validation

### 10.1 Unit Tests
**File: `tests/test_models.py`**

```python
import pytest

def test_gnn_encoder_shape():
    encoder = UniMolPlusEncoder(config)
    batch = create_dummy_batch(num_atoms=20)
    atom_feats, pair_feats, graph_feats = encoder(batch)
    
    assert atom_feats.shape == (1, 20, 256)
    assert pair_feats.shape == (1, 20*20, 128)
    assert graph_feats.shape == (1, 256)

def test_mamba_encoder_shape():
    encoder = MambaEncoder(config)
    batch = create_dummy_batch(num_atoms=20)
    atom_feats, pair_feats, graph_feats = encoder(batch)
    
    assert atom_feats.shape == (1, 20, 768)

def test_rc_type_prediction():
    model = HierRetro(encoder_type='gnn')
    batch = create_dummy_batch()
    outputs = model(batch.product_graph)
    
    assert 'rc_type' in outputs
    assert outputs['rc_type'].shape == (1, 1)
    assert 0 <= outputs['rc_type'].item() <= 1

def test_multi_step_inference():
    model = HierRetro(encoder_type='mamba')
    product = "CCO"
    
    searcher = BeamSearchRetrosynthesis(model, beam_width=5)
    results = searcher.search(product)
    
    assert len(results) <= 5
    assert all('mol' in r for r in results)
```

### 10.2 Integration Tests
**File: `tests/test_pipeline.py`**

```python
def test_full_pipeline():
    # 1. Load data
    data = load_uspto_sample()
    
    # 2. Train model
    model = HierRetro(encoder_type='gnn')
    trainer = Trainer(model, config)
    trainer.train(data.train, epochs=1)
    
    # 3. Evaluate
    benchmark = RetrosynthesisBenchmark(data.test)
    results = benchmark.evaluate_model(model)
    
    assert results['top_1_acc'] > 0.0
    assert results['rc_acc'] > 0.0

def test_gnn_vs_mamba_parity():
    """Ensure both encoders produce valid outputs"""
    data = load_uspto_sample()
    
    gnn_model = HierRetro(encoder_type='gnn')
    mamba_model = HierRetro(encoder_type='mamba')
    
    batch = next(iter(data.test))
    
    gnn_out = gnn_model(batch.product_graph)
    mamba_out = mamba_model(batch.product_graph)
    
    # Both should have same output structure
    assert gnn_out.keys() == mamba_out.keys()
    
    # Shapes should match (accounting for hidden dim differences)
    assert gnn_out['rc_type'].shape == mamba_out['rc_type'].shape
```

---

## Directory Structure

```
hierretro_mamba/
├── data/
│   ├── preprocess_uspto.py
│   ├── augmentation.py
│   └── dataset.py
├── models/
│   ├── encoders/
│   │   ├── base_encoder.py
│   │   ├── gnn_encoder.py
│   │   ├── mamba_encoder.py
│   │   └── graph_linearizer.py
│   ├── prediction_heads/
│   │   ├── rc_type_predictor.py
│   │   ├── atom_center_predictor.py
│   │   ├── bond_center_predictor.py
│   │   ├── action_predictor.py
│   │   └── termination_predictor.py
│   └── hierretro.py
├── training/
│   ├── train.py
│   ├── damt_loss.py
│   └── pretrain_contrastive.py
├── inference/
│   ├── beam_search.py
│   ├── explainability.py
│   ├── decision_path.py
│   └── cpu_optimize.py
├── evaluation/
│   ├── metrics.py
│   ├── benchmark.py
│   └── ablation.py
├── api/
│   └── inference_server.py
├── utils/
│   ├── model_io.py
│   ├── rdkit_utils.py
│   └── graph_utils.py
├── tests/
│   ├── test_models.py
│   └── test_pipeline.py
├── configs/
│   ├── gnn_config.yaml
│   └── mamba_config.yaml
├── scripts/
│   ├── train_gnn.sh
│   ├── train_mamba.sh
│   └── compare_models.sh
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_training.ipynb
    └── 03_results_analysis.ipynb
```

---

## Execution Timeline (8-10 weeks)

**Week 1-2: Setup & Data**
- Environment configuration
- USPTO-50K preprocessing
- Action vocabulary extraction

**Week 3-4: Model Development**
- Implement GNN encoder (Uni-Mol+)
- Implement Mamba encoder
- Implement prediction heads

**Week 5-6: Training**
- DAMT loss implementation
- Train GNN variant
- Train Mamba variant
- Contrastive pre-training (optional)

**Week 7: Inference & Explainability**
- Beam search implementation
- Attention visualization
- Decision path explanations

**Week 8: Evaluation**
- Benchmark suite
- GNN vs Mamba comparison
- Ablation studies

**Week 9-10: Optimization & Deployment**
- CPU optimization (quantization, TorchScript)
- API development
- Documentation & testing

---

## Key Implementation Notes

1. **Mamba Integration**: Use `mamba-ssm` library with causal convolutions for efficient SSM blocks. The graph linearization strategy (DFS vs BFS) will significantly impact performance—experiment with both.

2. **Memory Management**: For CPU inference, implement gradient checkpointing and batch size tuning. Mamba's linear complexity should give ~2x speedup over GNN Transformers on long sequences (>50 atoms).

3. **Explainability**: The hierarchical structure naturally provides explainability—each decision (RC type → RC location → action → termination) can be logged and visualized.

4. **Multi-Step Handling**: Use a replay buffer to store previous actions. Update atom/bond features with history embeddings before each step.

5. **Action Space**: Carefully design the action vocabulary from USPTO-50K. HierRetro reports 99.9% coverage—verify this and handle edge cases with an "unknown action" class.

6. **CPU Optimization**: Leverage Intel MKL-DNN, dynamic quantization (qint8), and TorchScript compilation. Profile with `torch.profiler` to identify bottlenecks.

This plan provides a complete roadmap from data preprocessing to deployment, balancing HierRetro's hierarchical reasoning with Mamba's efficient inference for practical retrosynthesis applications.