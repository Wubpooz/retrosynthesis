# Retrosynthesis

Implement hierarchical retrosynthesis framework combining:
- HierRetro (https://arxiv.org/pdf/2411.19503.pdf):
   - RC type classifier (atom vs bond)
   - RC location predictor
   - Action predictor
   - GNN encoder (Uni-Mol+)
- Mamba comparison (https://www.nature.com/articles/s44387-025-00009-7):
   - State-space model encoder
   - Linear-complexity attention
   - CPU-friendly inference

Stack: PyTorch, PyG, RDKit, mamba-ssm
Dataset: USPTO-50K
Goal: Explainable predictions + fast CPU inference