"""
Graph linearization strategies for converting molecular graphs to sequences.
Used by Mamba encoder.
"""

import torch
from typing import List, Tuple, Dict
from collections import deque


class GraphLinearizer:
    """
    Converts molecular graphs to sequences for processing by sequential models.
    Supports DFS and BFS traversal strategies.
    """
    
    def __init__(self, strategy: str = 'dfs', add_backtrack_tokens: bool = True):
        """
        Args:
            strategy: 'dfs' or 'bfs'
            add_backtrack_tokens: Add special tokens when backtracking in DFS
        """
        assert strategy in ['dfs', 'bfs'], f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.add_backtrack_tokens = add_backtrack_tokens
    
    def dfs_linearize(self, edge_index: torch.Tensor, num_atoms: int, 
                     start_node: int = 0) -> Tuple[List[int], List[int]]:
        """
        Depth-first traversal of molecular graph.
        
        Args:
            edge_index: [2, num_edges] edge connectivity
            num_atoms: number of atoms in molecule
            start_node: starting atom index
            
        Returns:
            sequence: List of atom indices in traversal order
            positions: Original positions in graph
        """
        # Build adjacency list
        adj = {i: [] for i in range(num_atoms)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src].append(dst)
        
        visited = set()
        sequence = []
        positions = []
        
        def dfs(node: int, parent: int = -1):
            visited.add(node)
            sequence.append(node)
            positions.append(node)
            
            # Visit neighbors
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor, node)
                    
                    # Add backtrack token if enabled
                    if self.add_backtrack_tokens:
                        sequence.append(-1)  # Special backtrack token
                        positions.append(node)  # Return to parent
        
        dfs(start_node)
        
        return sequence, positions
    
    def bfs_linearize(self, edge_index: torch.Tensor, num_atoms: int,
                     start_node: int = 0) -> Tuple[List[int], List[int]]:
        """
        Breadth-first traversal of molecular graph.
        
        Args:
            edge_index: [2, num_edges] edge connectivity
            num_atoms: number of atoms in molecule
            start_node: starting atom index
            
        Returns:
            sequence: List of atom indices in traversal order
            positions: Original positions in graph
        """
        # Build adjacency list
        adj = {i: [] for i in range(num_atoms)}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj[src].append(dst)
        
        visited = set([start_node])
        queue = deque([start_node])
        sequence = []
        positions = []
        
        while queue:
            node = queue.popleft()
            sequence.append(node)
            positions.append(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return sequence, positions
    
    def linearize(self, edge_index: torch.Tensor, num_atoms: int) -> Tuple[List[int], List[int]]:
        """
        Linearize graph using configured strategy.
        
        Returns:
            sequence: Traversal order
            positions: Original graph positions
        """
        if self.strategy == 'dfs':
            return self.dfs_linearize(edge_index, num_atoms)
        else:
            return self.bfs_linearize(edge_index, num_atoms)
    
    def linearize_batch(self, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linearize a batch of graphs.
        
        Args:
            edge_index: [2, num_edges] edge connectivity
            batch: [num_atoms] batch assignment
            
        Returns:
            sequence_indices: Flat tensor of atom indices in traversal order
            graph_sizes: Number of sequence elements per graph
        """
        # Separate by batch
        unique_batches = torch.unique(batch)
        num_atoms_per_graph = [(batch == b).sum().item() for b in unique_batches]
        
        all_sequences = []
        all_positions = []
        
        offset = 0
        for b_idx, num_atoms in enumerate(num_atoms_per_graph):
            # Get edges for this graph
            mask = (batch[edge_index[0]] == b_idx) & (batch[edge_index[1]] == b_idx)
            graph_edges = edge_index[:, mask] - offset
            
            # Linearize
            sequence, positions = self.linearize(graph_edges, num_atoms)
            
            # Add offset back
            all_sequences.extend([s + offset if s >= 0 else s for s in sequence])
            all_positions.extend([p + offset for p in positions])
            
            offset += num_atoms
        
        return torch.tensor(all_sequences), torch.tensor(all_positions)


if __name__ == '__main__':
    # Test linearization
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    print("DFS Linearization:")
    linearizer = GraphLinearizer(strategy='dfs', add_backtrack_tokens=True)
    seq, pos = linearizer.linearize(edge_index, num_atoms=4)
    print(f"  Sequence: {seq}")
    print(f"  Positions: {pos}")
    
    print("\nBFS Linearization:")
    linearizer = GraphLinearizer(strategy='bfs')
    seq, pos = linearizer.linearize(edge_index, num_atoms=4)
    print(f"  Sequence: {seq}")
    print(f"  Positions: {pos}")
