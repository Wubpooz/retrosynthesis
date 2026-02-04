"""
Dynamic Adaptive Multi-Task Learning (DAMT) Loss.
Dynamically adjusts task weights based on descent rates.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List


class DAMTLoss(nn.Module):
    """
    Dynamic Adaptive Multi-Task Learning loss function.
    
    Adjusts weights based on relative descent rates of each task:
    - Fast-descending tasks get lower weight
    - Slow-descending tasks get higher weight
    
    This prevents any single task from dominating training.
    """
    
    def __init__(self, num_tasks: int = 4, queue_size: int = 50, tau: float = 1.0):
        """
        Args:
            num_tasks: Number of tasks
            queue_size: Window size for computing descent rates
            tau: Temperature for softmax weighting
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        self.queue_size = queue_size
        self.tau = tau
        
        # Loss history queues
        self.loss_queues = {i: deque(maxlen=queue_size) for i in range(num_tasks)}
        
        # Initial uniform weights
        self.register_buffer('task_weights', torch.ones(num_tasks) / num_tasks)
        
    def compute_descent_rates(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute descent rate for each task.
        
        Descent rate = (old_loss - new_loss) / old_loss
        Positive rate = loss decreasing (good)
        Negative rate = loss increasing (bad)
        
        Args:
            current_losses: [num_tasks] tensor of current loss values
            
        Returns:
            [num_tasks] descent rates
        """
        descent_rates = torch.zeros(self.num_tasks, device=current_losses.device)
        
        for i in range(self.num_tasks):
            queue = self.loss_queues[i]
            
            if len(queue) == 0:
                # No history yet
                descent_rates[i] = 0.0
            else:
                # Compute average descent from history
                old_loss = sum(queue) / len(queue)
                new_loss = current_losses[i].item()
                
                if old_loss > 0:
                    descent_rates[i] = (old_loss - new_loss) / old_loss
                else:
                    descent_rates[i] = 0.0
            
            # Update queue
            queue.append(current_losses[i].item())
        
        return descent_rates
    
    def compute_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive task weights based on descent rates.
        
        Tasks with slower descent (harder tasks) get higher weight.
        
        Args:
            current_losses: [num_tasks] tensor
            
        Returns:
            [num_tasks] normalized weights
        """
        # Compute descent rates
        descent_rates = self.compute_descent_rates(current_losses)
        
        # Invert: slower descent = higher weight
        # Add small epsilon to avoid division by zero
        inverse_rates = 1.0 / (descent_rates + 1e-8)
        
        # Apply softmax with temperature
        weights = torch.softmax(inverse_rates / self.tau, dim=0)
        
        # Update stored weights
        self.task_weights = weights
        
        return weights
    
    def forward(self, losses_dict: Dict[str, torch.Tensor], 
                update_weights: bool = True) -> tuple:
        """
        Compute weighted multi-task loss.
        
        Args:
            losses_dict: Dict mapping task names to loss tensors
            update_weights: Whether to update weights based on current losses
            
        Returns:
            total_loss: Weighted sum of losses
            task_weights: Dict of weights used
            individual_losses: Dict of individual loss values
        """
        # Expected task order
        task_names = ['rc_type', 'rc_localization', 'action', 'termination']
        
        # Stack losses
        losses = []
        for name in task_names:
            if name in losses_dict:
                losses.append(losses_dict[name])
            else:
                # Use zero if task not present
                losses.append(torch.tensor(0.0, device=list(losses_dict.values())[0].device))
        
        losses = torch.stack(losses)
        
        # Compute or use existing weights
        if update_weights and len(self.loss_queues[0]) > 0:
            weights = self.compute_weights(losses.detach())
        else:
            weights = self.task_weights
        
        # Weighted sum
        total_loss = (weights * losses).sum()
        
        # Return detailed info
        task_weights_dict = {name: weights[i].item() for i, name in enumerate(task_names)}
        individual_losses = {name: losses[i].item() for i, name in enumerate(task_names)}
        
        return total_loss, task_weights_dict, individual_losses
    
    def reset(self):
        """Reset loss history and weights."""
        for queue in self.loss_queues.values():
            queue.clear()
        self.task_weights = torch.ones(self.num_tasks) / self.num_tasks


if __name__ == '__main__':
    # Test DAMT loss
    damt = DAMTLoss(num_tasks=4, queue_size=10, tau=1.0)
    
    # Simulate training
    print("Simulating multi-task learning:")
    for epoch in range(20):
        # Simulate different loss trajectories
        losses = {
            'rc_type': torch.tensor(1.0 / (epoch + 1)),  # Fast descent
            'rc_localization': torch.tensor(2.0 / (epoch * 0.5 + 1)),  # Medium descent
            'action': torch.tensor(3.0 / (epoch * 0.2 + 1)),  # Slow descent
            'termination': torch.tensor(0.5 / (epoch + 1))  # Fast descent
        }
        
        total_loss, weights, indiv_losses = damt(losses, update_weights=True)
        
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Losses: {indiv_losses}")
            print(f"  Weights: {weights}")
            print(f"  Total: {total_loss.item():.4f}")
