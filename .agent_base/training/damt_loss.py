import torch
import torch.nn as nn

class DAMTLoss(nn.Module):
    """Dynamic loss weighting based on descent rates"""
    def __init__(self, num_tasks):
        super(DAMTLoss, self).__init__()
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def forward(self, losses):
        """
        Compute dynamically weighted loss.
        Args:
            losses: List of individual task losses.
        Returns:
            Weighted sum of losses.
        """
        weighted_losses = [w * l for w, l in zip(self.task_weights, losses)]
        return sum(weighted_losses)