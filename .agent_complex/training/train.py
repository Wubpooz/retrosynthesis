"""
Main training script for HierRetro.
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models import HierRetro
from data.dataset import RetrosynthesisDataset, collate_fn
from training.damt_loss import DAMTLoss


class Trainer:
    """Training manager for HierRetro."""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to YAML config file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        encoder_type = self.config['model']['encoder_type']
        self.model = HierRetro(encoder_type=encoder_type, config=self.config['model'])
        self.model.to(self.device)
        
        print(f"Model initialized with {encoder_type} encoder")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize datasets
        data_config = self.config['data']
        self.train_dataset = RetrosynthesisDataset(
            data_path=data_config.get('train_path', 'data/processed/uspto_50k_processed.pkl'),
            split='train',
            vocab_path=data_config.get('vocab_path', 'data/processed/action_vocab.json')
        )
        
        self.val_dataset = RetrosynthesisDataset(
            data_path=data_config.get('val_path', 'data/processed/uspto_50k_processed.pkl'),
            split='val',
            vocab_path=data_config.get('vocab_path', 'data/processed/action_vocab.json')
        )
        
        # Data loaders
        train_config = self.config['training']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config['max_epochs']
        )
        
        # DAMT loss
        damt_config = train_config.get('damt_loss', {})
        self.criterion = DAMTLoss(
            num_tasks=4,
            queue_size=damt_config.get('queue_size', 50),
            tau=damt_config.get('tau', 1.0)
        )
        
        # Logging
        self.log_config = self.config['logging']
        if self.log_config.get('use_wandb', False):
            wandb.init(
                project=self.log_config.get('project_name', 'hierretro'),
                config=self.config
            )
        
        # Checkpointing
        self.checkpoint_dir = Path(self.log_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = train_config.get('early_stopping', {}).get('patience', 10)
    
    def compute_losses(self, batch, outputs):
        """Compute individual task losses."""
        losses = {}
        
        # RC type loss
        rc_type_loss = F.cross_entropy(outputs['rc_type_logits'], batch['rc_type'])
        losses['rc_type'] = rc_type_loss
        
        # RC localization loss (simplified - average over batch)
        rc_loc_losses = []
        for i, (atom_logits, atom_targets) in enumerate(zip(
            outputs['atom_center_logits'], batch['rc_atoms']
        )):
            if len(atom_targets) > 0:
                # Multi-label classification
                targets = torch.zeros(len(atom_logits), device=atom_logits.device)
                targets[atom_targets] = 1.0
                rc_loc_losses.append(F.binary_cross_entropy_with_logits(
                    atom_logits, targets
                ))
        
        if rc_loc_losses:
            losses['rc_localization'] = torch.stack(rc_loc_losses).mean()
        else:
            losses['rc_localization'] = torch.tensor(0.0, device=self.device)
        
        # Action loss (placeholder - simplified)
        # In full implementation, compute based on predicted actions
        losses['action'] = torch.tensor(0.0, device=self.device)
        
        # Termination loss
        term_loss = F.cross_entropy(outputs['termination_logits'], batch['terminate'])
        losses['termination'] = term_loss
        
        return losses
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            graph = batch['graph'].to(self.device)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(graph)
            
            # Compute losses
            task_losses = self.compute_losses(batch, outputs)
            
            # DAMT weighted loss
            loss, task_weights, indiv_losses = self.criterion(task_losses)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training'].get('gradient_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rc_type': f"{indiv_losses['rc_type']:.4f}"
            })
            
            if batch_idx % self.log_config.get('log_every', 10) == 0:
                if self.log_config.get('use_wandb', False):
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/rc_type_loss': indiv_losses['rc_type'],
                        'train/rc_loc_loss': indiv_losses['rc_localization'],
                        'train/term_loss': indiv_losses['termination'],
                        **{f'train/weight_{k}': v for k, v in task_weights.items()}
                    })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            graph = batch['graph'].to(self.device)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            outputs = self.model(graph)
            task_losses = self.compute_losses(batch, outputs)
            loss, _, _ = self.criterion(task_losses, update_weights=False)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        if self.log_config.get('use_wandb', False):
            wandb.log({'val/loss': avg_loss, 'epoch': epoch})
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        max_epochs = self.config['training']['max_epochs']
        
        for epoch in range(max_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(epoch)
            print(f"Val loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.log_config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs")
                
                if self.patience_counter >= self.max_patience:
                    print("Early stopping triggered!")
                    break
        
        print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()
