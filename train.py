import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse

# Import our custom modules (assuming they're in the same directory)
from unet_model import ConditionalUNet, ColorMapper
from dataset_loader import create_dataloaders

class PolygonColorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.color_mapper = ColorMapper()
        
        # Initialize model
        self.model = ConditionalUNet(
            n_channels=3,
            n_classes=3,
            num_colors=len(self.color_mapper.color_to_id),
            color_embed_dim=config['color_embed_dim'],
            bilinear=config['bilinear']
        ).to(self.device)
        
        # Loss function - combination of MSE and perceptual loss
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def combined_loss(self, output, target, alpha=0.7):
        """Combine MSE and L1 loss for better training stability"""
        mse_loss = self.criterion_mse(output, target)
        l1_loss = self.criterion_l1(output, target)
        return alpha * mse_loss + (1 - alpha) * l1_loss
    
    def calculate_metrics(self, output, target):
        """Calculate various metrics for evaluation"""
        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # MSE
        mse = np.mean((output_np - target_np) ** 2)
        
        # PSNR
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # SSIM would require additional implementation
        
        return {'mse': mse, 'psnr': psnr}
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {'mse': 0, 'psnr': 0}
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            color_ids = batch['color_id'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, color_ids)
            
            # Calculate loss
            loss = self.combined_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_metrics = self.calculate_metrics(outputs, targets)
            epoch_metrics['mse'] += batch_metrics['mse']
            epoch_metrics['psnr'] += batch_metrics['psnr']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
            
            # Log to wandb every 50 batches
            if batch_idx % 50 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_mse': batch_metrics['mse'],
                    'batch_psnr': batch_metrics['psnr'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Average metrics for the epoch
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = {'mse': 0, 'psnr': 0}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                color_ids = batch['color_id'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, color_ids)
                
                # Calculate loss
                loss = self.combined_loss(outputs, targets)
                
                # Update metrics
                epoch_loss += loss.item()
                batch_metrics = self.calculate_metrics(outputs, targets)
                epoch_metrics['mse'] += batch_metrics['mse']
                epoch_metrics['psnr'] += batch_metrics['psnr']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })
        
        # Average metrics for the epoch
        num_batches = len(val_loader)
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, avg_metrics
    
    def save_sample_predictions(self, val_loader, epoch, num_samples=4):
        """Save sample predictions for visual inspection"""
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(val_loader))
            inputs = batch['input'][:num_samples].to(self.device)
            targets = batch['target'][:num_samples].to(self.device)
            color_ids = batch['color_id'][:num_samples].to(self.device)
            color_names = batch['color_name'][:num_samples]
            
            outputs = self.model(inputs, color_ids)
            
            # Convert to numpy for visualization
            inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)
            outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Create comparison plot
            fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*3, 9))
            
            for i in range(num_samples):
                # Input
                axes[0, i].imshow(np.clip(inputs_np[i], 0, 1))
                axes[0, i].set_title(f'Input')
                axes[0, i].axis('off')
                
                # Target
                axes[1, i].imshow(np.clip(targets_np[i], 0, 1))
                axes[1, i].set_title(f'Target ({color_names[i]})')
                axes[1, i].axis('off')
                
                # Prediction
                axes[2, i].imshow(np.clip(outputs_np[i], 0, 1))
                axes[2, i].set_title(f'Prediction')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f"predictions_epoch_{epoch}": wandb.Image(fig)})
            plt.close()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(wandb.run.dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(wandb.run.dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mse': train_metrics['mse'],
                'val_mse': val_metrics['mse'],
                'train_psnr': train_metrics['psnr'],
                'val_psnr': val_metrics['psnr'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Save sample predictions every 5 epochs
            if epoch % 5 == 0:
                self.save_sample_predictions(val_loader, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
