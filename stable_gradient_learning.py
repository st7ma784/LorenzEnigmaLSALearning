#!/usr/bin/env python3
"""
Improved Stable Gradient-Based Permutation Matrix Learning
Addresses instability issues with better regularization and adaptive techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from enigma_lorenz_analysis import EnigmaMachine, LorenzCipher, PermutationMatrixAnalyzer
from scipy.optimize import linear_sum_assignment
import math

class StableDifferentiablePermutation(nn.Module):
    """Improved differentiable permutation with multiple stability techniques"""
    
    def __init__(self, size: int = 26, temperature: float = 1.0, 
                 noise_std: float = 0.01, entropy_reg: float = 0.01):
        super().__init__()
        self.size = size
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.noise_std = noise_std
        self.entropy_reg = entropy_reg
        
        # Initialize with small random values around identity-like structure
        init_logits = torch.randn(size, size) * 0.1
        # Add slight bias toward identity to encourage stability
        init_logits += torch.eye(size) * 0.5
        self.logits = nn.Parameter(init_logits)
        
        # Moving averages for stability
        self.register_buffer('ema_logits', torch.zeros_like(init_logits))
        self.ema_decay = 0.999
        
    def forward(self, hard: bool = False, use_ema: bool = False):
        """Forward pass with stability improvements"""
        # Use EMA logits for more stable gradients
        current_logits = self.ema_logits if use_ema else self.logits
        
        # Temperature annealing - start hot, cool down
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # Add noise for regularization during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(current_logits) * self.noise_std
            scaled_logits = (current_logits + noise) / temp
        else:
            scaled_logits = current_logits / temp
            
        # Improved Sinkhorn normalization with momentum
        soft_perm = self.stable_sinkhorn_normalize(
            torch.softmax(scaled_logits, dim=-1)
        )
        
        if hard:
            hard_perm = self.hungarian_assignment(soft_perm)
            # Improved straight-through estimator with gradient scaling
            alpha = 0.1  # Gradient flow control
            return hard_perm + alpha * (soft_perm - soft_perm.detach())
        
        return soft_perm
    
    def stable_sinkhorn_normalize(self, matrix: torch.Tensor, 
                                num_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
        """Stable Sinkhorn normalization with convergence monitoring"""
        prev_matrix = matrix.clone()
        
        for i in range(num_iters):
            # Row normalization with numerical stability
            row_sums = torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / torch.clamp(row_sums, min=eps)
            
            # Column normalization with numerical stability  
            col_sums = torch.sum(matrix, dim=0, keepdim=True)
            matrix = matrix / torch.clamp(col_sums, min=eps)
            
            # Early stopping if converged
            if i > 5 and torch.allclose(matrix, prev_matrix, atol=1e-4):
                break
            prev_matrix = matrix.clone()
        
        return matrix
    
    def hungarian_assignment(self, soft_matrix: torch.Tensor) -> torch.Tensor:
        """Convert soft assignment to hard permutation"""
        with torch.no_grad():
            cost_matrix = -soft_matrix.detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            
            hard_matrix = torch.zeros_like(soft_matrix)
            hard_matrix[row_idx, col_idx] = 1.0
            
        return hard_matrix
    
    def update_ema(self):
        """Update exponential moving average"""
        with torch.no_grad():
            self.ema_logits.mul_(self.ema_decay).add_(
                self.logits, alpha=1 - self.ema_decay
            )
    
    def get_regularization_loss(self, soft_matrix: torch.Tensor) -> torch.Tensor:
        """Comprehensive regularization loss"""
        losses = {}
        
        # Doubly stochastic constraints
        row_sums = torch.sum(soft_matrix, dim=1)
        col_sums = torch.sum(soft_matrix, dim=0)
        losses['doubly_stochastic'] = (
            torch.mean((row_sums - 1.0) ** 2) + 
            torch.mean((col_sums - 1.0) ** 2)
        )
        
        # Entropy regularization (encourage exploration)
        entropy = -torch.sum(soft_matrix * torch.log(soft_matrix + 1e-8))
        losses['entropy'] = -self.entropy_reg * entropy  # Negative for maximization
        
        # Orthogonality constraint (soft version of permutation property)
        should_be_identity = torch.matmul(soft_matrix, soft_matrix.t())
        identity = torch.eye(self.size, device=soft_matrix.device)
        losses['orthogonality'] = torch.norm(should_be_identity - identity, p='fro') ** 2
        
        # Temperature regularization (prevent extreme temperatures)
        losses['temperature'] = 0.01 * (self.temperature - 1.0) ** 2
        
        return losses

class StableEnigmaRotorNetwork(nn.Module):
    """Stable rotor network with improved training dynamics"""
    
    def __init__(self, alphabet_size: int = 26):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.permutation = StableDifferentiablePermutation(alphabet_size)
        
        # Position as continuous parameter with periodic activation
        self.position_raw = nn.Parameter(torch.zeros(1))
        
        # Learnable position embedding
        self.position_embedding = nn.Embedding(alphabet_size, alphabet_size)
        
    def get_position(self):
        """Get position with periodic boundary conditions"""
        return torch.fmod(self.position_raw, self.alphabet_size)
    
    def forward(self, input_chars: torch.Tensor, hard_assignment: bool = False, use_ema: bool = False):
        """Forward pass with position embedding"""
        perm_matrix = self.permutation(hard=hard_assignment, use_ema=use_ema)
        
        # Continuous position handling
        position = self.get_position()
        pos_int = torch.round(position) % self.alphabet_size
        
        # Learnable position transformation
        pos_embed = self.position_embedding(pos_int.long())
        
        # Apply transformation
        output = torch.matmul(input_chars, perm_matrix)
        
        # Apply position effect (simplified for stability)
        position_effect = torch.softmax(pos_embed, dim=-1)
        output = output * position_effect.unsqueeze(0)
        
        return output
    
    def get_regularization_loss(self):
        """Get regularization losses from all components"""
        soft_matrix = self.permutation(hard=False, use_ema=False)
        reg_losses = self.permutation.get_regularization_loss(soft_matrix)
        
        # Add position regularization
        reg_losses['position_smoothness'] = 0.01 * torch.sum(self.position_raw ** 2)
        
        return reg_losses

class StablePermutationLearner:
    """Improved learner with adaptive techniques and stability measures"""
    
    def __init__(self, device: str = 'cpu', config: Optional[Dict] = None):
        self.device = device
        self.config = config or self.get_default_config()
        
        # Initialize networks
        self.enigma_model = self._create_enigma_model()
        self.mask_predictor = self._create_mask_predictor()
        
        # Training state
        self.training_history = {
            'mask_losses': [], 'rotor_losses': [], 
            'reg_losses': [], 'temperatures': [],
            'gradient_norms': [], 'learning_rates': []
        }
        
    def get_default_config(self):
        """Get default stable configuration"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'temperature_decay': 0.995,
            'min_temperature': 0.1,
            'regularization_weights': {
                'doubly_stochastic': 1.0,
                'entropy': 0.01,
                'orthogonality': 0.1,
                'temperature': 0.01,
                'position_smoothness': 0.01
            },
            'adaptive_lr': True,
            'lr_schedule': 'cosine',
            'warmup_epochs': 10,
            'patience': 20,
            'min_lr': 1e-6
        }
    
    def _create_enigma_model(self):
        """Create stable Enigma model"""
        model = nn.ModuleList([
            StableEnigmaRotorNetwork() for _ in range(3)
        ])
        return model.to(self.device)
    
    def _create_mask_predictor(self):
        """Create stable mask predictor with better architecture"""
        return nn.Sequential(
            nn.Linear(26*26*3 + 3, 512),
            nn.LayerNorm(512),  # Add normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def train_with_stability(self, training_data: List[Dict], epochs: int = 100):
        """Train with comprehensive stability techniques"""
        # Setup optimizers with different learning rates
        enigma_optimizer = optim.AdamW(
            self.enigma_model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        mask_optimizer = optim.AdamW(
            self.mask_predictor.parameters(),
            lr=self.config['learning_rate'] * 2,  # Faster for easier task
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate schedulers
        enigma_scheduler = self._create_scheduler(enigma_optimizer, epochs)
        mask_scheduler = self._create_scheduler(mask_optimizer, epochs)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_losses = self._train_epoch(
                training_data, enigma_optimizer, mask_optimizer, epoch
            )
            
            # Update learning rates
            if enigma_scheduler:
                enigma_scheduler.step()
            if mask_scheduler:
                mask_scheduler.step()
            
            # Update EMA
            for rotor in self.enigma_model:
                rotor.permutation.update_ema()
            
            # Anneal temperatures
            self._anneal_temperatures(epoch)
            
            # Log progress
            self._log_epoch(epoch, epoch_losses, enigma_optimizer, mask_optimizer)
            
            # Early stopping
            current_loss = sum(epoch_losses.values())
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return self.training_history
    
    def _train_epoch(self, training_data, enigma_optimizer, mask_optimizer, epoch):
        """Train single epoch with stability measures"""
        epoch_losses = {'mask': 0, 'rotor': 0, 'regularization': 0}
        
        # Shuffle training data
        np.random.shuffle(training_data)
        
        for batch_idx, data in enumerate(training_data):
            # Mask predictor training (easier task first)
            mask_loss = self._train_mask_predictor(data, mask_optimizer)
            epoch_losses['mask'] += mask_loss
            
            # Rotor learning (harder task)
            rotor_loss, reg_loss = self._train_rotor_learner(data, enigma_optimizer, epoch)
            epoch_losses['rotor'] += rotor_loss
            epoch_losses['regularization'] += reg_loss
            
            # Log batch progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Mask={mask_loss:.4f}, Rotor={rotor_loss:.4f}, Reg={reg_loss:.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(training_data)
        
        return epoch_losses
    
    def _train_mask_predictor(self, data, optimizer):
        """Train mask predictor with stability"""
        rotor_matrices = data['rotor_matrices'].to(self.device)
        rotor_positions = data['rotor_positions'].to(self.device)
        target_mask = data['mask'].to(self.device)
        
        # Forward pass
        flattened = rotor_matrices.flatten()
        features = torch.cat([flattened, rotor_positions])
        
        # Predict mask bits sequentially
        total_loss = 0
        for i in range(len(target_mask)):
            pred = self.mask_predictor(features)
            loss = nn.BCELoss()(pred.squeeze(), target_mask[i:i+1].float())
            total_loss += loss
        
        avg_loss = total_loss / len(target_mask)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.mask_predictor.parameters(), 
            self.config['gradient_clip']
        )
        optimizer.step()
        
        return avg_loss.item()
    
    def _train_rotor_learner(self, data, optimizer, epoch):
        """Train rotor learner with comprehensive regularization"""
        target_mask = data['mask'].to(self.device)
        
        # Create dummy input (we're learning rotors to match mask)
        dummy_input = torch.eye(26).to(self.device)
        
        # Forward pass through learned rotors
        current = dummy_input
        total_reg_loss = 0
        
        # Apply each rotor
        for rotor in self.enigma_model:
            current = rotor(current, hard_assignment=False, use_ema=(epoch > 10))
            # Accumulate regularization losses
            reg_losses = rotor.get_regularization_loss()
            for key, loss in reg_losses.items():
                weight = self.config['regularization_weights'].get(key, 0.1)
                total_reg_loss += weight * loss
        
        # Convert output to binary mask (simplified)
        predicted_mask = (current.sum(dim=1) > 13).float()
        
        # Mask reconstruction loss
        mask_loss = nn.MSELoss()(
            predicted_mask[:len(target_mask)], 
            target_mask.float()
        )
        
        # Total loss
        total_loss = mask_loss + total_reg_loss
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping and monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.enigma_model.parameters(),
            self.config['gradient_clip']
        )
        
        optimizer.step()
        
        return mask_loss.item(), total_reg_loss.item()
    
    def _create_scheduler(self, optimizer, epochs):
        """Create learning rate scheduler"""
        if not self.config['adaptive_lr']:
            return None
            
        if self.config['lr_schedule'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=self.config['min_lr']
            )
        elif self.config['lr_schedule'] == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=self.config['min_lr']
            )
        
        return None
    
    def _anneal_temperatures(self, epoch):
        """Anneal temperatures for stability"""
        for rotor in self.enigma_model:
            with torch.no_grad():
                current_temp = rotor.permutation.temperature.data
                new_temp = max(
                    current_temp * self.config['temperature_decay'],
                    self.config['min_temperature']
                )
                rotor.permutation.temperature.data.fill_(new_temp)
    
    def _log_epoch(self, epoch, losses, enigma_opt, mask_opt):
        """Log epoch progress"""
        # Get current learning rates
        enigma_lr = enigma_opt.param_groups[0]['lr']
        mask_lr = mask_opt.param_groups[0]['lr']
        
        # Get current temperatures
        temps = [rotor.permutation.temperature.item() for rotor in self.enigma_model]
        avg_temp = np.mean(temps)
        
        # Store history
        self.training_history['mask_losses'].append(losses['mask'])
        self.training_history['rotor_losses'].append(losses['rotor'])
        self.training_history['reg_losses'].append(losses['regularization'])
        self.training_history['temperatures'].append(avg_temp)
        self.training_history['learning_rates'].append((enigma_lr, mask_lr))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  f"Mask={losses['mask']:.4f}, "
                  f"Rotor={losses['rotor']:.4f}, "
                  f"Reg={losses['regularization']:.4f}, "
                  f"Temp={avg_temp:.3f}, "
                  f"LR=({enigma_lr:.6f}, {mask_lr:.6f})")

def run_stable_experiment():
    """Run the stable gradient learning experiment"""
    print("Starting Stable Gradient-Based Learning Experiment...")
    
    # Create learner with stability config
    config = {
        'learning_rate': 0.0005,  # Lower learning rate
        'gradient_clip': 0.5,     # Aggressive clipping
        'temperature_decay': 0.99,
        'regularization_weights': {
            'doubly_stochastic': 2.0,  # Strong constraint
            'entropy': 0.05,
            'orthogonality': 0.5,
            'temperature': 0.02
        }
    }
    
    learner = StablePermutationLearner(config=config)
    
    # Generate smaller, more controlled training data
    training_data = []
    for i in range(20):  # Smaller dataset for stability
        # Simple, consistent rotor configs
        rotor_configs = [
            ('EKMFLGDQVZNTOWYHXUSPAIBRCJ', i % 26),
            ('AJDKSIRUXBLHWTMCQGZNPYFVOE', (i * 2) % 26),
            ('BDFHJLCPRTXVZNYEIWGAKMUSQO', (i * 3) % 26)
        ]
        
        enigma = EnigmaMachine(rotor_configs)
        plaintext = "ATTACKATDAWN" * 3  # Consistent plaintext
        ciphertext = enigma.encode_message(plaintext)
        
        lorenz = LorenzCipher()
        mask = lorenz.extract_mask(plaintext, ciphertext)
        
        training_data.append({
            'rotor_matrices': torch.stack([
                torch.tensor(rotor.permutation_matrix, dtype=torch.float32)
                for rotor in enigma.rotors
            ]),
            'rotor_positions': torch.tensor([
                rotor.position for rotor in enigma.rotors
            ], dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32)
        })
    
    # Train with stability measures
    history = learner.train_with_stability(training_data, epochs=50)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['mask_losses'], label='Mask Loss')
    plt.plot(history['rotor_losses'], label='Rotor Loss')
    plt.plot(history['reg_losses'], label='Regularization Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.yscale('log')
    
    plt.subplot(2, 3, 2)
    plt.plot(history['temperatures'])
    plt.title('Temperature Annealing')
    plt.ylabel('Temperature')
    
    plt.subplot(2, 3, 3)
    enigma_lrs = [lr[0] for lr in history['learning_rates']]
    mask_lrs = [lr[1] for lr in history['learning_rates']]
    plt.plot(enigma_lrs, label='Enigma LR')
    plt.plot(mask_lrs, label='Mask LR')
    plt.legend()
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/user/Documents/enigmalorenz/CascadeProjects/windsurf-project/stable_training_results.png')
    plt.show()
    
    return learner, history

if __name__ == "__main__":
    learner, history = run_stable_experiment()