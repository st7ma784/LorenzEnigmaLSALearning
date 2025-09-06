#!/usr/bin/env python3
"""
Stable Training Demonstration
Shows stability techniques for permutation matrix learning without heavy dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
import random
import math

class StabilityConfig:
    """Configuration for stability parameters"""
    def __init__(self):
        self.learning_rate = 0.0005  # Lower for stability
        self.temperature = 1.0       # Sinkhorn temperature
        self.gradient_clip = 0.5     # Aggressive clipping
        self.reg_strength = 1.5      # Strong regularization
        self.noise_std = 0.01        # Small noise
        self.ema_decay = 0.99        # EMA smoothing
        self.min_temperature = 0.1   # Temperature floor
        self.temp_decay = 0.995      # Temperature annealing

class StablePermutationLearner:
    """Simplified stable permutation learner"""
    
    def __init__(self, size=26, config=None):
        self.size = size
        self.config = config or StabilityConfig()
        
        # Initialize parameters
        self.logits = np.random.randn(size, size) * 0.1
        self.logits += np.eye(size) * 0.5  # Bias toward identity
        
        # EMA buffers
        self.ema_logits = np.zeros_like(self.logits)
        
        # Training history
        self.history = {
            'losses': [], 'temperatures': [], 'gradient_norms': [],
            'permutation_errors': [], 'stability_scores': []
        }
    
    def sinkhorn_normalize(self, matrix, num_iters=20, eps=1e-6):
        """Stable Sinkhorn normalization"""
        for i in range(num_iters):
            # Row normalization with stability
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(row_sums, eps)
            
            # Column normalization with stability
            col_sums = np.sum(matrix, axis=0, keepdims=True)
            matrix = matrix / np.maximum(col_sums, eps)
        
        return matrix
    
    def get_soft_permutation(self, use_ema=False):
        """Get soft permutation matrix with stability techniques"""
        # Use EMA or current logits
        logits = self.ema_logits if use_ema else self.logits
        
        # Add noise for regularization
        if self.config.noise_std > 0:
            noise = np.random.normal(0, self.config.noise_std, logits.shape)
            logits = logits + noise
        
        # Temperature scaling
        temp = max(self.config.temperature, self.config.min_temperature)
        scaled_logits = logits / temp
        
        # Softmax + Sinkhorn
        soft_matrix = softmax(scaled_logits, axis=1)
        doubly_stochastic = self.sinkhorn_normalize(soft_matrix)
        
        return doubly_stochastic
    
    def get_hard_permutation(self, soft_matrix):
        """Convert soft to hard permutation using Hungarian algorithm"""
        cost_matrix = -soft_matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        hard_matrix = np.zeros_like(soft_matrix)
        hard_matrix[row_idx, col_idx] = 1.0
        
        return hard_matrix
    
    def compute_regularization_loss(self, soft_matrix):
        """Compute regularization losses"""
        losses = {}
        
        # Doubly stochastic constraint
        row_sums = np.sum(soft_matrix, axis=1)
        col_sums = np.sum(soft_matrix, axis=0)
        losses['doubly_stochastic'] = (
            np.mean((row_sums - 1.0) ** 2) + 
            np.mean((col_sums - 1.0) ** 2)
        )
        
        # Entropy regularization (encourage exploration)
        entropy = -np.sum(soft_matrix * np.log(soft_matrix + 1e-8))
        losses['entropy'] = -0.01 * entropy  # Negative for maximization
        
        # Orthogonality constraint
        should_be_identity = np.dot(soft_matrix, soft_matrix.T)
        identity = np.eye(self.size)
        losses['orthogonality'] = np.linalg.norm(should_be_identity - identity, 'fro') ** 2
        
        return losses
    
    def compute_gradients(self, soft_matrix, target_permutation):
        """Simplified gradient computation"""
        # Main loss: difference from target
        main_loss = np.linalg.norm(soft_matrix - target_permutation, 'fro') ** 2
        
        # Regularization losses
        reg_losses = self.compute_regularization_loss(soft_matrix)
        total_reg = self.config.reg_strength * sum(reg_losses.values())
        
        # Simple gradient approximation
        grad = 2 * (soft_matrix - target_permutation)
        
        # Add regularization gradients (simplified)
        row_sums = np.sum(soft_matrix, axis=1, keepdims=True)
        col_sums = np.sum(soft_matrix, axis=0, keepdims=True)
        
        grad += self.config.reg_strength * 2 * (
            (row_sums - 1.0) + 
            (col_sums - 1.0).T
        )
        
        return grad, main_loss + total_reg, main_loss, total_reg
    
    def clip_gradients(self, grad):
        """Apply gradient clipping"""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.config.gradient_clip:
            grad = grad * (self.config.gradient_clip / grad_norm)
        return grad, grad_norm
    
    def update_ema(self):
        """Update exponential moving average"""
        decay = self.config.ema_decay
        self.ema_logits = decay * self.ema_logits + (1 - decay) * self.logits
    
    def anneal_temperature(self, epoch):
        """Anneal temperature for stability"""
        self.config.temperature = max(
            self.config.temperature * self.config.temp_decay,
            self.config.min_temperature
        )
    
    def train_step(self, target_permutation, epoch):
        """Single training step with stability measures"""
        # Forward pass
        soft_matrix = self.get_soft_permutation(use_ema=(epoch > 10))
        
        # Compute gradients
        grad, total_loss, main_loss, reg_loss = self.compute_gradients(
            soft_matrix, target_permutation
        )
        
        # Clip gradients
        clipped_grad, grad_norm = self.clip_gradients(grad)
        
        # Update logits
        self.logits -= self.config.learning_rate * clipped_grad
        
        # Update EMA
        self.update_ema()
        
        # Anneal temperature
        self.anneal_temperature(epoch)
        
        # Compute permutation quality
        hard_matrix = self.get_hard_permutation(soft_matrix)
        perm_error = np.linalg.norm(hard_matrix - target_permutation, 'fro')
        
        # Stability score (higher = more stable)
        stability_score = 1.0 / (1.0 + grad_norm + perm_error)
        
        # Store history
        self.history['losses'].append(total_loss)
        self.history['temperatures'].append(self.config.temperature)
        self.history['gradient_norms'].append(grad_norm)
        self.history['permutation_errors'].append(perm_error)
        self.history['stability_scores'].append(stability_score)
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'reg_loss': reg_loss,
            'grad_norm': grad_norm,
            'perm_error': perm_error,
            'stability_score': stability_score,
            'soft_matrix': soft_matrix,
            'hard_matrix': hard_matrix
        }

def create_target_permutation(size=26, seed=None):
    """Create a target permutation matrix to learn"""
    if seed:
        np.random.seed(seed)
    
    # Create random permutation
    perm_indices = np.random.permutation(size)
    target = np.zeros((size, size))
    target[np.arange(size), perm_indices] = 1.0
    
    return target

def run_stable_training_comparison():
    """Compare stable vs unstable training configurations"""
    print("🔬 STABLE vs UNSTABLE TRAINING COMPARISON")
    print("=" * 60)
    
    # Create target permutation
    target = create_target_permutation(size=8, seed=42)  # Smaller for demo
    epochs = 100
    
    # Stable configuration
    stable_config = StabilityConfig()
    stable_config.learning_rate = 0.005
    stable_config.gradient_clip = 0.5
    stable_config.reg_strength = 1.0
    stable_config.temperature = 1.5
    
    # Unstable configuration
    unstable_config = StabilityConfig()
    unstable_config.learning_rate = 0.05  # Too high
    unstable_config.gradient_clip = 5.0   # Too high
    unstable_config.reg_strength = 0.1    # Too low
    unstable_config.temperature = 0.1     # Too low
    
    # Train both models
    print("Training stable model...")
    stable_learner = StablePermutationLearner(size=8, config=stable_config)
    stable_results = []
    
    for epoch in range(epochs):
        result = stable_learner.train_step(target, epoch)
        stable_results.append(result)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss={result['total_loss']:.4f}, "
                  f"Grad={result['grad_norm']:.4f}, "
                  f"Stability={result['stability_score']:.4f}")
    
    print("\nTraining unstable model...")
    unstable_learner = StablePermutationLearner(size=8, config=unstable_config)
    unstable_results = []
    
    for epoch in range(epochs):
        result = unstable_learner.train_step(target, epoch)
        unstable_results.append(result)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss={result['total_loss']:.4f}, "
                  f"Grad={result['grad_norm']:.4f}, "
                  f"Stability={result['stability_score']:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Stable vs Unstable Training Comparison', fontsize=16)
    
    # Loss comparison
    stable_losses = [r['total_loss'] for r in stable_results]
    unstable_losses = [r['total_loss'] for r in unstable_results]
    
    axes[0,0].semilogy(stable_losses, 'b-', label='Stable', linewidth=2)
    axes[0,0].semilogy(unstable_losses, 'r-', label='Unstable', linewidth=2)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Gradient norms
    stable_grads = [r['grad_norm'] for r in stable_results]
    unstable_grads = [r['grad_norm'] for r in unstable_results]
    
    axes[0,1].semilogy(stable_grads, 'b-', label='Stable')
    axes[0,1].semilogy(unstable_grads, 'r-', label='Unstable')
    axes[0,1].set_title('Gradient Norms')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Stability scores
    stable_stability = [r['stability_score'] for r in stable_results]
    unstable_stability = [r['stability_score'] for r in unstable_results]
    
    axes[0,2].plot(stable_stability, 'b-', label='Stable')
    axes[0,2].plot(unstable_stability, 'r-', label='Unstable')
    axes[0,2].set_title('Stability Score')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Temperature annealing
    axes[1,0].plot(stable_learner.history['temperatures'], 'b-', label='Stable')
    axes[1,0].plot(unstable_learner.history['temperatures'], 'r-', label='Unstable')
    axes[1,0].set_title('Temperature Annealing')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Permutation errors
    stable_errors = [r['perm_error'] for r in stable_results]
    unstable_errors = [r['perm_error'] for r in unstable_results]
    
    axes[1,1].semilogy(stable_errors, 'b-', label='Stable')
    axes[1,1].semilogy(unstable_errors, 'r-', label='Unstable')
    axes[1,1].set_title('Permutation Error')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # Final learned matrices
    final_stable = stable_results[-1]['hard_matrix']
    im = axes[1,2].imshow(final_stable, cmap='Blues', interpolation='nearest')
    axes[1,2].set_title('Final Stable Permutation')
    plt.colorbar(im, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('/home/user/Documents/enigmalorenz/CascadeProjects/windsurf-project/stable_vs_unstable_training.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    stable_final_loss = stable_losses[-1]
    unstable_final_loss = unstable_losses[-1]
    
    stable_final_error = stable_errors[-1]
    unstable_final_error = unstable_errors[-1]
    
    stable_avg_grad = np.mean(stable_grads[-20:])  # Last 20 epochs
    unstable_avg_grad = np.mean(unstable_grads[-20:])
    
    stable_final_stability = stable_stability[-1]
    unstable_final_stability = unstable_stability[-1]
    
    print(f"STABLE MODEL:")
    print(f"  Final Loss: {stable_final_loss:.6f}")
    print(f"  Final Permutation Error: {stable_final_error:.6f}")
    print(f"  Average Gradient Norm (final): {stable_avg_grad:.6f}")
    print(f"  Final Stability Score: {stable_final_stability:.6f}")
    
    print(f"\nUNSTABLE MODEL:")
    print(f"  Final Loss: {unstable_final_loss:.6f}")
    print(f"  Final Permutation Error: {unstable_final_error:.6f}")
    print(f"  Average Gradient Norm (final): {unstable_avg_grad:.6f}")
    print(f"  Final Stability Score: {unstable_final_stability:.6f}")
    
    print(f"\nIMPROVEMENT RATIOS:")
    print(f"  Loss Improvement: {unstable_final_loss / stable_final_loss:.2f}x")
    print(f"  Error Improvement: {unstable_final_error / stable_final_error:.2f}x")
    print(f"  Gradient Stability: {unstable_avg_grad / stable_avg_grad:.2f}x")
    print(f"  Stability Score: {stable_final_stability / unstable_final_stability:.2f}x")
    
    # Show final permutations
    print(f"\nTARGET PERMUTATION (8x8):")
    print(target.astype(int))
    
    print(f"\nSTABLE MODEL FINAL RESULT:")
    print(final_stable.astype(int))
    
    print(f"\nUNSTABLE MODEL FINAL RESULT:")
    final_unstable = unstable_results[-1]['hard_matrix']
    print(final_unstable.astype(int))
    
    # Check if stable model learned correctly
    stable_matches = np.sum(final_stable == target) / target.size
    unstable_matches = np.sum(final_unstable == target) / target.size
    
    print(f"\nACCURACY:")
    print(f"  Stable Model: {stable_matches:.1%}")
    print(f"  Unstable Model: {unstable_matches:.1%}")
    
    return stable_learner, unstable_learner, stable_results, unstable_results

if __name__ == "__main__":
    stable_learner, unstable_learner, stable_results, unstable_results = run_stable_training_comparison()