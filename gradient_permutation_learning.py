import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from enigma_lorenz_analysis import EnigmaMachine, LorenzCipher, PermutationMatrixAnalyzer
from scipy.optimize import linear_sum_assignment
import seaborn as sns

class DifferentiablePermutation(nn.Module):
    """Differentiable approximation of permutation matrices using Sinkhorn normalization"""
    
    def __init__(self, size: int = 26, temperature: float = 1.0):
        super().__init__()
        self.size = size
        self.temperature = temperature
        self.logits = nn.Parameter(torch.randn(size, size))
        
    def forward(self, hard: bool = False):
        """Forward pass with optional hard assignment"""
        # Apply temperature scaling
        scaled_logits = self.logits / self.temperature
        
        # Sinkhorn normalization for doubly stochastic matrix
        soft_perm = self.sinkhorn_normalize(torch.softmax(scaled_logits, dim=-1))
        
        if hard:
            # Hard assignment using Hungarian algorithm
            hard_perm = self.hungarian_assignment(soft_perm)
            # Straight-through estimator
            return hard_perm + soft_perm - soft_perm.detach()
        
        return soft_perm
    
    def sinkhorn_normalize(self, matrix: torch.Tensor, num_iters: int = 50) -> torch.Tensor:
        """Sinkhorn normalization to make matrix doubly stochastic"""
        for _ in range(num_iters):
            # Row normalization
            matrix = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8)
            # Column normalization
            matrix = matrix / (torch.sum(matrix, dim=0, keepdim=True) + 1e-8)
        return matrix
    
    def hungarian_assignment(self, soft_matrix: torch.Tensor) -> torch.Tensor:
        """Convert soft assignment to hard permutation using Hungarian algorithm"""
        with torch.no_grad():
            cost_matrix = -soft_matrix.cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            
            hard_matrix = torch.zeros_like(soft_matrix)
            hard_matrix[row_idx, col_idx] = 1.0
            
        return hard_matrix

class EnigmaRotorNetwork(nn.Module):
    """Neural network model of Enigma rotor with learnable permutation"""
    
    def __init__(self, alphabet_size: int = 26):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.permutation = DifferentiablePermutation(alphabet_size)
        self.position = nn.Parameter(torch.zeros(1))
        
    def forward(self, input_chars: torch.Tensor, hard_assignment: bool = False):
        """Forward pass through rotor"""
        # Get permutation matrix
        perm_matrix = self.permutation(hard=hard_assignment)
        
        # Apply position rotation (circular shift)
        position_int = torch.round(self.position) % self.alphabet_size
        shift_matrix = torch.roll(torch.eye(self.alphabet_size), shifts=int(position_int.item()), dims=0)
        
        # Apply rotor transformation
        effective_perm = torch.matmul(shift_matrix, perm_matrix)
        output = torch.matmul(input_chars, effective_perm)
        
        return output

class EnigmaNetworkModel(nn.Module):
    """Complete Enigma machine as neural network"""
    
    def __init__(self, num_rotors: int = 3, alphabet_size: int = 26):
        super().__init__()
        self.num_rotors = num_rotors
        self.alphabet_size = alphabet_size
        
        # Create rotor networks
        self.rotors = nn.ModuleList([
            EnigmaRotorNetwork(alphabet_size) for _ in range(num_rotors)
        ])
        
        # Reflector (fixed for now, could be learnable)
        reflector_perm = torch.eye(alphabet_size)
        reflector_perm = reflector_perm[torch.randperm(alphabet_size)]
        self.register_buffer('reflector', reflector_perm)
        
    def forward(self, input_text: torch.Tensor, hard_assignment: bool = False):
        """Forward pass through entire Enigma machine"""
        current = input_text
        
        # Forward pass through rotors
        for rotor in self.rotors:
            current = rotor(current, hard_assignment)
        
        # Reflector
        current = torch.matmul(current, self.reflector)
        
        # Backward pass through rotors (in reverse)
        for rotor in reversed(self.rotors):
            # For backward pass, use transpose of permutation
            perm_matrix = rotor.permutation(hard=hard_assignment)
            position_int = torch.round(rotor.position) % self.alphabet_size
            shift_matrix = torch.roll(torch.eye(self.alphabet_size), shifts=int(position_int.item()), dims=0)
            effective_perm = torch.matmul(shift_matrix, perm_matrix)
            current = torch.matmul(current, effective_perm.T)
        
        return current
    
    def step_rotors(self):
        """Step rotor positions (simplified)"""
        with torch.no_grad():
            self.rotors[0].position += 1
            if self.rotors[0].position >= self.alphabet_size:
                self.rotors[0].position = 0
                self.rotors[1].position += 1
                if self.rotors[1].position >= self.alphabet_size:
                    self.rotors[1].position = 0
                    self.rotors[2].position += 1

class LorenzMaskPredictor(nn.Module):
    """Neural network to predict Lorenz mask from rotor configurations"""
    
    def __init__(self, rotor_feature_dim: int = 26*26*3, hidden_dim: int = 512):
        super().__init__()
        self.rotor_feature_dim = rotor_feature_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(rotor_feature_dim + 3, hidden_dim),  # +3 for positions
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim//2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Predict mask bit probability
            nn.Sigmoid()
        )
        
    def forward(self, rotor_matrices: torch.Tensor, rotor_positions: torch.Tensor, sequence_length: int):
        """Predict Lorenz mask from rotor configuration"""
        batch_size = rotor_matrices.shape[0]
        
        # Flatten rotor matrices
        flattened_matrices = rotor_matrices.view(batch_size, -1)
        
        # Combine with positions
        features = torch.cat([flattened_matrices, rotor_positions], dim=1)
        
        # Encode features
        encoded = self.encoder(features)
        
        # Generate mask predictions for sequence
        mask_predictions = []
        for i in range(sequence_length):
            # Add position encoding
            pos_encoding = torch.sin(torch.tensor(i / 100.0)).repeat(batch_size, 1)
            combined = torch.cat([encoded, pos_encoding], dim=1)
            mask_bit = self.mask_predictor(combined)
            mask_predictions.append(mask_bit)
        
        return torch.cat(mask_predictions, dim=1)

class PermutationMatrixLearner:
    """Main class for learning permutation matrices through gradient-based optimization"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.enigma_model = EnigmaNetworkModel().to(device)
        self.mask_predictor = LorenzMaskPredictor().to(device)
        
    def generate_training_data(self, num_samples: int = 1000, sequence_length: int = 100):
        """Generate training data with known rotor configs and corresponding masks"""
        training_data = []
        
        test_text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 5
        
        for _ in range(num_samples):
            # Random rotor configuration
            rotor_configs = []
            for i in range(3):
                alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                np.random.shuffle(alphabet)
                wiring = ''.join(alphabet)
                position = np.random.randint(0, 26)
                rotor_configs.append((wiring, position))
            
            # Generate Enigma encoding
            enigma = EnigmaMachine(rotor_configs)
            ciphertext = enigma.encode_message(test_text[:sequence_length])
            
            # Extract Lorenz mask
            lorenz = LorenzCipher()
            mask = lorenz.extract_mask(test_text[:sequence_length], ciphertext)
            
            # Convert to tensors
            rotor_matrices = torch.stack([
                torch.tensor(rotor.permutation_matrix, dtype=torch.float32) 
                for rotor in enigma.rotors
            ])
            
            rotor_positions = torch.tensor([
                rotor.position for rotor in enigma.rotors
            ], dtype=torch.float32)
            
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            
            training_data.append({
                'rotor_matrices': rotor_matrices,
                'rotor_positions': rotor_positions,
                'mask': mask_tensor,
                'plaintext': test_text[:sequence_length],
                'ciphertext': ciphertext
            })
        
        return training_data
    
    def train_mask_predictor(self, training_data: List[Dict], epochs: int = 100, batch_size: int = 32):
        """Train the mask predictor network"""
        optimizer = optim.Adam(self.mask_predictor.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                # Prepare batch
                rotor_matrices = torch.stack([item['rotor_matrices'] for item in batch])
                rotor_positions = torch.stack([item['rotor_positions'] for item in batch])
                masks = torch.stack([item['mask'] for item in batch])
                
                rotor_matrices = rotor_matrices.to(self.device)
                rotor_positions = rotor_positions.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predicted_masks = self.mask_predictor(rotor_matrices, rotor_positions, masks.shape[1])
                
                # Loss
                loss = criterion(predicted_masks, masks.unsqueeze(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def learn_rotor_permutations(self, target_masks: List[torch.Tensor], epochs: int = 200):
        """Learn rotor permutations that generate specific Lorenz masks"""
        optimizer = optim.Adam(self.enigma_model.parameters(), lr=0.01)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for target_mask in target_masks:
                target_mask = target_mask.to(self.device)
                
                # Create dummy input
                input_length = len(target_mask)
                dummy_input = torch.eye(26)[:input_length].to(self.device)
                
                # Forward pass through Enigma model
                enigma_output = self.enigma_model(dummy_input)
                
                # Convert output to binary mask (simplified)
                binary_output = (enigma_output.sum(dim=1) > 13).float()
                
                # Loss against target mask
                loss = nn.MSELoss()(binary_output, target_mask.float())
                
                # Regularization for permutation matrices
                reg_loss = 0
                for rotor in self.enigma_model.rotors:
                    perm_matrix = rotor.permutation()
                    # Encourage doubly stochastic properties
                    row_sums = torch.sum(perm_matrix, dim=1)
                    col_sums = torch.sum(perm_matrix, dim=0)
                    reg_loss += torch.sum((row_sums - 1)**2) + torch.sum((col_sums - 1)**2)
                
                total_loss = loss + 0.01 * reg_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def evaluate_learned_permutations(self):
        """Evaluate how well learned permutations match expected behavior"""
        results = {}
        
        for i, rotor in enumerate(self.enigma_model.rotors):
            perm_matrix = rotor.permutation(hard=True)
            position = rotor.position.item()
            
            # Check if matrix is approximately permutation
            row_sums = torch.sum(perm_matrix, dim=1)
            col_sums = torch.sum(perm_matrix, dim=0)
            
            results[f'rotor_{i}'] = {
                'position': position,
                'row_sum_error': torch.mean(torch.abs(row_sums - 1)).item(),
                'col_sum_error': torch.mean(torch.abs(col_sums - 1)).item(),
                'permutation_matrix': perm_matrix.detach().cpu().numpy()
            }
        
        return results

def run_gradient_learning_experiment():
    """Run the complete gradient-based learning experiment"""
    print("Starting gradient-based permutation matrix learning experiment...")
    
    learner = PermutationMatrixLearner()
    
    # Generate training data
    print("Generating training data...")
    training_data = learner.generate_training_data(num_samples=200, sequence_length=50)
    
    # Train mask predictor
    print("Training mask predictor...")
    mask_losses = learner.train_mask_predictor(training_data, epochs=50)
    
    # Extract some target masks for rotor learning
    target_masks = [item['mask'] for item in training_data[:10]]
    
    # Learn rotor permutations
    print("Learning rotor permutations...")
    rotor_losses = learner.learn_rotor_permutations(target_masks, epochs=100)
    
    # Evaluate results
    print("Evaluating learned permutations...")
    evaluation_results = learner.evaluate_learned_permutations()
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training losses
    axes[0, 0].plot(mask_losses)
    axes[0, 0].set_title('Mask Predictor Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(rotor_losses)
    axes[0, 1].set_title('Rotor Learning Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    # Plot learned permutation matrix
    learned_perm = evaluation_results['rotor_0']['permutation_matrix']
    im1 = axes[1, 0].imshow(learned_perm, cmap='Blues')
    axes[1, 0].set_title('Learned Permutation Matrix (Rotor 0)')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Plot permutation errors
    rotor_names = list(evaluation_results.keys())
    row_errors = [evaluation_results[r]['row_sum_error'] for r in rotor_names]
    col_errors = [evaluation_results[r]['col_sum_error'] for r in rotor_names]
    
    x = np.arange(len(rotor_names))
    axes[1, 1].bar(x - 0.2, row_errors, 0.4, label='Row Sum Error')
    axes[1, 1].bar(x + 0.2, col_errors, 0.4, label='Col Sum Error')
    axes[1, 1].set_title('Permutation Matrix Errors')
    axes[1, 1].set_xlabel('Rotor')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(rotor_names)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/user/Documents/enigmalorenz/CascadeProjects/windsurf-project/gradient_learning_results.png')
    
    # Print summary
    print("\nGradient Learning Results Summary:")
    print(f"Final mask predictor loss: {mask_losses[-1]:.4f}")
    print(f"Final rotor learning loss: {rotor_losses[-1]:.4f}")
    
    for rotor_name, results in evaluation_results.items():
        print(f"\n{rotor_name}:")
        print(f"  Position: {results['position']:.2f}")
        print(f"  Row sum error: {results['row_sum_error']:.4f}")
        print(f"  Col sum error: {results['col_sum_error']:.4f}")
    
    return learner, evaluation_results, mask_losses, rotor_losses

if __name__ == "__main__":
    run_gradient_learning_experiment()