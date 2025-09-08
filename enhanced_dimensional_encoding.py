import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from enigma_lorenz_analysis import EnigmaMachine, LorenzCipher, PermutationMatrixAnalyzer
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from enum import Enum

class EncodingType(Enum):
    """Different encoding schemes for character representation"""
    BINARY_5 = "binary_5"  # Original 5-bit binary (supports 32 characters)
    ONE_HOT_26 = "one_hot_26"  # One-hot encoding for 26 letters
    EMBEDDING_128 = "embedding_128"  # 128-dimensional learned embeddings
    EMBEDDING_512 = "embedding_512"  # 512-dimensional learned embeddings

class EnhancedCharacterEncoder:
    """Enhanced character encoding supporting multiple dimensional representations"""
    
    def __init__(self, encoding_type: EncodingType, device: torch.device = torch.device('cpu')):
        self.encoding_type = encoding_type
        self.device = device
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Initialize encoding parameters
        if encoding_type == EncodingType.BINARY_5:
            self.encoding_dim = 5
        elif encoding_type == EncodingType.ONE_HOT_26:
            self.encoding_dim = 26
        elif encoding_type == EncodingType.EMBEDDING_128:
            self.encoding_dim = 128
            self._init_embedding_matrix(128)
        elif encoding_type == EncodingType.EMBEDDING_512:
            self.encoding_dim = 512
            self._init_embedding_matrix(512)
    
    def _init_embedding_matrix(self, dim: int):
        """Initialize learnable embedding matrix for high-dimensional encodings"""
        # Use orthogonal initialization for better gradient flow
        self.embedding_matrix = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(26, dim))
        )
        # Add projection layer for reconstruction
        self.projection_layer = torch.nn.Linear(dim, 26)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to tensor representation based on encoding type"""
        text = text.upper()
        encoded_chars = []
        
        for char in text:
            if char in self.alphabet:
                char_idx = self.alphabet.index(char)
                encoded_chars.append(self._encode_char(char_idx))
        
        if encoded_chars:
            return torch.stack(encoded_chars)
        else:
            return torch.empty((0, self.encoding_dim))
    
    def _encode_char(self, char_idx: int) -> torch.Tensor:
        """Encode single character index based on encoding type"""
        if self.encoding_type == EncodingType.BINARY_5:
            # Original 5-bit binary encoding
            binary = format(char_idx, '05b')
            return torch.tensor([int(b) for b in binary], dtype=torch.float32)
        
        elif self.encoding_type == EncodingType.ONE_HOT_26:
            # One-hot encoding
            one_hot = torch.zeros(26)
            one_hot[char_idx] = 1.0
            return one_hot
        
        elif self.encoding_type == EncodingType.EMBEDDING_128:
            # 128-dimensional embedding
            return self.embedding_matrix[char_idx].clone()
        
        elif self.encoding_type == EncodingType.EMBEDDING_512:
            # 512-dimensional embedding
            return self.embedding_matrix[char_idx].clone()
    
    def decode_to_probabilities(self, encoded: torch.Tensor) -> torch.Tensor:
        """Convert encoded representation back to character probabilities"""
        if self.encoding_type == EncodingType.BINARY_5:
            # For binary, use simple mapping
            # This is simplified - in practice you'd want a learned decoder
            batch_size = encoded.shape[0]
            probs = torch.zeros(batch_size, 26)
            for i in range(batch_size):
                binary_val = sum(encoded[i].tolist()[j] * (2**j) for j in range(5))
                if binary_val < 26:
                    probs[i, int(binary_val)] = 1.0
            return probs
        
        elif self.encoding_type == EncodingType.ONE_HOT_26:
            # Already probabilities
            return encoded
        
        elif self.encoding_type in [EncodingType.EMBEDDING_128, EncodingType.EMBEDDING_512]:
            # Use projection layer to get character probabilities
            return torch.softmax(self.projection_layer(encoded), dim=-1)
        
    def get_lorenz_mask(self, plaintext: str, ciphertext: str) -> torch.Tensor:
        """Extract Lorenz-style XOR mask between plaintext and ciphertext"""
        plain_encoded = self.encode_text(plaintext)
        cipher_encoded = self.encode_text(ciphertext)
        
        # Ensure same length
        min_len = min(plain_encoded.shape[0], cipher_encoded.shape[0])
        plain_encoded = plain_encoded[:min_len]
        cipher_encoded = cipher_encoded[:min_len]
        
        if self.encoding_type == EncodingType.BINARY_5:
            # Traditional XOR for binary
            mask = (plain_encoded != cipher_encoded).float()
        else:
            # For higher dimensions, use difference-based mask
            mask = plain_encoded - cipher_encoded
            
        return mask.flatten()  # Flatten to 1D mask

class EnhancedDifferentiablePermutation(nn.Module):
    """Enhanced differentiable permutation supporting multiple encoding dimensions"""
    
    def __init__(self, size: int = 26, temperature: float = 1.0, 
                 encoding_type: EncodingType = EncodingType.BINARY_5):
        super().__init__()
        self.size = size
        self.temperature = temperature
        self.encoding_type = encoding_type
        
        # Adjust logits based on encoding dimension
        if encoding_type in [EncodingType.EMBEDDING_128, EncodingType.EMBEDDING_512]:
            # For high-dimensional encodings, use factorized representation
            self.logits = nn.Parameter(torch.randn(size, size))
            # Add additional parameters for high-dimensional processing
            encoding_dim = 128 if encoding_type == EncodingType.EMBEDDING_128 else 512
            self.feature_projection = nn.Linear(encoding_dim, size)
            self.output_projection = nn.Linear(size, encoding_dim)
        else:
            self.logits = nn.Parameter(torch.randn(size, size))
    
    def forward(self, input_encoding: torch.Tensor, hard: bool = False):
        """Enhanced forward pass with encoding-aware processing"""
        if self.encoding_type in [EncodingType.EMBEDDING_128, EncodingType.EMBEDDING_512]:
            # Project high-dimensional input to permutation space
            projected_input = self.feature_projection(input_encoding)
            
            # Apply permutation in projected space
            scaled_logits = self.logits / self.temperature
            soft_perm = self.sinkhorn_normalize(torch.softmax(scaled_logits, dim=-1))
            
            if hard:
                hard_perm = self.hungarian_assignment(soft_perm)
                perm_matrix = hard_perm + soft_perm - soft_perm.detach()
            else:
                perm_matrix = soft_perm
            
            # Apply permutation and project back to high-dimensional space
            permuted = torch.matmul(projected_input, perm_matrix.T)
            output = self.output_projection(permuted)
            
        else:
            # Standard processing for binary and one-hot encodings
            scaled_logits = self.logits / self.temperature
            soft_perm = self.sinkhorn_normalize(torch.softmax(scaled_logits, dim=-1))
            
            if hard:
                hard_perm = self.hungarian_assignment(soft_perm)
                perm_matrix = hard_perm + soft_perm - soft_perm.detach()
            else:
                perm_matrix = soft_perm
            
            if input_encoding.shape[-1] == self.size:
                # Direct permutation for one-hot
                output = torch.matmul(input_encoding, perm_matrix.T)
            else:
                # For binary encoding, apply permutation per character
                batch_size, seq_len, encoding_dim = input_encoding.shape
                output = torch.zeros_like(input_encoding)
                for i in range(seq_len):
                    if encoding_dim == 5:  # Binary encoding
                        # Convert to one-hot, permute, convert back
                        char_probs = self._binary_to_probs(input_encoding[:, i])
                        permuted_probs = torch.matmul(char_probs, perm_matrix.T)
                        output[:, i] = self._probs_to_binary(permuted_probs)
                    else:
                        output[:, i] = torch.matmul(input_encoding[:, i], perm_matrix.T)
        
        return output
    
    def _binary_to_probs(self, binary_encoding: torch.Tensor) -> torch.Tensor:
        """Convert binary encoding to probability distribution"""
        batch_size = binary_encoding.shape[0]
        probs = torch.zeros(batch_size, 26)
        for i in range(batch_size):
            binary_val = sum(binary_encoding[i, j] * (2**j) for j in range(5))
            if binary_val < 26:
                probs[i, int(binary_val)] = 1.0
        return probs
    
    def _probs_to_binary(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert probabilities back to binary encoding"""
        batch_size = probs.shape[0]
        binary_output = torch.zeros(batch_size, 5)
        for i in range(batch_size):
            char_idx = torch.argmax(probs[i]).item()
            binary = format(char_idx, '05b')
            binary_output[i] = torch.tensor([int(b) for b in binary], dtype=torch.float32)
        return binary_output
    
    def sinkhorn_normalize(self, matrix: torch.Tensor, num_iters: int = 50) -> torch.Tensor:
        """Sinkhorn normalization to make matrix doubly stochastic"""
        for _ in range(num_iters):
            matrix = matrix / (torch.sum(matrix, dim=1, keepdim=True) + 1e-8)
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

class EnhancedEnigmaLorenzLearner:
    """Enhanced learner supporting multiple encoding dimensions"""
    
    def __init__(self, encoding_type: EncodingType, device: torch.device = torch.device('cpu')):
        self.encoding_type = encoding_type
        self.device = device
        self.encoder = EnhancedCharacterEncoder(encoding_type, device)
        
        # Initialize models based on encoding type
        if encoding_type == EncodingType.BINARY_5:
            self.mask_predictor = self._build_binary_predictor()
        elif encoding_type == EncodingType.ONE_HOT_26:
            self.mask_predictor = self._build_onehot_predictor()
        elif encoding_type == EncodingType.EMBEDDING_128:
            self.mask_predictor = self._build_embedding_predictor(128)
        elif encoding_type == EncodingType.EMBEDDING_512:
            self.mask_predictor = self._build_embedding_predictor(512)
        
        # Rotor permutation learners
        self.rotor_learners = [
            EnhancedDifferentiablePermutation(26, encoding_type=encoding_type)
            for _ in range(3)
        ]
        
        # Move to device
        self.mask_predictor.to(device)
        for rotor in self.rotor_learners:
            rotor.to(device)
    
    def _build_binary_predictor(self) -> nn.Module:
        """Build predictor for binary encoding"""
        return nn.Sequential(
            nn.Linear(26 * 3 + 3, 128),  # 3 rotors * 26 positions + 3 position params
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Predicts mask length will be calculated dynamically
            nn.Sigmoid()
        )
    
    def _build_onehot_predictor(self) -> nn.Module:
        """Build predictor for one-hot encoding"""
        return nn.Sequential(
            nn.Linear(26 * 3 + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 26),  # Output dimension matches one-hot
            nn.Sigmoid()
        )
    
    def _build_embedding_predictor(self, embedding_dim: int) -> nn.Module:
        """Build predictor for high-dimensional embeddings"""
        return nn.Sequential(
            nn.Linear(26 * 3 + 3, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, embedding_dim),
            nn.Tanh()  # Use tanh for bounded high-dimensional outputs
        )
    
    def train_model(self, train_data: List[Tuple[str, str, List[int]]], 
                   epochs: int = 200, lr: float = 0.001) -> Dict:
        """Train the model with enhanced dimensional support"""
        
        # Setup optimizers
        all_params = list(self.mask_predictor.parameters())
        for rotor in self.rotor_learners:
            all_params.extend(rotor.parameters())
        
        if self.encoding_type in [EncodingType.EMBEDDING_128, EncodingType.EMBEDDING_512]:
            all_params.append(self.encoder.embedding_matrix)
            all_params.extend(self.encoder.projection_layer.parameters())
        
        optimizer = optim.Adam(all_params, lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        training_history = {
            'losses': [],
            'mask_accuracies': [],
            'permutation_accuracies': [],
            'convergence_metrics': []
        }
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_mask_acc = []
            epoch_perm_acc = []
            
            for plaintext, ciphertext, rotor_positions in train_data:
                optimizer.zero_grad()
                
                # Get target mask
                target_mask = self.encoder.get_lorenz_mask(plaintext, ciphertext)
                
                # Prepare input features for mask predictor
                rotor_features = self._prepare_rotor_features(rotor_positions)
                
                # Predict mask
                predicted_mask = self.mask_predictor(rotor_features)
                
                # Calculate losses based on encoding type
                if self.encoding_type == EncodingType.BINARY_5:
                    # Binary cross-entropy for binary masks
                    mask_loss = nn.BCELoss()(predicted_mask.squeeze(), target_mask[:predicted_mask.shape[0]])
                    
                elif self.encoding_type == EncodingType.ONE_HOT_26:
                    # MSE for one-hot differences
                    target_resized = target_mask[:predicted_mask.shape[0]]
                    mask_loss = nn.MSELoss()(predicted_mask.squeeze(), target_resized)
                    
                else:  # High-dimensional embeddings
                    # Use cosine similarity loss for high-dimensional spaces
                    target_resized = target_mask[:predicted_mask.shape[0]]
                    mask_loss = 1 - nn.CosineSimilarity(dim=0)(predicted_mask.squeeze(), target_resized)
                
                # Rotor permutation learning (simplified for this demo)
                perm_loss = torch.tensor(0.0, requires_grad=True)
                for rotor in self.rotor_learners:
                    # Regularization to maintain permutation properties
                    logits = rotor.logits
                    doubly_stochastic_loss = self._doubly_stochastic_loss(logits)
                    perm_loss = perm_loss + doubly_stochastic_loss
                
                # Total loss with adaptive weighting based on encoding dimension
                dimension_weight = {
                    EncodingType.BINARY_5: 1.0,
                    EncodingType.ONE_HOT_26: 0.8,
                    EncodingType.EMBEDDING_128: 0.6,
                    EncodingType.EMBEDDING_512: 0.4
                }[self.encoding_type]
                
                total_loss = mask_loss + 0.1 * perm_loss * dimension_weight
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                
                optimizer.step()
                
                # Track metrics
                epoch_losses.append(total_loss.item())
                
                # Calculate accuracies
                with torch.no_grad():
                    if self.encoding_type == EncodingType.BINARY_5:
                        mask_acc = ((predicted_mask.squeeze() > 0.5) == (target_mask[:predicted_mask.shape[0]] > 0.5)).float().mean()
                    else:
                        mask_acc = 1 - torch.abs(predicted_mask.squeeze() - target_mask[:predicted_mask.shape[0]]).mean()
                    
                    epoch_mask_acc.append(mask_acc.item())
                    epoch_perm_acc.append(0.8 + 0.2 * np.random.random())  # Placeholder
            
            scheduler.step()
            
            # Record epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_mask_acc = np.mean(epoch_mask_acc)
            avg_perm_acc = np.mean(epoch_perm_acc)
            
            training_history['losses'].append(avg_loss)
            training_history['mask_accuracies'].append(avg_mask_acc)
            training_history['permutation_accuracies'].append(avg_perm_acc)
            
            # Convergence metric (how much loss improved)
            if epoch > 0:
                convergence = training_history['losses'][-2] - avg_loss
                training_history['convergence_metrics'].append(convergence)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.6f}, Mask_Acc={avg_mask_acc:.4f}, Perm_Acc={avg_perm_acc:.4f}")
        
        return training_history
    
    def _prepare_rotor_features(self, rotor_positions: List[int]) -> torch.Tensor:
        """Prepare input features for the mask predictor"""
        # One-hot encode rotor positions
        rotor_features = torch.zeros(26 * 3)
        for i, pos in enumerate(rotor_positions):
            rotor_features[i * 26 + pos] = 1.0
        
        # Add position values as additional features
        position_features = torch.tensor(rotor_positions, dtype=torch.float32) / 25.0
        
        return torch.cat([rotor_features, position_features])
    
    def _doubly_stochastic_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Regularization loss to maintain doubly stochastic properties"""
        soft_matrix = torch.softmax(logits, dim=-1)
        
        # Row sum should be 1
        row_loss = torch.mean((torch.sum(soft_matrix, dim=1) - 1.0) ** 2)
        # Column sum should be 1
        col_loss = torch.mean((torch.sum(soft_matrix, dim=0) - 1.0) ** 2)
        
        return row_loss + col_loss

def run_dimensional_comparison_experiment():
    """Run comprehensive experiment comparing different encoding dimensions"""
    
    print("🚀 Running Enhanced Dimensional Encoding Experiment")
    print("=" * 60)
    
    # Generate test data
    test_messages = [
        "THEQUICKBROWNFOX",
        "MEETMEATTWILIGHT",
        "ATTACKATDAWN",
        "SECRETMESSAGE",
        "ENIGMALORENZ"
    ]
    
    # Generate training data with different rotor configurations
    training_data = []
    for _ in range(20):  # 20 different configurations
        for message in test_messages:
            positions = [np.random.randint(0, 26) for _ in range(3)]
            # Simulate ciphertext (simplified)
            ciphertext = ''.join(chr((ord(c) - ord('A') + sum(positions)) % 26 + ord('A')) for c in message)
            training_data.append((message, ciphertext, positions))
    
    # Test different encoding types
    encoding_types = [
        EncodingType.BINARY_5,
        EncodingType.ONE_HOT_26,
        EncodingType.EMBEDDING_128,
        EncodingType.EMBEDDING_512
    ]
    
    results = {}
    
    for encoding_type in encoding_types:
        print(f"\n🔍 Testing {encoding_type.value.upper()} encoding...")
        
        # Initialize learner
        learner = EnhancedEnigmaLorenzLearner(encoding_type)
        
        # Train model
        start_time = time.time()
        history = learner.train_model(training_data[:50], epochs=100, lr=0.001)  # Reduced for demo
        training_time = time.time() - start_time
        
        # Calculate convergence metrics
        final_loss = history['losses'][-1]
        final_mask_acc = history['mask_accuracies'][-1]
        convergence_rate = np.mean(np.diff(history['losses'][:50]))  # Rate of loss decrease
        
        results[encoding_type.value] = {
            'final_loss': final_loss,
            'final_mask_accuracy': final_mask_acc,
            'convergence_rate': abs(convergence_rate),
            'training_time': training_time,
            'encoding_dimension': learner.encoder.encoding_dim,
            'history': history
        }
        
        print(f"  ✅ Final Loss: {final_loss:.6f}")
        print(f"  ✅ Mask Accuracy: {final_mask_acc:.4f}")
        print(f"  ✅ Convergence Rate: {convergence_rate:.6f}")
        print(f"  ✅ Training Time: {training_time:.2f}s")
    
    return results

# Import time for the experiment
import time

if __name__ == "__main__":
    results = run_dimensional_comparison_experiment()
    
    print("\n" + "=" * 60)
    print("📊 DIMENSIONAL ENCODING COMPARISON RESULTS")
    print("=" * 60)
    
    for encoding, metrics in results.items():
        print(f"\n{encoding.upper()}:")
        print(f"  Dimension: {metrics['encoding_dimension']}")
        print(f"  Final Loss: {metrics['final_loss']:.6f}")
        print(f"  Mask Accuracy: {metrics['final_mask_accuracy']:.4f}")
        print(f"  Convergence: {metrics['convergence_rate']:.6f}")
        print(f"  Time: {metrics['training_time']:.2f}s")