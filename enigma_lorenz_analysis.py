import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import circulant
import pandas as pd
from typing import List, Tuple, Dict
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class EnigmaRotor:
    """Represents a single Enigma rotor with permutation mapping"""
    
    def __init__(self, wiring: str, position: int = 0, ring_setting: int = 0):
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.wiring = wiring.upper()
        self.position = position % 26
        self.ring_setting = ring_setting % 26
        self.permutation_matrix = self._create_permutation_matrix()
    
    def _create_permutation_matrix(self) -> np.ndarray:
        """Create 26x26 permutation matrix for this rotor"""
        matrix = np.zeros((26, 26))
        for i, char in enumerate(self.alphabet):
            output_char = self.wiring[i]
            j = self.alphabet.index(output_char)
            matrix[i, j] = 1
        return matrix
    
    def encode_char(self, char: str) -> str:
        """Encode a single character through this rotor"""
        if char not in self.alphabet:
            return char
            
        # Apply rotor position offset
        input_pos = (self.alphabet.index(char) + self.position - self.ring_setting) % 26
        output_char = self.wiring[input_pos]
        output_pos = (self.alphabet.index(output_char) - self.position + self.ring_setting) % 26
        
        return self.alphabet[output_pos]
    
    def step(self):
        """Advance rotor position by one"""
        self.position = (self.position + 1) % 26

class EnigmaMachine:
    """3-rotor Enigma machine implementation"""
    
    def __init__(self, rotor_configs: List[Tuple[str, int]], reflector: str = None):
        # Default rotor wirings (simplified)
        default_rotors = [
            'EKMFLGDQVZNTOWYHXUSPAIBRCJ',  # Rotor I
            'AJDKSIRUXBLHWTMCQGZNPYFVOE',  # Rotor II  
            'BDFHJLCPRTXVZNYEIWGAKMUSQO'   # Rotor III
        ]
        
        self.rotors = []
        for i, (wiring, position) in enumerate(rotor_configs):
            if wiring is None:
                wiring = default_rotors[i]
            self.rotors.append(EnigmaRotor(wiring, position))
        
        # Default reflector
        self.reflector = reflector or 'YRUHQSLDPXNGOKMIEBFZCWVJAT'
        
    def encode_message(self, message: str) -> str:
        """Encode entire message through Enigma machine"""
        encoded = []
        
        for char in message.upper():
            if char not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                encoded.append(char)
                continue
                
            # Step rotors (simplified stepping)
            self.rotors[0].step()
            if self.rotors[0].position == 0:
                self.rotors[1].step()
                if self.rotors[1].position == 0:
                    self.rotors[2].step()
            
            # Forward pass through rotors
            current_char = char
            for rotor in self.rotors:
                current_char = rotor.encode_char(current_char)
            
            # Reflector
            pos = ord(current_char) - ord('A')
            current_char = self.reflector[pos]
            
            # Backward pass through rotors (reversed)
            for rotor in reversed(self.rotors):
                # Simplified reverse encoding
                input_pos = rotor.wiring.index(current_char)
                current_char = rotor.alphabet[input_pos]
            
            encoded.append(current_char)
        
        return ''.join(encoded)

class LorenzCipher:
    """Lorenz cipher implementation using XOR operations"""
    
    @staticmethod
    def text_to_binary(text: str) -> np.ndarray:
        """Convert text to binary array (5-bit Baudot-like encoding)"""
        binary_array = []
        for char in text.upper():
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                # 5-bit encoding A=0, B=1, ..., Z=25
                val = ord(char) - ord('A')
                binary_array.extend([int(x) for x in format(val, '05b')])
        return np.array(binary_array)
    
    @staticmethod
    def binary_to_text(binary: np.ndarray) -> str:
        """Convert binary array back to text"""
        text = []
        for i in range(0, len(binary), 5):
            if i + 4 < len(binary):
                val = int(''.join(map(str, binary[i:i+5])), 2)
                if val < 26:
                    text.append(chr(val + ord('A')))
        return ''.join(text)
    
    @staticmethod
    def extract_mask(plaintext: str, ciphertext: str) -> np.ndarray:
        """Extract Lorenz-style mask from plaintext and ciphertext"""
        plain_binary = LorenzCipher.text_to_binary(plaintext)
        cipher_binary = LorenzCipher.text_to_binary(ciphertext)
        
        # Ensure same length
        min_len = min(len(plain_binary), len(cipher_binary))
        plain_binary = plain_binary[:min_len]
        cipher_binary = cipher_binary[:min_len]
        
        # XOR to get mask
        mask = np.bitwise_xor(plain_binary, cipher_binary)
        return mask

class PermutationMatrixAnalyzer:
    """Statistical analysis of permutation matrices and their relationship to Lorenz masks"""
    
    def __init__(self):
        self.alphabet_size = 26
    
    def create_doubly_stochastic_matrix(self, perm_matrix: np.ndarray) -> np.ndarray:
        """Convert permutation matrix to doubly stochastic using magic square relation"""
        # Use Sinkhorn-Knopp algorithm approximation
        matrix = perm_matrix.astype(float)
        
        # Add small noise to break ties
        matrix += np.random.normal(0, 0.01, matrix.shape)
        
        # Iterative normalization
        for _ in range(100):
            # Row normalization
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(row_sums, 1e-10)
            
            # Column normalization  
            col_sums = np.sum(matrix, axis=0, keepdims=True)
            matrix = matrix / np.maximum(col_sums, 1e-10)
        
        return matrix
    
    def extract_matrix_features(self, matrix: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from a matrix"""
        features = {}
        
        # Basic statistics
        features['trace'] = np.trace(matrix)
        features['determinant'] = np.linalg.det(matrix)
        features['frobenius_norm'] = np.linalg.norm(matrix, 'fro')
        features['spectral_norm'] = np.linalg.norm(matrix, 2)
        
        # Eigenvalue statistics
        eigenvals = np.linalg.eigvals(matrix)
        features['max_eigenval'] = np.max(np.real(eigenvals))
        features['min_eigenval'] = np.min(np.real(eigenvals))
        features['eigenval_variance'] = np.var(np.real(eigenvals))
        
        # Matrix structure features
        features['sparsity'] = np.sum(matrix == 0) / matrix.size
        features['symmetry'] = np.linalg.norm(matrix - matrix.T, 'fro')
        features['diagonal_dominance'] = np.sum(np.abs(np.diag(matrix))) / np.sum(np.abs(matrix))
        
        # Entropy-like measures
        flat_matrix = matrix.flatten()
        flat_matrix = flat_matrix[flat_matrix > 0]
        if len(flat_matrix) > 0:
            features['matrix_entropy'] = -np.sum(flat_matrix * np.log(flat_matrix + 1e-10))
        else:
            features['matrix_entropy'] = 0
            
        return features
    
    def analyze_mask_statistics(self, mask: np.ndarray) -> Dict[str, float]:
        """Analyze statistical properties of Lorenz mask"""
        stats_dict = {}
        
        # Basic statistics
        stats_dict['mean'] = np.mean(mask)
        stats_dict['variance'] = np.var(mask)
        stats_dict['entropy'] = stats.entropy(np.bincount(mask) + 1)
        
        # Autocorrelation analysis
        if len(mask) > 1:
            autocorr = np.correlate(mask, mask, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            stats_dict['autocorr_1'] = autocorr[1] / autocorr[0] if len(autocorr) > 1 else 0
            stats_dict['autocorr_2'] = autocorr[2] / autocorr[0] if len(autocorr) > 2 else 0
        else:
            stats_dict['autocorr_1'] = 0
            stats_dict['autocorr_2'] = 0
        
        # Run length analysis
        runs = []
        current_run = 1
        for i in range(1, len(mask)):
            if mask[i] == mask[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        stats_dict['avg_run_length'] = np.mean(runs)
        stats_dict['run_variance'] = np.var(runs)
        
        # Frequency analysis
        ones_freq = np.sum(mask) / len(mask)
        stats_dict['ones_frequency'] = ones_freq
        stats_dict['balance'] = abs(0.5 - ones_freq)
        
        return stats_dict
    
    def correlate_matrices_and_mask(self, rotor_matrices: List[np.ndarray], 
                                  rotor_positions: List[int], mask: np.ndarray) -> Dict[str, float]:
        """Find statistical correlations between rotor matrices/positions and Lorenz mask"""
        correlations = {}
        
        # Extract features from all rotor matrices
        all_matrix_features = []
        for i, matrix in enumerate(rotor_matrices):
            ds_matrix = self.create_doubly_stochastic_matrix(matrix)
            features = self.extract_matrix_features(ds_matrix)
            for key, value in features.items():
                all_matrix_features.append(value)
                correlations[f'rotor_{i}_{key}'] = value
        
        # Add rotor positions
        for i, pos in enumerate(rotor_positions):
            correlations[f'rotor_{i}_position'] = pos
            all_matrix_features.append(pos)
        
        # Get mask statistics
        mask_stats = self.analyze_mask_statistics(mask)
        
        # Calculate correlations between matrix features and mask properties
        correlation_results = {}
        
        for mask_property, mask_value in mask_stats.items():
            correlations_with_mask = []
            
            for feature_name in [k for k in correlations.keys()]:
                feature_value = correlations[feature_name]
                
                # Create simple correlation by comparing normalized values
                if np.std([feature_value]) > 1e-10 and np.std([mask_value]) > 1e-10:
                    # Pearson correlation coefficient approximation
                    corr = np.corrcoef([feature_value], [mask_value])[0, 1]
                    if not np.isnan(corr):
                        correlation_results[f'{feature_name}_vs_{mask_property}'] = corr
        
        return correlation_results, mask_stats

def generate_test_data(num_samples: int = 100) -> Tuple[List[Dict], List[np.ndarray]]:
    """Generate test data with various rotor configurations and corresponding masks"""
    
    test_plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 3
    
    configurations = []
    masks = []
    analyzer = PermutationMatrixAnalyzer()
    
    for _ in range(num_samples):
        # Random rotor configurations
        rotor_configs = []
        for i in range(3):
            # Random wiring
            alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            np.random.shuffle(alphabet)
            wiring = ''.join(alphabet)
            position = np.random.randint(0, 26)
            rotor_configs.append((wiring, position))
        
        # Create Enigma machine and encode
        enigma = EnigmaMachine(rotor_configs)
        ciphertext = enigma.encode_message(test_plaintext)
        
        # Extract Lorenz mask
        lorenz = LorenzCipher()
        mask = lorenz.extract_mask(test_plaintext, ciphertext)
        
        # Store configuration data
        config_data = {
            'rotor_configs': rotor_configs,
            'rotor_matrices': [rotor.permutation_matrix for rotor in enigma.rotors],
            'rotor_positions': [rotor.position for rotor in enigma.rotors],
            'mask': mask
        }
        
        configurations.append(config_data)
        masks.append(mask)
    
    return configurations, masks

def main():
    """Main analysis function"""
    print("Generating test data...")
    
    configurations, masks = generate_test_data(50)
    analyzer = PermutationMatrixAnalyzer()
    
    print("Analyzing correlations between rotor configurations and Lorenz masks...")
    
    all_correlations = []
    all_mask_stats = []
    
    for i, config in enumerate(configurations):
        correlations, mask_stats = analyzer.correlate_matrices_and_mask(
            config['rotor_matrices'], 
            config['rotor_positions'], 
            config['mask']
        )
        
        all_correlations.append(correlations)
        all_mask_stats.append(mask_stats)
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(configurations)} configurations")
    
    # Aggregate results
    print("\nAnalyzing aggregate patterns...")
    
    # Find most significant correlations
    correlation_summary = {}
    for corr_dict in all_correlations:
        for key, value in corr_dict.items():
            if key not in correlation_summary:
                correlation_summary[key] = []
            if not np.isnan(value):
                correlation_summary[key].append(value)
    
    # Calculate mean correlations
    mean_correlations = {}
    for key, values in correlation_summary.items():
        if len(values) > 0:
            mean_correlations[key] = np.mean(values)
    
    # Print top correlations
    print("\nTop correlations between rotor features and mask properties:")
    sorted_correlations = sorted(mean_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for key, corr in sorted_correlations[:20]:
        print(f"{key}: {corr:.4f}")
    
    # Analyze mask statistics distribution
    print("\nMask statistics summary:")
    mask_df = pd.DataFrame(all_mask_stats)
    print(mask_df.describe())
    
    return configurations, all_correlations, all_mask_stats, mean_correlations

if __name__ == "__main__":
    main()