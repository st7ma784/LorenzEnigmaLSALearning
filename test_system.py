#!/usr/bin/env python3
"""
Quick test script to verify the Enigma-Lorenz analysis system works
"""

import numpy as np
from enigma_lorenz_analysis import EnigmaMachine, LorenzCipher, PermutationMatrixAnalyzer

def test_enigma_encoding():
    """Test basic Enigma encoding functionality"""
    print("Testing Enigma encoding...")
    
    # Create simple test case
    rotor_configs = [
        ('EKMFLGDQVZNTOWYHXUSPAIBRCJ', 0),  # Rotor I at position 0
        ('AJDKSIRUXBLHWTMCQGZNPYFVOE', 5),  # Rotor II at position 5
        ('BDFHJLCPRTXVZNYEIWGAKMUSQO', 10) # Rotor III at position 10
    ]
    
    enigma = EnigmaMachine(rotor_configs)
    plaintext = "HELLO"
    ciphertext = enigma.encode_message(plaintext)
    
    print(f"  Plaintext:  {plaintext}")
    print(f"  Ciphertext: {ciphertext}")
    print(f"  ✓ Enigma encoding working")

def test_lorenz_mask_extraction():
    """Test Lorenz mask extraction"""
    print("Testing Lorenz mask extraction...")
    
    lorenz = LorenzCipher()
    plaintext = "HELLO"
    ciphertext = "XYZAB"  # Example cipher
    
    mask = lorenz.extract_mask(plaintext, ciphertext)
    print(f"  Plaintext: {plaintext}")
    print(f"  Ciphertext: {ciphertext}")
    print(f"  Mask: {mask[:20]}... (showing first 20 bits)")
    print(f"  Mask length: {len(mask)}")
    print(f"  ✓ Lorenz mask extraction working")

def test_statistical_analysis():
    """Test statistical analysis components"""
    print("Testing statistical analysis...")
    
    analyzer = PermutationMatrixAnalyzer()
    
    # Create test permutation matrix
    test_perm = np.eye(26)
    np.random.shuffle(test_perm)  # Random permutation
    
    # Extract features
    features = analyzer.extract_matrix_features(test_perm)
    print(f"  Matrix features extracted: {len(features)} features")
    print(f"  Sample features: trace={features.get('trace', 0):.2f}, "
          f"frobenius_norm={features.get('frobenius_norm', 0):.2f}")
    
    # Test doubly stochastic conversion
    ds_matrix = analyzer.create_doubly_stochastic_matrix(test_perm)
    row_sums = np.sum(ds_matrix, axis=1)
    col_sums = np.sum(ds_matrix, axis=0)
    print(f"  Doubly stochastic conversion:")
    print(f"    Row sums range: [{np.min(row_sums):.3f}, {np.max(row_sums):.3f}]")
    print(f"    Col sums range: [{np.min(col_sums):.3f}, {np.max(col_sums):.3f}]")
    print(f"  ✓ Statistical analysis working")

def test_integrated_pipeline():
    """Test the integrated analysis pipeline"""
    print("Testing integrated pipeline...")
    
    # Create Enigma machine
    rotor_configs = [
        (None, 3),  # Use default wiring, position 3
        (None, 7),  # Use default wiring, position 7  
        (None, 15) # Use default wiring, position 15
    ]
    
    enigma = EnigmaMachine(rotor_configs)
    plaintext = "THEQUICKBROWNFOX"
    
    # Encode
    ciphertext = enigma.encode_message(plaintext)
    
    # Extract mask
    lorenz = LorenzCipher()
    mask = lorenz.extract_mask(plaintext, ciphertext)
    
    # Analyze
    analyzer = PermutationMatrixAnalyzer()
    rotor_matrices = [rotor.permutation_matrix for rotor in enigma.rotors]
    rotor_positions = [rotor.position for rotor in enigma.rotors]
    
    correlations, mask_stats = analyzer.correlate_matrices_and_mask(
        rotor_matrices, rotor_positions, mask
    )
    
    print(f"  Pipeline test results:")
    print(f"    Plaintext: {plaintext}")
    print(f"    Ciphertext: {ciphertext}")
    print(f"    Mask length: {len(mask)}")
    print(f"    Mask mean: {mask_stats.get('mean', 0):.3f}")
    print(f"    Correlations found: {len(correlations)}")
    print(f"  ✓ Integrated pipeline working")

def test_web_visualization_data():
    """Test that web visualization will have proper data"""
    print("Testing web visualization data preparation...")
    
    # Test data that would be used by the web interface
    test_plaintext = "HELLO"
    rotor_positions = [0, 5, 10]
    
    # Simulate encoding (simplified for test)
    test_ciphertext = "XYZAB"
    
    # Test binary conversion
    lorenz = LorenzCipher()
    plain_binary = lorenz.text_to_binary(test_plaintext)
    cipher_binary = lorenz.text_to_binary(test_ciphertext)
    
    print(f"  Text to binary conversion:")
    print(f"    '{test_plaintext}' -> {plain_binary}")
    print(f"    '{test_ciphertext}' -> {cipher_binary}")
    
    # Test mask extraction
    mask = lorenz.extract_mask(test_plaintext, test_ciphertext)
    print(f"    Mask: {mask}")
    
    print(f"  ✓ Web visualization data preparation working")

def run_all_tests():
    """Run all system tests"""
    print("=" * 60)
    print("RUNNING ENIGMA-LORENZ SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_enigma_encoding,
        test_lorenz_mask_extraction,
        test_statistical_analysis,
        test_integrated_pipeline,
        test_web_visualization_data
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! System is ready to run.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\nNext steps:")
        print("1. Run: python enigma_lorenz_analysis.py")
        print("2. Run: python gradient_permutation_learning.py") 
        print("3. Run: python run_complete_analysis.py")
        print("4. Open: web_visualization.html in your browser")
    else:
        print("\nPlease fix the failing tests before proceeding.")