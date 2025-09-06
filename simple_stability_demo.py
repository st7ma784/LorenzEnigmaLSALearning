#!/usr/bin/env python3
"""
Simple Stability Demonstration for Permutation Matrix Learning
Pure Python implementation showing key stability concepts
"""

import random
import math

class SimpleStabilityDemo:
    """Demonstrates key stability concepts in permutation learning"""
    
    def __init__(self):
        self.stable_config = {
            'learning_rate': 0.001,
            'gradient_clip': 0.5,
            'temperature': 1.0,
            'regularization': 1.0,
            'noise_std': 0.01
        }
        
        self.unstable_config = {
            'learning_rate': 0.05,    # Too high
            'gradient_clip': 10.0,    # No clipping
            'temperature': 0.1,       # Too low
            'regularization': 0.1,    # Too weak
            'noise_std': 0.1          # Too much noise
        }
    
    def simulate_training_step(self, config, epoch, current_loss):
        """Simulate one training step with given configuration"""
        
        # Simulate gradient computation
        base_gradient = random.uniform(0.5, 2.0)
        
        # Learning rate effect
        lr_factor = config['learning_rate']
        update_magnitude = base_gradient * lr_factor
        
        # Temperature effect on gradient stability
        temp_factor = 1.0 / max(config['temperature'], 0.1)
        temp_instability = temp_factor * random.uniform(0, 0.5)
        
        # Gradient clipping effect
        if update_magnitude > config['gradient_clip']:
            update_magnitude = config['gradient_clip']
            clipped = True
        else:
            clipped = False
        
        # Regularization effect
        reg_penalty = config['regularization'] * random.uniform(0.1, 0.3)
        
        # Noise effect
        noise = random.gauss(0, config['noise_std'])
        
        # Calculate new loss
        loss_change = -update_magnitude + temp_instability + noise
        new_loss = max(current_loss + loss_change, 0.001)  # Prevent negative loss
        
        # Calculate stability indicators
        gradient_explosion = update_magnitude > 1.0
        temperature_collapse = config['temperature'] < 0.2
        high_learning_rate = config['learning_rate'] > 0.01
        
        return {
            'loss': new_loss,
            'gradient_magnitude': base_gradient,
            'update_magnitude': update_magnitude,
            'clipped': clipped,
            'gradient_explosion': gradient_explosion,
            'temperature_collapse': temperature_collapse,
            'high_learning_rate': high_learning_rate,
            'stability_score': 1.0 / (1.0 + abs(loss_change))
        }
    
    def run_training_simulation(self, config, epochs=50, config_name=""):
        """Run a complete training simulation"""
        results = []
        current_loss = 1.0  # Starting loss
        
        print(f"\n{config_name} Training:")
        print("-" * 40)
        
        for epoch in range(epochs):
            result = self.simulate_training_step(config, epoch, current_loss)
            current_loss = result['loss']
            results.append(result)
            
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                warnings = []
                if result['gradient_explosion']:
                    warnings.append("GRAD_EXPLOSION")
                if result['temperature_collapse']:
                    warnings.append("TEMP_COLLAPSE")
                if result['high_learning_rate']:
                    warnings.append("HIGH_LR")
                
                warning_str = " | " + ", ".join(warnings) if warnings else ""
                
                print(f"Epoch {epoch:2d}: Loss={current_loss:.4f}, "
                      f"Grad={result['gradient_magnitude']:.3f}, "
                      f"Stability={result['stability_score']:.3f}{warning_str}")
        
        return results
    
    def analyze_results(self, stable_results, unstable_results):
        """Analyze and compare results"""
        print("\n" + "=" * 60)
        print("STABILITY ANALYSIS RESULTS")
        print("=" * 60)
        
        # Calculate averages
        stable_final_loss = stable_results[-1]['loss']
        unstable_final_loss = unstable_results[-1]['loss']
        
        stable_avg_stability = sum(r['stability_score'] for r in stable_results[-10:]) / 10
        unstable_avg_stability = sum(r['stability_score'] for r in unstable_results[-10:]) / 10
        
        stable_explosions = sum(1 for r in stable_results if r['gradient_explosion'])
        unstable_explosions = sum(1 for r in unstable_results if r['gradient_explosion'])
        
        stable_clips = sum(1 for r in stable_results if r['clipped'])
        unstable_clips = sum(1 for r in unstable_results if r['clipped'])
        
        # Print comparison
        print(f"FINAL RESULTS:")
        print(f"  Stable Model Final Loss:    {stable_final_loss:.6f}")
        print(f"  Unstable Model Final Loss:  {unstable_final_loss:.6f}")
        print(f"  Loss Improvement Ratio:     {unstable_final_loss / stable_final_loss:.2f}x")
        
        print(f"\nSTABILITY METRICS:")
        print(f"  Stable Avg Stability Score:   {stable_avg_stability:.4f}")
        print(f"  Unstable Avg Stability Score: {unstable_avg_stability:.4f}")
        print(f"  Stability Improvement:        {stable_avg_stability / unstable_avg_stability:.2f}x")
        
        print(f"\nINSTABILITY EVENTS:")
        print(f"  Stable Gradient Explosions:   {stable_explosions}")
        print(f"  Unstable Gradient Explosions: {unstable_explosions}")
        print(f"  Stable Gradient Clips:        {stable_clips}")
        print(f"  Unstable Gradient Clips:      {unstable_clips}")
        
        # Success determination
        stable_success = stable_final_loss < 0.1 and stable_explosions < 5
        unstable_success = unstable_final_loss < 0.1 and unstable_explosions < 5
        
        print(f"\nTRAINING SUCCESS:")
        print(f"  Stable Configuration:   {'✅ SUCCESS' if stable_success else '❌ FAILED'}")
        print(f"  Unstable Configuration: {'✅ SUCCESS' if unstable_success else '❌ FAILED'}")
        
        return stable_success, unstable_success
    
    def demonstrate_techniques(self):
        """Demonstrate key stability techniques"""
        print("\n" + "=" * 60)
        print("STABILITY TECHNIQUES DEMONSTRATION")
        print("=" * 60)
        
        techniques = {
            "Gradient Clipping": {
                "problem": "Gradients can explode, causing unstable updates",
                "solution": "Clip gradients to maximum norm (e.g., 0.5-1.0)",
                "example": "grad = grad * (clip_norm / max(grad_norm, clip_norm))"
            },
            
            "Learning Rate Scheduling": {
                "problem": "High learning rates cause oscillations",
                "solution": "Start with lower rates, anneal during training",
                "example": "lr = initial_lr * (decay_rate ** epoch)"
            },
            
            "Temperature Annealing": {
                "problem": "Fixed temperature doesn't balance exploration/exploitation",
                "solution": "Start hot (soft assignments), cool down gradually",
                "example": "temp = max(temp * 0.99, min_temp)"
            },
            
            "Regularization": {
                "problem": "Permutations can degenerate or become non-doubly-stochastic",
                "solution": "Add penalties for constraint violations",
                "example": "loss += λ * (row_sum_penalty + col_sum_penalty)"
            },
            
            "EMA (Exponential Moving Average)": {
                "problem": "Parameters can jump around unstably",
                "solution": "Smooth parameter updates using moving averages",
                "example": "ema_params = 0.99 * ema_params + 0.01 * current_params"
            },
            
            "Noise Regularization": {
                "problem": "Model can overfit to specific permutation patterns",
                "solution": "Add small noise to prevent overfitting",
                "example": "logits += noise ~ N(0, small_std)"
            }
        }
        
        for i, (technique, details) in enumerate(techniques.items(), 1):
            print(f"{i}. {technique}")
            print(f"   Problem:  {details['problem']}")
            print(f"   Solution: {details['solution']}")
            print(f"   Example:  {details['example']}")
            print()
    
    def demonstrate_sinkhorn_stability(self):
        """Demonstrate Sinkhorn normalization stability issues"""
        print("\n" + "=" * 60)
        print("SINKHORN NORMALIZATION STABILITY")
        print("=" * 60)
        
        print("Sinkhorn Normalization converts any matrix to doubly-stochastic:")
        print("(All row sums = 1, all column sums = 1)")
        print()
        
        # Demonstrate with simple 3x3 example
        print("Example 3x3 Matrix Transformation:")
        print()
        
        # Simulate matrix values
        original = [[0.8, 0.1, 0.2], [0.3, 0.9, 0.1], [0.2, 0.4, 0.7]]
        
        print("Original Matrix:")
        for row in original:
            print(f"  {[f'{x:.2f}' for x in row]}")
        
        # Show row/column sums
        row_sums = [sum(row) for row in original]
        col_sums = [sum(original[i][j] for i in range(3)) for j in range(3)]
        
        print(f"Row sums: {[f'{x:.2f}' for x in row_sums]}")
        print(f"Col sums: {[f'{x:.2f}' for x in col_sums]}")
        print()
        
        print("After Sinkhorn Normalization (simulated):")
        # Simulate normalized result
        normalized = [[0.42, 0.08, 0.50], [0.25, 0.60, 0.15], [0.33, 0.32, 0.35]]
        for row in normalized:
            print(f"  {[f'{x:.2f}' for x in row]}")
        
        norm_row_sums = [sum(row) for row in normalized]
        norm_col_sums = [sum(normalized[i][j] for i in range(3)) for j in range(3)]
        
        print(f"Row sums: {[f'{x:.2f}' for x in norm_row_sums]}")
        print(f"Col sums: {[f'{x:.2f}' for x in norm_col_sums]}")
        print()
        
        print("Key Stability Issues:")
        print("1. High temperatures → too soft, slow convergence")
        print("2. Low temperatures → numerical instability, hard assignments")
        print("3. Many iterations → accumulated numerical errors")
        print("4. No convergence check → wasted computation")
        print()
        
        print("Stability Solutions:")
        print("• Temperature annealing: start high, gradually reduce")
        print("• Early stopping: monitor convergence, stop when stable")
        print("• Numerical clipping: prevent division by near-zero values")
        print("• Gradient scaling: control how much soft→hard conversion affects gradients")

def main():
    """Run the complete stability demonstration"""
    print("🔬 PERMUTATION MATRIX LEARNING STABILITY DEMONSTRATION")
    print("Testing different hyperparameter configurations...")
    
    demo = SimpleStabilityDemo()
    
    # Run both configurations
    print("\n🟢 Running STABLE configuration...")
    stable_results = demo.run_training_simulation(
        demo.stable_config, epochs=50, config_name="STABLE"
    )
    
    print("\n🔴 Running UNSTABLE configuration...")
    unstable_results = demo.run_training_simulation(
        demo.unstable_config, epochs=50, config_name="UNSTABLE"
    )
    
    # Analyze results
    stable_success, unstable_success = demo.analyze_results(stable_results, unstable_results)
    
    # Show techniques
    demo.demonstrate_techniques()
    
    # Show Sinkhorn details
    demo.demonstrate_sinkhorn_stability()
    
    # Final recommendations
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED STABLE CONFIGURATION")
    print("=" * 60)
    print("For stable permutation matrix learning:")
    print()
    print("✅ Learning Rate: 0.0005 - 0.001 (low)")
    print("✅ Gradient Clipping: 0.5 - 1.0")
    print("✅ Temperature: Start at 1.0-2.0, anneal to 0.1-0.3")
    print("✅ Regularization: 1.0 - 2.0 (strong constraints)")
    print("✅ Noise: 0.01 - 0.05 (small)")
    print("✅ EMA Decay: 0.99 - 0.999")
    print()
    print("⚠️  AVOID:")
    print("❌ Learning rates > 0.01")
    print("❌ No gradient clipping")
    print("❌ Fixed low temperature < 0.2")
    print("❌ Weak regularization < 0.5")
    print("❌ High noise > 0.1")
    
    if stable_success and not unstable_success:
        print("\n🎉 DEMONSTRATION SUCCESSFUL!")
        print("Stable configuration significantly outperformed unstable configuration.")
    elif stable_success and unstable_success:
        print("\n✅ Both configurations worked, but stable was more reliable.")
    else:
        print("\n⚠️  Both configurations had issues - may need further tuning.")
    
    print("\n📊 Check the web visualization for interactive parameter exploration!")

if __name__ == "__main__":
    main()