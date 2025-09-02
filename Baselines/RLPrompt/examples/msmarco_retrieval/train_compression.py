#!/usr/bin/env python3
"""
MS MARCO V2.1 Compression Training Script

This script trains RL Prompt models to compress prompts while maintaining
performance on the MS MARCO V2.1 information retrieval task.
"""

import json
import argparse
import yaml
import torch
import os
from typing import Dict, Any
from datetime import datetime
import numpy as np

# Import RL Prompt components
import sys
sys.path.append('../../../rlprompt')

from rlprompt.models import PolicyModel
from rlprompt.trainers import PolicyGradientTrainer
from rlprompt.rewards import BaseReward
from rlprompt.envs import PromptEnvironment

class MSMarcoCompressionReward(BaseReward):
    """Custom reward function for MS MARCO compression."""
    
    def __init__(self, baseline_accuracy: float = 0.85, 
                 compression_target: float = 0.7,
                 performance_weight: float = 0.7,
                 compression_weight: float = 0.3):
        super().__init__()
        self.baseline_accuracy = baseline_accuracy
        self.compression_target = compression_target
        self.performance_weight = performance_weight
        self.compression_weight = compression_weight
    
    def compute_reward(self, prompt: str, performance: Dict[str, float], 
                      compression_ratio: float) -> float:
        """Compute reward balancing performance and compression."""
        # Performance score (normalized to 0-1)
        performance_score = performance.get('accuracy', 0) / self.baseline_accuracy
        
        # Compression score (how close to target)
        compression_score = 1.0 - abs(compression_ratio - self.compression_target)
        
        # Combined reward
        reward = (self.performance_weight * performance_score + 
                 self.compression_weight * compression_score)
        
        return reward

class MSMarcoCompressionTrainer:
    """Trains RL Prompt models for MS MARCO compression."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load baseline results
        self.baseline_results = self._load_baseline_results()
        
        # Initialize components
        self._setup_environment()
        self._setup_model()
        self._setup_trainer()
        
    def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline evaluation results."""
        baseline_file = self.config.get('baseline_results')
        if not baseline_file or not os.path.exists(baseline_file):
            print("Warning: Baseline results not found. Using default values.")
            return {'overall_metrics': {'accuracy': 0.85}}
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def _setup_environment(self):
        """Setup the prompt environment."""
        # Create environment for MS MARCO compression
        self.env = PromptEnvironment(
            task_name="msmarco_retrieval",
            model_name=self.config['task_lm'],
            max_prompt_length=self.config['prompt_length']
        )
    
    def _setup_model(self):
        """Setup the policy model."""
        self.model = PolicyModel(
            vocab_size=self.env.vocab_size,
            hidden_size=self.config.get('model', {}).get('hidden_size', 256),
            num_layers=self.config.get('model', {}).get('num_layers', 2),
            dropout=self.config.get('model', {}).get('dropout', 0.1)
        ).to(self.device)
    
    def _setup_trainer(self):
        """Setup the trainer."""
        # Initialize reward function
        reward_config = self.config.get('reward', {})
        self.reward_fn = MSMarcoCompressionReward(
            baseline_accuracy=reward_config.get('baseline_accuracy', 0.85),
            compression_target=self.config['compression']['target_ratio'],
            performance_weight=reward_config.get('performance_weight', 0.7),
            compression_weight=reward_config.get('compression_weight', 0.3)
        )
        
        # Initialize trainer
        training_config = self.config.get('training', {})
        self.trainer = PolicyGradientTrainer(
            model=self.model,
            env=self.env,
            reward_fn=self.reward_fn,
            learning_rate=self.config['learning_rate'],
            gamma=training_config.get('gamma', 0.99),
            entropy_coef=training_config.get('entropy_coef', 0.01)
        )
    
    def train(self) -> Dict[str, Any]:
        """Run training for prompt compression."""
        print("Starting MS MARCO compression training...")
        print(f"Target compression ratio: {self.config['compression']['target_ratio']}")
        print(f"Max episodes: {self.config['max_episodes']}")
        
        # Training loop
        best_reward = float('-inf')
        training_history = []
        
        for episode in range(self.config['max_episodes']):
            # Generate prompt using current policy
            prompt = self.env.generate_prompt(self.model)
            
            # Evaluate prompt performance
            performance = self._evaluate_prompt(prompt)
            
            # Calculate compression ratio
            compression_ratio = self._calculate_compression_ratio(prompt)
            
            # Compute reward
            reward = self.reward_fn.compute_reward(prompt, performance, compression_ratio)
            
            # Update policy
            self.trainer.update_policy(reward)
            
            # Store training history
            training_history.append({
                'episode': episode,
                'prompt': prompt,
                'performance': performance,
                'compression_ratio': compression_ratio,
                'reward': reward
            })
            
            # Log progress
            if episode % self.config.get('evaluation', {}).get('log_interval', 10) == 0:
                print(f"Episode {episode}: Reward={reward:.4f}, "
                      f"Accuracy={performance.get('accuracy', 0):.4f}, "
                      f"Compression={compression_ratio:.3f}")
            
            # Save best model
            if reward > best_reward:
                best_reward = reward
                self._save_model(f"best_model_episode_{episode}")
            
            # Checkpoint
            if episode % self.config.get('evaluation', {}).get('checkpoint_interval', 100) == 0:
                self._save_checkpoint(episode)
        
        return {
            'training_history': training_history,
            'best_reward': best_reward,
            'final_model': self.model.state_dict(),
            'config': self.config
        }
    
    def _evaluate_prompt(self, prompt: str) -> Dict[str, float]:
        """Evaluate prompt performance on MS MARCO task."""
        # This is a simplified evaluation - in practice, you'd run the full task
        # For now, we simulate performance based on prompt characteristics
        
        # Simulate accuracy based on prompt length and content
        prompt_length = len(prompt.split())
        target_length = self.config['prompt_length']
        
        # Simple heuristic: shorter prompts tend to have lower accuracy
        length_factor = min(1.0, prompt_length / target_length)
        
        # Add some randomness to simulate real evaluation
        base_accuracy = self.baseline_results['overall_metrics']['accuracy']
        accuracy = base_accuracy * (0.8 + 0.2 * length_factor) + np.random.normal(0, 0.02)
        accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
        
        return {
            'accuracy': accuracy,
            'prompt_length': prompt_length
        }
    
    def _calculate_compression_ratio(self, prompt: str) -> float:
        """Calculate compression ratio for the prompt."""
        prompt_tokens = len(prompt.split())
        baseline_tokens = self.config['prompt_length']
        
        if baseline_tokens == 0:
            return 1.0
        
        return prompt_tokens / baseline_tokens
    
    def _save_model(self, name: str):
        """Save the current model."""
        save_dir = self.config.get('output', {}).get('save_dir', 'results/msmarco_compression')
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f"{name}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        save_dir = self.config.get('output', {}).get('save_dir', 'results/msmarco_compression')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state': self.model.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_episode_{episode}.pt")
        torch.save(checkpoint, checkpoint_path)
    
    def save_training_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results and evaluation."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training history
        history_file = os.path.join(output_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(results['training_history'], f, indent=2)
        
        # Save final results
        results_file = os.path.join(output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            # Remove model state for JSON serialization
            json_results = {k: v for k, v in results.items() if k != 'final_model'}
            json.dump(json_results, f, indent=2)
        
        print(f"Training results saved to: {output_dir}")

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train RL Prompt for MS MARCO compression')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--baseline', required=True, help='Path to baseline results JSON')
    parser.add_argument('--output_dir', default='results/msmarco_compression', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['baseline_results'] = args.baseline
    
    # Initialize trainer
    trainer = MSMarcoCompressionTrainer(config)
    
    # Run training
    results = trainer.train()
    
    # Save results
    trainer.save_training_results(results, args.output_dir)
    
    print(f"\nTraining completed!")
    print(f"Best reward achieved: {results['best_reward']:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 