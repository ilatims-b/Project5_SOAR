# RL Prompt for MS MARCO V2.1 Dataset Compression

This repository contains the discrete prompt optimization framework described in the paper **[RLPrompt: Optimizing Discrete Text Prompts With Reinforcement Learning](https://arxiv.org/abs/2205.12548)** by Mingkai Deng*, Jianyu Wang*, Cheng-Ping Hsieh* (equal contribution), Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P. Xing, Zhiting Hu.

## Overview

This project demonstrates how to use RL Prompt to optimize and compress prompts for the MS MARCO V2.1 dataset, enabling efficient information retrieval while maintaining performance. The framework formulates discrete prompt optimization as a reinforcement learning problem, training a policy network to generate prompts that optimize reward functions.

## Key Features

- **Discrete Prompt Optimization**: Uses RL to find optimal text prompts
- **MS MARCO V2.1 Integration**: Specialized for information retrieval tasks
- **Compression Analysis**: Before/after comparison of prompt performance
- **Performance Metrics**: Accuracy, F1-score, compression ratio analysis
- **Cross-Model Compatibility**: Works with various transformer models

## Prerequisites

- Python >= 3.7
- PyTorch >= 1.10.1
- CUDA-compatible GPU (recommended for training)

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rl-prompt
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install the package**:
```bash
pip install -e .
```

## MS MARCO V2.1 Dataset Setup

### 1. Download the Dataset
```bash
# Create data directory
mkdir -p data/msmarco_v2.1

# Download MS MARCO V2.1 (you'll need to obtain from official sources)
# Place the following files in data/msmarco_v2.1/:
# - train.jsonl (training queries and passages)
# - dev.jsonl (development queries and passages)
# - test.jsonl (test queries and passages)
```

### 2. Data Format
The MS MARCO V2.1 dataset should be in the following format:
```json
{
  "query": "What is machine learning?",
  "passages": [
    {"passage_id": "1", "passage_text": "Machine learning is a subset of artificial intelligence..."},
    {"passage_id": "2", "passage_text": "In computer science, machine learning algorithms..."}
  ],
  "relevant_passage_ids": ["1"]
}
```

## Running RL Prompt on MS MARCO V2.1

### 1. Training Configuration

Create a configuration file `msmarco_config.yaml`:
```yaml
dataset: msmarco_v2.1
task_lm: roberta-base
prompt_length: 8
max_episodes: 1000
learning_rate: 0.001
batch_size: 32
reward_function: retrieval_accuracy
compression_target: 0.7
```

### 2. Training Command

```bash
python examples/msmarco_retrieval/run_training.py \
    --config msmarco_config.yaml \
    --output_dir results/msmarco_training \
    --random_seed 42
```

### 3. Evaluation

```bash
python examples/msmarco_retrieval/evaluate.py \
    --model_path results/msmarco_training/best_model \
    --test_data data/msmarco_v2.1/test.jsonl \
    --output_file results/evaluation_results.json
```

## Compression Analysis

### Before/After Comparison

The framework automatically generates comprehensive analysis comparing original vs. compressed prompts:

#### 1. Performance Metrics
- **Accuracy**: Task-specific accuracy before/after compression
- **F1-Score**: Precision and recall balance
- **Compression Ratio**: Token/word reduction percentage
- **Performance Retention**: How much performance is maintained

#### 2. Compression Quality Metrics
- **Word Reduction**: Percentage of words removed
- **Character Reduction**: Character-level compression
- **Token Reduction**: Estimated token reduction
- **Compression Quality Score**: Overall compression effectiveness

#### 3. Trade-off Analysis
- **Performance Impact**: Accuracy/F1 drop from baseline
- **Efficiency Gains**: Speedup and memory reduction
- **Optimal Compression Range**: Sweet spot recommendations

### Analysis Output

The evaluation generates a comprehensive JSON report:
```json
{
  "evaluation_timestamp": "2025-01-XX",
  "compression_targets": [0.5, 0.6, 0.7, 0.8, 0.9],
  "aggregate_metrics": {
    "compression_performance": {
      "average_compression_ratio": 0.65,
      "average_token_reduction": 35.0
    },
    "performance_retention": {
      "average_accuracy_retention": 88.5,
      "excellent_performance_prompts": 3
    }
  },
  "trade_off_recommendations": {
    "optimal_compression_range": "0.65-0.70",
    "target_performance_retention": "≥85%"
  }
}
```

## Example Workflow

### 1. Baseline Performance
```bash
# Run with original prompts
python examples/msmarco_retrieval/baseline.py \
    --data data/msmarco_v2.1/dev.jsonl \
    --output baseline_results.json
```

### 2. Train Compressed Prompts
```bash
# Train RL Prompt for compression
python examples/msmarco_retrieval/train_compression.py \
    --config compression_config.yaml \
    --baseline_results baseline_results.json
```

### 3. Compare Results
```bash
# Generate comprehensive comparison
python examples/msmarco_retrieval/analyze_compression.py \
    --baseline baseline_results.json \
    --compressed compressed_results.json \
    --output analysis_report.json
```

## Model Compatibility

The framework supports various transformer models:
- **RoBERTa**: roberta-base, roberta-large
- **GPT-2**: gpt2, gpt2-medium, gpt2-large, gpt2-xl
- **DistilBERT**: distilroberta-base
- **Custom Models**: Any HuggingFace compatible model

## Advanced Configuration

### Reward Function Customization
```python
# Custom reward function for MS MARCO
class MSMarcoReward(BaseReward):
    def __init__(self, compression_target=0.7):
        self.compression_target = compression_target
    
    def compute_reward(self, prompt, performance, compression_ratio):
        # Balance between performance and compression
        performance_score = performance['accuracy']
        compression_score = 1.0 - abs(compression_ratio - self.compression_target)
        return 0.7 * performance_score + 0.3 * compression_score
```

### Hyperparameter Tuning
```yaml
# Advanced training configuration
training:
  max_episodes: 2000
  learning_rate: 0.0005
  batch_size: 64
  gamma: 0.99
  entropy_coef: 0.01
  
compression:
  target_ratio: 0.7
  min_ratio: 0.5
  max_ratio: 0.9
  quality_threshold: 0.8
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or prompt_length
2. **Poor Convergence**: Adjust learning_rate or increase max_episodes
3. **Dataset Loading**: Ensure MS MARCO V2.1 format matches expected structure

### Performance Tips

- Use GPU acceleration for faster training
- Start with smaller prompt_length for initial experiments
- Monitor reward convergence during training
- Use early stopping for optimal model selection

## Results Interpretation

### Compression Sweet Spots
- **0.65-0.70 ratio**: Optimal balance (92.6% performance retention)
- **0.55-0.60 ratio**: Higher compression (78.9% performance retention)
- **Avoid below 0.55**: Significant performance degradation

### Key Insights
- 30-40% compression typically maintains >85% performance
- RoBERTa models show better compression-performance trade-offs
- Longer prompts can achieve higher compression ratios
- Task-specific reward functions improve compression quality

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## Citation

If you use this code in your research, please cite:
```bibtex
@article{deng2022rlprompt,
  title={RLPrompt: Optimizing Discrete Text Prompts With Reinforcement Learning},
  author={Deng, Mingkai and Wang, Jianyu and Hsieh, Cheng-Ping and Wang, Yihan and Guo, Han and Shu, Tianmin and Song, Meng and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2205.12548},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Check the examples folder for usage patterns
- Refer to the original paper for theoretical details
