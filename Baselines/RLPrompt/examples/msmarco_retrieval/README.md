# MS MARCO V2.1 Retrieval with RL Prompt

This example demonstrates how to use RL Prompt for compressing prompts in information retrieval tasks using the MS MARCO V2.1 dataset.

## Quick Start

### 1. Setup MS MARCO V2.1 Dataset
```bash
# Download and prepare the dataset
python prepare_msmarco.py --data_dir data/msmarco_v2.1
```

### 2. Run Baseline Evaluation
```bash
# Evaluate original prompts
python baseline_evaluation.py \
    --data data/msmarco_v2.1/dev.jsonl \
    --model roberta-base \
    --output baseline_results.json
```

### 3. Train Compressed Prompts
```bash
# Train RL Prompt for compression
python train_compression.py \
    --config compression_config.yaml \
    --baseline baseline_results.json \
    --output_dir results/compression_training
```

### 4. Analyze Compression Results
```bash
# Compare before/after performance
python analyze_compression.py \
    --baseline baseline_results.json \
    --compressed results/compression_training/eval_results.json \
    --output compression_analysis.json
```

## Configuration

### Compression Config (`compression_config.yaml`)
```yaml
dataset: msmarco_v2.1
task_lm: roberta-base
prompt_length: 8
max_episodes: 1000
learning_rate: 0.001
batch_size: 32

compression:
  target_ratio: 0.7
  min_ratio: 0.5
  max_ratio: 0.9
  quality_threshold: 0.8

reward:
  performance_weight: 0.7
  compression_weight: 0.3
  baseline_accuracy: 0.85
```

## Expected Output

The analysis generates comprehensive metrics:
- Compression ratios (0.5-0.9)
- Performance retention percentages
- Optimal compression sweet spots
- Trade-off recommendations

## Files

- `prepare_msmarco.py`: Dataset preparation
- `baseline_evaluation.py`: Original prompt evaluation
- `train_compression.py`: RL Prompt training
- `analyze_compression.py`: Compression analysis
- `compression_config.yaml`: Configuration file 