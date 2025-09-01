# Token Influence Analysis with Statistical Guarantees

A memory-optimized implementation for analyzing token influence in Large Language Models using stochastic dropout, empirical null distributions, and statistical significance testing. This project uses the MS MARCO dataset to provide statistically rigorous token annotations with False Discovery Rate (FDR) control.

## Overview

This system implements a comprehensive three-phase approach for identifying and labeling statistically significant tokens that influence model predictions:

- **Phase 1**: Stochastic token influence analysis using dropout-based masking
- **Phase 2**: Empirical null distribution calculation from punctuation tokens
- **Phase 3**: Statistical analysis with FDR-controlled labeling (TC/TH/TR classification)

## Features

- **Memory-Optimized**: GPU memory management with configurable batch sizes and sequential processing
- **Statistical Rigor**: Empirical null distributions with shrunken Z-scores and FDR control
- **Flexible Positional Analysis**: Configurable positional binning (first 10%, middle 80%, last 10%) or uniform analysis
- **MS MARCO Integration**: Optimized dataset filtering and sampling by query types
- **Comprehensive Output**: Per-token attributions with detailed statistical metrics
- **Production Ready**: Error handling, progress tracking, and reproducible results

## Requirements

```bash
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0
spacy>=3.7.0
scipy>=1.11.0
```

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration

Edit `config.py` to customize the analysis:

### Core Settings
```python
RANDOM_SEED = 42                    # Reproducibility seed
QUERY_TYPES = ["NUMERIC"]           # MS MARCO query types to analyze
TOTAL_EXAMPLES = 1                  # Examples to process per query type
NULL_EXAMPLES = 1                   # Examples for null distribution

MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"  # HuggingFace model
NUM_STOCHASTIC_REPEATS = 3          # Stochastic repeats per token
```

### Key Configuration Options

**Positional Binning Mode:**
```python
USE_POSITIONAL_BINNING = True       # Enable/disable positional analysis
```
- `True`: Analyzes tokens in 3 bins (first 10%, middle 80%, last 10%)
- `False`: Treats all punctuation equally across entire prompt

**Statistical Parameters:**
```python
Q_CRITICAL = 0.15                   # FDR level for critical spans
Q_HARMFUL = 0.05                    # FDR level for harmful spans
```

### Supported Query Types
- `NUMERIC`: Numerical answer questions
- `ENTITY`: Named entity questions  
- `DESCRIPTION`: Descriptive questions
- `PERSON`: Person-related questions
- `LOCATION`: Location-based questions

## Usage

### Basic Analysis
```bash
python msmarco_analysis.py
```

### Validate Configuration
```bash
python config.py
```

## Output Files

The analysis generates three key output files:

1. **`null_distribution_stats.json`**: Phase 2 null distribution statistics
2. **`msmarco_phase3_results.json`**: Complete analysis results with span annotations
3. **`token_attributions_detailed.json`**: Per-token attributions with all metrics

### Token Attribution Format
```json
{
  "example_id": 0,
  "query": "What is...",
  "token": "word",
  "attribution": 0.123,
  "std_deviation": 0.045,
  "class": "critical|harmful|redundant",
  "shrunken_z_score": 2.34,
  "positional_bin": "first_10|middle_80|last_10|all_positions",
  "token_idx": 5,
  "span_info": {...}
}
```

## Token Classification

- **TC (Critical)**: Tokens significantly increasing prediction confidence
- **TH (Harmful)**: Tokens significantly decreasing prediction confidence  
- **TR (Redundant)**: Tokens with no significant statistical impact

## Memory Management

The system is optimized for GPU memory efficiency:
- Sequential token processing
- Automatic cache clearing
- Configurable batch sizes
- Error recovery with memory cleanup

## Architecture

## Performance

### Optimizations Implemented
- **Dataset Filtering**: Efficient query-type-specific filtering before sampling
- **Sequential Processing**: Memory-efficient token-by-token analysis
- **GPU Memory Management**: Automatic cache clearing and garbage collection
- **Batch Processing**: Configurable batch sizes for different hardware

### Expected Performance
- **Memory Usage**: ~2-4GB GPU memory for Phi-3-mini
- **Processing Speed**: ~1-2 minutes per example (depends on prompt length)
- **Scalability**: Linear scaling with number of examples

## Algorithm Details

### Phase 1: Token Influence Analysis
1. Tokenize input prompt
2. Establish baseline prediction with full prompt
3. For each token, create masked version (token removal)
4. Calculate influence as difference in log-odds
5. Repeat process with stochastic dropout for uncertainty estimation

### Phase 2: Null Distribution
1. Collect punctuation tokens from sample prompts
2. Calculate token influences for punctuation (assumed neutral)
3. Build empirical null distribution by positional bins or uniformly
4. Compute mean, std, and percentile statistics

### Phase 3: Statistical Classification
1. Calculate shrunken Z-scores using null distribution
2. Group tokens into linguistic spans via spaCy NER
3. Apply Benjamini-Hochberg FDR control
4. Assign final labels: TC (Critical), TH (Harmful), TR (Redundant)

## Troubleshooting

### Common Issues
- **CUDA OOM**: Reduce `BATCH_SIZE` or `TOTAL_EXAMPLES`
- **spaCy Missing**: Run `python -m spacy download en_core_web_sm`
- **Dataset Loading**: Check internet connection for HuggingFace datasets
- **Empty Results**: Verify query types match available data

### Debug Mode
Set small values for testing:
```python
TOTAL_EXAMPLES = 1
NULL_EXAMPLES = 1
NUM_STOCHASTIC_REPEATS = 2
```

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Configure analysis** in `config.py`:
```python
TOTAL_EXAMPLES = 5      # Start small for testing
NULL_EXAMPLES = 10      # Sufficient for null distribution
USE_POSITIONAL_BINNING = True  # Enable positional analysis
```

3. **Run analysis:**
```bash
python msmarco_analysis.py
```

4. **Check outputs:**
- `null_distribution_stats.json` - Null distribution statistics
- `msmarco_phase3_results.json` - Complete analysis results  
- `token_attributions_detailed.json` - Per-token attributions

## Example Output

### Null Distribution Statistics
```json
{
  "first_10": {
    "mean": -0.000123,
    "std": 0.045678,
    "count": 150,
    "percentiles": {"95": 0.089234, "99": 0.156789}
  }
}
```

### Token Classification Results
```json
{
  "token": "important",
  "attribution": 0.234,
  "class": "critical",
  "shrunken_z_score": 2.45,
  "positional_bin": "middle_80"
}
```

## Methodology

### Phase 1: Stochastic Token Influence
1. **Baseline Calculation**: Compute model's prediction probability for the original prompt
2. **Token Masking**: Remove each token individually and measure prediction change  
3. **Stochastic Sampling**: Use dropout with multiple repeats to get robust estimates
4. **Influence Metric**: Calculate log-odds difference between masked and baseline

### Phase 2: Empirical Null Distribution
1. **Punctuation Identification**: Find all punctuation tokens in prompts
2. **Positional Binning** (if enabled): Categorize tokens into bins:
   - First 10% of prompt (sentence beginnings)
   - Middle 80% of prompt (main content)
   - Last 10% of prompt (sentence endings)
3. **Statistical Calculation**: Compute mean, standard deviation, and percentiles for each bin

### Phase 3: Statistical Classification
1. **Shrunken Z-scores**: Apply James-Stein-like shrinkage using null distribution
2. **Linguistic Spans**: Group tokens into meaningful units via spaCy NER
3. **FDR Control**: Apply Benjamini-Hochberg procedure for multiple testing correction
4. **Final Labeling**: Assign TC/TH/TR labels based on statistical significance

## Citation

If you use this code in your research, please cite:

```bibtex
@software{token_influence_analysis,
  title={Token Influence Analysis with Statistical Guarantees},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/token-influence-analysis}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Microsoft for the MS MARCO dataset
- HuggingFace for model hosting and transformers library
- spaCy team for linguistic processing tools
- The Phi-3 team for the efficient language model
