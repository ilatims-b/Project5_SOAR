# Baselines

## Selective Context for Prompt Compression

This section describes an adaptation of the LongBench evaluation framework to incorporate prompt compression using the Selective Context method. The original code is forked from [THUDM/LongBench](https://github.com/THUDM/LongBench) and the Selective Context implementation is adapted from [liyucheng09/Selective_Context](https://github.com/liyucheng09/Selective_Context).

**Core Idea:**

The primary goal is to evaluate the performance of language models on long-context tasks when the context is compressed *before* being fed to the model. `Selective Context` is a technique that identifies and removes less important parts of the context, aiming to reduce the computational load and context length while preserving key information.

**Two-Step Benchmark Workflow:**

The evaluation process has been refactored into a two-step pipeline.

1.  **Step 1: Compression**: The entire `THUDM/LongBench-v2` dataset is compressed once and saved to disk. This step uses the `Selective Context` method.
2.  **Step 2: Prediction**: The benchmark is run using the pre-compressed dataset.

This approach allows for multiple experiments on the same compressed data without re-running the costly compression step each time.

**File Descriptions:**

*   `compress.py`: Compresses the `THUDM/LongBench-v2` dataset.
*   `pred_compress.py`: Runs the LongBench benchmark on a compressed dataset.
*   `compressor/`: Contains the Selective Context implementation.

## Setup & How to Run

Due to dependency conflicts, you will need two separate Conda environments.

1.  **Environment for vLLM:**
    ```bash
    conda env create -f vllm.yml
    ```

2.  **Environment for Selective Context:**
    ```bash
    conda env create -f selective_context.yml
    python -m spacy download en_core_web_sm
    ```

**How to Run:**

The benchmark process requires two separate shell sessions because the model server and the evaluation script use conflicting dependencies.

**Step 1: Compress the Dataset**

In your first terminal, activate the `selective_context` environment and run the compression script. Using `--dryrun` is recommended for the first time to ensure everything is set up correctly.

```bash
conda activate selective_context
python compress.py \
    --save_dir results \
    --compressor_model_type microsoft/Phi-3-mini-128k-instruct \
    --dryrun 5
```
*Remove `--dryrun 5` to process the entire dataset.*

**Step 2: Serve the Language Model**

In a second, separate terminal, activate the `vllm` environment and start the vLLM server. This server will handle requests from the benchmark script. Make sure the model name matches what you intend to evaluate.

```bash
conda activate vllm
vllm serve Qwen/Qwen2-0.5B-Instruct --port 8000 --api-key token-abc123
```
*This process will occupy this terminal. Leave it running.*

**Step 3: Run the Benchmark**

Return to your first terminal (with the `selective_context` environment still active) and run the benchmark script. It will connect to the vLLM server you just started.

```bash
python pred_compress.py \
    --model Qwen/Qwen2-0.5B-Instruct \
    --compressed_data_dir results \
    --n_proc 1
```
*Make sure the `--model` argument here matches the model served in Step 2.*
