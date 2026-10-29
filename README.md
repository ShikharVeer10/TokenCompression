# TokenCompression

A notebook-based long-context inference prototype that combines **prompt compression** and **KV-cache eviction** to reduce the cost of running `Qwen/Qwen2.5-7B-Instruct` on a single GPU.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShikharVeer10/TokenCompression/blob/main/Token_Compression_Capstone_Final.ipynb)

## Overview

Long prompts increase both the amount of memory required for the KV cache and the work performed during generation. This project evaluates a two-stage inference pipeline:

1. **Prompt compression:** LLMLingua-2 removes lower-value prompt content before generation.
2. **KV-cache compression:** a SnapKV-inspired attention hook retains attention sinks, recent tokens, and selected heavy-hitter tokens during decoding.

The notebook is designed as an experimental capstone rather than a packaged Python library. All setup, model loading, inference, benchmarking, and visualization are contained in `Token_Compression_Capstone_Final.ipynb`.

## Features

- 4-bit NF4 loading of `Qwen/Qwen2.5-7B-Instruct` with BitsAndBytes
- LLMLingua-2 prompt compression with configurable retention rates
- Dynamic KV-cache eviction compatible with Hugging Face cache layouts
- Adaptive retention-rate selection based on context characteristics
- Protection for technical entities such as error codes, timestamps, identifiers, and node names
- Four-way comparison of raw, prompt-compressed, KV-compressed, and hybrid inference
- Token F1, ROUGE-L, entity recall, latency, throughput, and peak VRAM telemetry
- Gradio interface for interactive compression and generation
- Publication-style benchmark visualization saved as `capstone_evaluation_dashboard.png` when the visualization cell is run

## Architecture

```text
Raw context + query
        |
        v
Adaptive retention rate + critical-entity extraction
        |
        v
LLMLingua-2 prompt compression
        |
        v
Qwen/Qwen2.5-7B-Instruct (4-bit NF4)
        |
        v
Optional SnapKV-style KV-cache eviction
        |
        v
Generated answer + telemetry
```

### Prompt compression

The notebook uses:

```text
microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
```

The compressor can preserve structural punctuation and extracted technical entities while reporting original tokens, compressed tokens, token savings, and pruning percentage.

### KV-cache eviction

`SnapKVManager` installs forward hooks on the model’s self-attention layers. When a cache exceeds its configured capacity, it retains:

- an initial sink region (`sink_size`)
- top attention heavy hitters selected from the prefix
- the most recent observation window (`window_size`)

The default configuration is a 512-token capacity, 32-token observation window, 16-token sink region, and pooling kernel size 5.

## Benchmark modes

The notebook runs the same long-context QA workload under four conditions:

1. Baseline: raw context with the full KV cache
2. LLMLingua-2 only
3. SnapKV only
4. Hybrid engine: LLMLingua-2 plus SnapKV

Results are generated at runtime and displayed as a telemetry table. The benchmark reports input and output tokens, latency, throughput, peak VRAM, token F1, ROUGE-L, and target-entity recall. Results depend on the GPU, installed library versions, and runtime state, so the README does not hard-code a benchmark table.

## Interactive interface

The final notebook section launches a Gradio application with controls for:

- raw context and user query
- Stage 1 retention rate
- enabling or disabling SnapKV eviction
- KV-cache capacity

The interface returns the generated answer, token counts, pruning percentage, latency, and peak VRAM usage. In Colab, `demo.launch(share=True)` creates a temporary share URL.

## Requirements

- Python 3.10+ recommended
- NVIDIA GPU with CUDA support
- Approximately 15 GB of VRAM recommended for the target Tesla T4 setup
- Google Colab is the easiest way to run the notebook

The notebook installs the main dependencies, including:

- PyTorch
- Transformers
- Accelerate
- BitsAndBytes
- LLMLingua
- Pydantic
- Gradio
- Pandas, NumPy, and Seaborn
- ROUGE and evaluation utilities

## Run the project

1. Open the notebook in Google Colab using the badge above.
2. Select a GPU runtime.
3. Run the cells from top to bottom.
4. Allow the model and LLMLingua-2 checkpoint to download.
5. Review the Stage 1 sanity check, four-condition benchmark, visualization, and Gradio interface.

The notebook expects a CUDA device. CPU-only execution is not the supported configuration for the 7B model and compression pipeline.

## Repository structure

```text
TokenCompression/
├── README.md
└── Token_Compression_Capstone_Final.ipynb
```

## Scope and limitations

- This is an experimental notebook, not an installable package or command-line application.
- Benchmark outputs are machine- and run-dependent.
- SnapKV eviction is implemented as a project-specific attention hook inspired by the SnapKV approach; it is not the upstream SnapKV package.
- The notebook’s optional Git integration cell can push changes from Colab when a `GITHUB_TOKEN` secret is configured. Do not place tokens directly in the notebook or repository.

## References

- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968)
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
