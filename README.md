# Token Compression for LLM

A system for improving the efficiency of long-context Large Language Model (LLM) inference by combining **prompt compression** and **KV-cache compression**.

## Overview

Long-context LLM inference requires increasing amounts of computation and GPU memory as the input context grows. This project explores a two-stage approach to reduce these requirements while maintaining model performance.

* **LLMLingua-2** is used before inference to compress prompts by removing redundant or low-information tokens.
* **SnapKV** is used during inference to reduce the size of the KV cache by retaining the most important cached representations.
* **4-bit quantization** is used to enable 7–8B parameter models to run within constrained GPU memory.

The system focuses on evaluating whether combining these techniques can make long-context inference more practical on a single GPU.

## Models

The project supports:

* Qwen2.5-7B-Instruct
* Llama-3-8B-Instruct

Both models are evaluated using 4-bit quantization.

## Methods

### LLMLingua-2

LLMLingua-2 performs pre-inference prompt compression to reduce the number of tokens passed to the main language model.

The project evaluates different compression levels, including:

* 2×
* 3×
* 5×

### SnapKV

SnapKV performs KV-cache compression during inference by identifying important KV positions using attention information and retaining a smaller cache.

The objective is to reduce GPU memory consumption and improve inference efficiency while preserving response quality.

## Evaluation

The system compares four configurations:

1. **Baseline** — No compression
2. **LLMLingua-2** — Prompt compression only
3. **SnapKV** — KV-cache compression only
4. **Combined** — LLMLingua-2 + SnapKV

The following metrics are used for comparison:

* Compression ratio
* Task accuracy
* Perplexity
* Time-to-First-Token (TTFT)
* Peak VRAM usage
* Generation throughput
* Generation latency

Evaluation is performed using long-context benchmarks such as **LongBench** and **LM-Eval-Harness**.

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* LLMLingua-2
* SnapKV
* BitsAndBytes
* Accelerate
* FastAPI
* Pydantic
* Server-Sent Events (SSE)

## Project Status

🚧 **Under Development**

The current project focuses on integrating LLMLingua-2 and SnapKV into a unified long-context inference pipeline and evaluating their individual and combined impact on memory usage, latency, throughput, and model performance.

## References

* LLMLingua — *Compressing Prompts for Accelerated Inference of Large Language Models*
* LongLLMLingua — *Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*
* LLMLingua-2 — *Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*
* SnapKV — *LLM Knows What You are Looking for Before Generation*
* TOVA — *Token Omission Via Attention*
* StreamingLLM — *Efficient Streaming Language Models with Attention Sinks*
* CacheGen — *KV Cache Compression and Streaming for Fast Large Language Model Serving*
* KIVI — *A Tuning-Free Asymmetric 2bit Quantization for KV Cache*
* PyramidKV — *Dynamic KV Cache Compression Based on Pyramidal Information Funneling*
* QUEST — *Query-Aware Sparsity for Efficient Long-Context LLM Inference*

## Acknowledgements

This project builds upon the research and open-source implementations of the methods referenced above.
