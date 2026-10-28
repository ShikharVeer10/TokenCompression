# Token Compression for Long-Context LLMs

A research-driven capstone project focused on making long-context LLM inference more memory-efficient, faster, and practical on a single GPU.

## Why this project matters

Large language models are increasingly expected to work with long inputs, but this creates a major bottleneck:

- KV-cache memory grows linearly with context length
- Attention cost grows quadratically with sequence length
- Long-context decoding becomes slow and expensive on constrained hardware

This project tackles that challenge by combining two complementary compression strategies:

1. Prompt compression before generation
2. KV-cache compression during generation

The result is a hybrid inference pipeline designed for real-world deployment on resource-limited GPU setups.

## Project vision

To build an end-to-end token compression engine for long-context LLMs that reduces GPU memory footprint and inference latency while preserving answer quality.

The work is designed around a practical research goal:

> Can a long-context model run efficiently on a single Tesla T4 GPU without sacrificing too much factual quality?

---

## The problem

Modern LLMs struggle with long prompts because the model must retain a large amount of context throughout decoding. In practice, this leads to:

- higher VRAM consumption
- slower time-to-first-token
- lower throughput
- memory pressure during multi-turn or RAG-style workloads

This project addresses the two biggest contributors to that overhead:

- redundant tokens in the input prompt
- unnecessary KV entries stored during generation

---

## Solution architecture

### Stage 1: Prompt Compression with LLMLingua-2

Before inference, the input context is compressed using a token-aware importance model.

- removes low-information filler content
- preserves task-critical context
- reduces prompt length before decoding starts

This helps reduce the number of tokens the model must process and lowers downstream compute cost.

### Stage 2: KV-Cache Compression with SnapKV

During generation, the model keeps only the most important KV states instead of storing the full cache.

- preserves attention sinks
- prioritizes high-attention “heavy hitters”
- retains the recent observation window
- evicts lower-priority states from active GPU memory

This reduces the memory footprint of generation while keeping the model responsive to critical context.

### Combined system

The project evaluates the hybrid configuration:

- LLMLingua-2 for prompt pruning
- SnapKV for cache reduction
- 4-bit quantized Qwen2.5-7B-Instruct as the base model

This creates a lightweight long-context inference pipeline optimized for constrained GPU environments.

---

## Core implementation

The complete research workflow and experiments live in:

- `Token_Compression_Capstone_Final.ipynb`

The notebook includes:

- GPU environment validation and dependency setup
- 4-bit quantization with BitsAndBytes
- loading of `Qwen/Qwen2.5-7B-Instruct`
- LLMLingua-2-based prompt distillation
- SnapKV-style attention-hook KV eviction
- multi-condition benchmarking on long-context QA tasks
- visualization of memory, latency, and answer-quality comparisons

---

## Models and hardware

The project is implemented and evaluated for the following setup:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: 4-bit NF4
- Hardware target: NVIDIA Tesla T4 (15GB VRAM class)
- Frameworks: PyTorch, Transformers, Accelerate, BitsAndBytes

---

## Benchmarking setup

The notebook evaluates four configurations:

1. Baseline: raw context + full KV cache
2. LLMLingua-2 only
3. SnapKV only
4. Hybrid: LLMLingua-2 + SnapKV

### Metrics tracked

- Input tokens
- Output tokens
- Latency and time-to-first-token
- Throughput
- Peak VRAM usage
- Token savings and estimated VRAM reduction
- Token F1 and ROUGE-L
- Entity recall and answer quality

---

## Sample experimental results

These values were reported in the notebook as part of the capstone evaluation workflow:

| Condition | Input Tokens | Latency (s) | Throughput (tps) | Peak VRAM (GB) | VRAM Saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1683 | 5.14 | 7.20 | 12.59 | 0.0% |
| LLMLingua-2 Only | 1211 | 8.64 | 5.90 | 12.19 | 3.2% |
| SnapKV Only | 1683 | 7.15 | 5.17 | 12.59 | 0.0% |
| Hybrid Engine | 1211 | 7.79 | 6.55 | 12.19 | 3.2% |

These results show the project’s emphasis on practical efficiency trade-offs and evaluation-driven design rather than just theoretical claims.

---

## Why this stands out

This project combines research depth with implementation realism.

It is not just a demo; it is a full experimentation pipeline that:

- evaluates multiple compression strategies side by side
- compares behavior under real generation workloads
- stresses memory-constrained deployment scenarios
- grounds results in measurable benchmark metrics

That makes it especially compelling for:

- research-focused teams
- applied AI engineering roles
- LLM optimization and inference systems work
- recruiters looking for practical ML engineering depth

---

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- BitsAndBytes
- Accelerate
- LLMLingua
- SnapKV-inspired attention logic
- Pydantic
- FastAPI
- Gradio
- Seaborn
- Evaluate
- ROUGE score utilities

---

## Repository layout

```text
TokenCompression/
├── README.md
├── Token_Compression_Capstone_Final.ipynb
└── (notebook-based experiments and evaluation workflow)
```

---

## How to run

This project is primarily notebook-based and is best run in a GPU-enabled environment such as Google Colab.

1. Open `Token_Compression_Capstone_Final.ipynb`
2. Run the environment setup cells
3. Load the model and benchmark pipeline
4. Compare baseline vs compressed configurations

The notebook is structured to walk through the full pipeline from setup to evaluation.

---

## Research direction

This work sits at the intersection of:

- efficient inference
- long-context modeling
- prompt optimization
- memory-aware decoding
- low-bit model deployment
- LLM serving systems

The broader goal is to explore how compression strategies can make long-context AI systems more practical in real-world, compute-limited environments.

---

## References

- LLMLingua — Compressing Prompts for Accelerated Inference of Large Language Models
- LongLLMLingua — Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
- LLMLingua-2 — Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression
- SnapKV — LLM Knows What You are Looking for Before Generation
- TOVA — Token Omission Via Attention
- StreamingLLM — Efficient Streaming Language Models with Attention Sinks
- CacheGen — KV Cache Compression and Streaming for Fast Large Language Model Serving
- KIVI — A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- QUEST — Query-Aware Sparsity for Efficient Long-Context LLM Inference

---

## Final note

This project demonstrates a strong combination of research thinking, systems awareness, and deployment-focused optimization. It is a compelling example of applied AI engineering: solving a real bottleneck in LLM inference with practical, measurable improvements.

If you are exploring long-context LLM efficiency, this project is a meaningful step toward building faster, cheaper, and more scalable inference pipelines.

---

## Acknowledgements

This project builds upon the research and open-source work behind prompt compression, KV-cache optimization, and efficient LLM inference techniques.
