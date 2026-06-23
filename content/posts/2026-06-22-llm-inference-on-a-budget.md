---
title: "Running LLM Inference on a Budget"
date: 2026-06-22
description: "How to run large language models on consumer hardware using quantization, GGUF, and the right tooling choices."
eyebrow: "Engineering"
tags: [LLM, Inference, Quantization, MLOps]
slug: llm-inference-on-a-budget
draft: false
---

[TOC]

## Why this matters

Running a 70B parameter model used to require a cluster of A100s. Today you can run a quantized version on a laptop with 16 GB of RAM. This post walks through the practical choices that make that possible.

## The core problem

Full-precision (FP32) weights for a 7B model consume roughly 28 GB of memory. BF16 halves that to ~14 GB. INT8 takes it to ~7 GB. INT4 gets you to ~3.5 GB — small enough to fit on a mid-range GPU.

The tradeoff is accuracy. The art is knowing how much you can lose before it matters for your use case.

## Quantization formats in practice

### GGUF (llama.cpp)

GGUF is the format used by llama.cpp and Ollama. It supports mixed-precision — you can quantize most layers to Q4 while keeping attention layers at Q8, which recovers most of the accuracy lost by pure Q4.

```bash
# Pull and run a quantized model with Ollama
ollama pull llama3.2:3b-instruct-q4_K_M
ollama run llama3.2:3b-instruct-q4_K_M
```

The `Q4_K_M` suffix means 4-bit weights using the K-quant method, medium variant. K-quants group weights and quantize per-group, which significantly reduces the error compared to naive rounding.

### GPTQ

GPTQ is a post-training quantization method that minimizes the reconstruction error layer by layer using second-order information. It produces INT4 models that run well on NVIDIA GPUs via the `auto-gptq` library.

```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_triton=True,
)
```

### AWQ

AWQ (Activation-aware Weight Quantization) is similar to GPTQ but preserves the weights that matter most based on activation statistics. In practice it often outperforms GPTQ at Q4, especially for instruction-following tasks.

## Choosing a backend

| Backend | Format | Best for |
|---|---|---|
| llama.cpp / Ollama | GGUF | CPU + Apple Silicon, local dev |
| vLLM | GPTQ, AWQ, FP8 | GPU serving, high throughput |
| TGI | GPTQ, AWQ | Production APIs |
| ExLlamaV2 | EXL2 | Maximum throughput on NVIDIA |

## Memory estimation

A rough formula for VRAM needed:

```
VRAM (GB) ≈ (params_B × bits) / 8 × 1.2
```

The `1.2` factor accounts for KV cache and activation memory during inference. For a 7B model at Q4:

```
(7 × 4) / 8 × 1.2 = 4.2 GB
```

Comfortably fits on an RTX 3060 12 GB or an M2 MacBook with 16 GB unified memory.

## Practical tips

- **Start with Q4_K_M** in GGUF — it hits the sweet spot of size vs quality for most tasks.
- **Use Q8 for math/code** — numeric precision matters more there.
- **Benchmark on your actual task**, not just perplexity. A model that scores well on benchmarks can still fail on your specific prompt distribution.
- **KV cache quantization** (supported in vLLM) can cut memory further without touching weight precision.

## Conclusion

Budget inference is no longer a compromise — it is an engineering discipline. Pick the right quantization format for your hardware, benchmark on your task, and iterate. The tooling in 2026 makes this accessible to any ML engineer with a single consumer GPU.
