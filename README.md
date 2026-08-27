# Qwen Quantization Benchmark: Throughput vs. Fidelity under 8GB VRAM

This repository contains an empirical benchmark evaluating how aggressive quantization impacts local Python code generation. 

A common assumption in local LLM deployment is that a larger parameter count can compensate for lower bit-depth. This benchmark tests that hypothesis by evaluating the Qwen model family across 1-bit, 4-bit, and 6-bit quantization tiers under a strict 8GB VRAM constraint. 

The goal was to isolate exactly how failure occurs at different compression levels: does the model simply lose complex reasoning, or does quantization corrupt basic language syntax itself?

### 🔬 The TL;DR Findings
* **1-bit (27B IQ1_S):** Fails fatally. Extreme compression destroyed foundational Python syntax knowledge (e.g., calling `item.__dict__()` as a method), proving the model cannot be "prompt-engineered" out of hardware-level brain damage.
* **6-bit (7B Q6_K):** The production winner for latency (~57 tok/s). Safe and executable, though it exhibits a persistent "rebel syndrome" (adding unrequested defensive framework imports) which must be handled via deterministic post-generation linting.
* **4.25-bit (27B IQ4_XS):** Architecturally perfect and 100% instruction-exact. However, limited to ~2.5 tok/s due to partial CPU offload, proving the current bottleneck for this tier is RAM capacity, not model logic.

### 📄 Full Technical Report
For the complete breakdown of the 10-stage experiment, structural framework constraints, and the adopted compensation protocol, read the full benchmark document:

👉 **[Local LLM Quantization & Code Generation Benchmark](local-quant-benchmark.md)**
