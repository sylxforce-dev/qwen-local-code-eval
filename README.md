# Qwen Quantization Benchmark: Throughput vs. Fidelity under 8GB VRAM

This repository contains an empirical benchmark evaluating how aggressive quantization impacts local Python code generation. 

A common assumption in local LLM deployment is that a larger parameter count can compensate for lower bit-depth. This benchmark tests that hypothesis by evaluating the Qwen model family across 1-bit, 4-bit, and 6-bit quantization tiers under a strict 8GB VRAM constraint. 

The goal was to isolate exactly how failure occurs at different compression levels: does the model simply lose complex reasoning, or does quantization corrupt basic language syntax itself?

### 🔬 The TL;DR Findings
* **1-bit (27B IQ1_S):** Fails fatally. Extreme compression destroyed foundational Python syntax knowledge (e.g., calling `item.__dict__()` as a method) and triggered total logic collapse under algorithmic stress.
* **6-bit (7B Q6_K):** The optimal production balance (~54–60 tok/s). Reliable and exact for standard framework tasks, but vulnerable to "The Complexity Cliff"—it structurally breaks down when forced to hold extreme constraint density (e.g., 19 simultaneous explicit rules).
* **4.25-bit (27B IQ4_XS):** The quality reference. Architecturally perfect and capable of handling extreme constraint density and autonomous meta-testing flawlessly. Limited to ~1.6–2.6 tok/s due to partial CPU offload, proving the bottleneck is strictly hardware RAM capacity, not model logic.

### 📄 Full Technical Report
For the complete breakdown of the 4-part experiment—including framework validation, The Complexity Cliff stress test, and the Q4 Meta-Test—read the full benchmark document:

👉 **[Local LLM Quantization & Code Generation Benchmark](local-quant-benchmark.md)**
