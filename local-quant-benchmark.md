# Local Code Generation Inference Benchmark

## Overview

This experiment evaluated local Qwen models for Patchsmith's asynchronous batch code-generation workload across **two quantization variants of the exact same 27B base model** and a dedicated smaller code specialist, to determine which configuration provides the best combination of:

- Code correctness
- Framework compliance (minimizing technical debt)
- Generation stability & resistance to repetition loops
- Latency & throughput
- VRAM/RAM efficiency and hardware constraints

All inference was performed locally through `llama_cpp`.

---

## Scope & Reproducibility

**Disclaimer: My Testing, My Findings, My Hardware.** 
This benchmark represents independent testing executed strictly on a specific consumer-tier hardware configuration. Local LLM inference is highly sensitive to hardware constraints, particularly VRAM capacity, memory bandwidth, and offload distribution. The generation speeds, failure modes, and VRAM thrashing behaviors observed here are a direct result of this specific environment. **Your results may differ** depending on your silicon, system RAM, and quantization build.

---

## Hardware

- **CPU:** AMD Ryzen 7 7700
- **GPU:** NVIDIA RTX 5060 Ti 8GB VRAM
- **System RAM:** 16GB
- **Storage:** M.2 SSD
- **Runtime:** `llama_cpp`
- **Inference:** Local, GPU + partial CPU offload where the model exceeds VRAM
- **Workload:** Python code completion
- **Target:** Asynchronous batch processing
- **Context:** 8192 tokens (4096 for the largest quant tier)

---

## Models Under Test

To isolate the effect of extreme quantization, the 27B tests directly compare two configurations of the **exact same base model** (`Qwen 3.8-27B-UD`).

| Model | Quantization | Approx. size | Offload |
|---|---|---|---|
| Qwen 3.8-27B-UD-IQ1_S | ~1.56 bits/weight | ~6.2 GB | Full GPU |
| Qwen 2.5-Coder-7B-Instruct-Q6_K | 6-bit | — | Full GPU |
| Qwen 3.8-27B-UD-IQ4_XS | ~4.25 bits/weight | 14.3 GB | Partial GPU/CPU |

---

# Part 1 — Token Bucket (Algorithmic Logic)

Every model in this part received the exact same prefix/prompt to complete:

**The Prompt / Input:**
```python
import time

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()

    def consume(self, amount: int) -> bool:
        # Calculate elapsed time
        # Refill tokens according to refill_rate
        # Do not exceed capacity
        # Update last_refill
        # Consume tokens when sufficient
        pass
```

## The IQ1_S Baseline (Forced Thinking)

The 1-bit quantized model (27B IQ1_S) was evaluated on its default reasoning path to establish a baseline.

| Metric | Result |
|---|---:|
| Latency | ~242 s |
| Generated tokens | 7,973 |
| Throughput | 32.95 tok/s |
| Correct completion | ❌ Failed |

**Observation:** The model demonstrated enough knowledge to identify the intended algorithm but failed to converge efficiently. It was caught in an infinite loop of reconsidering its own plan. The primary failure mode was **repetitive reasoning**, not lack of basic programming knowledge.

## Dedicated Code Model (7B Q6_K)

Instruct mode, temperature 0.2.

| Metric | Result |
|---|---:|
| Latency | 2.51 s |
| Generated tokens | 147 |
| Throughput | 58.52 tok/s |
| Correct completion | ✅ Yes |

**Observation:** Successful completion on the first generation. All six required behaviors implemented correctly, no repetition, no reasoning runaway, no prompt manipulation needed.

---

# Part 2 — Structural & Framework Constraints

## SQLAlchemy Structure (27B IQ1_S)

**Task Description:** Generate a SQLAlchemy `User` model with an integer primary key, unique username, unique email, and a `created_at` default. 

| Metric | Result |
|---|---:|
| Latency | 2.35 s |
| Generated tokens | 82 |
| Throughput | 34.92 tok/s |
| Correct completion | ❌ No |

**Observation:** Output was syntactically plausible but violated multiple explicit requirements: `email` missing `unique=True`; `created_at` used `default=datetime.now()` (an execution-time trap); no declarative `Base` inheritance. Short, fast output is not the same as correct output.

## FastAPI + Pydantic (Three-Tier Comparison)

The same FastAPI registration prompt was evaluated across three configurations to test framework syntax and logic handling.

**The Exact Prompt:**
> "Write a complete Python FastAPI endpoint for POST '/register'. First, define a Pydantic model 'RegisterRequest' with 'username' (string) and 'age' (integer, minimum 18). Then, write the endpoint function that accepts this model and returns a JSON response: {'status': 'success', 'user': username}. Do not add any explanations, just the Python code."

| Model | Implementation Behavior | Verdict |
|---|---|:---|
| **27B IQ1_S** | Produced a framework-aware solution but introduced an unimported `Any` type and failed to satisfy the task reliably. | ❌ **Failed.** NameError at runtime. |
| **7B Q6_K** | Produced executable code and enforced the age requirement manually with `if request.age < 18` + `HTTPException`, but did not encode the minimum-age constraint in the Pydantic model. | ⚠️ **Functional.** Fast and safe, but bypassed framework elegance. |
| **27B IQ4_XS** | Correctly encoded the requirement as `age: int = Field(..., ge=18)` and produced a minimal executable endpoint without unnecessary validation logic. | ✅ **Correct.** Highest instruction fidelity. |

**Observation:** The three configurations demonstrate a clear quality separation. The 7B model remained highly usable because its output was executable and fast, while the IQ4_XS 27B model showed the strongest ability to preserve the semantic structure of the instruction itself. The IQ1 configuration showed the least reliable constraint preservation.

## Architecture Dictation (TaskBot)

Thinking disabled entirely. The exact same highly restrictive prompt was run across all three models to test strict adherence, explicitly dictating imports, SQL logic, and response schemas.

**The Exact Prompt:**
> "Write a complete Python script for a FastAPI application managing an inventory system. ARCHITECTURE STRICTLY DEFINED BY THE ARCHITECT: 1. Imports: You MUST use exactly these imports and nothing else: 'import sqlite3', 'from fastapi import FastAPI', 'from pydantic import BaseModel'. 2. Model: Create a Pydantic model 'InventoryItem' with 'item_id' (integer), 'name' (string), and 'price' (float). 3. Endpoint: Create a POST endpoint '/inventory/add' accepting the 'InventoryItem'. 4. DB Connection: Inside the endpoint, strictly use exactly this line: conn = sqlite3.connect(':memory:') 5. Table: Execute exact SQL: CREATE TABLE IF NOT EXISTS inventory (item_id INTEGER PRIMARY KEY, name TEXT, price REAL) 6. Insert: Insert the item into the table. 7. Logic: If 'price' > 1000, calculate 'premium_tax' as price * 0.2 (float). 8. Return: A JSON response with 'status': 'success', 'item': the item data, and 'premium_tax' if applicable. CRITICAL COMMAND: Do not explain. Do not think. Output ONLY the raw Python code."

| Model | Speed | Instruction Adherence | Code Safety | Verdict |
|---|---:|:---|:---|:---|
| **27B IQ1_S** | 29.15 tok/s | **Failed.** Used dictated imports, but syntax broke. | ❌ **Fatal Error.** Treated attribute as method: `item.__dict__()`. | ❌ Unusable |
| **7B Q6_K** | 57.05 tok/s | **Suboptimal.** Added unrequested `HTTPException`. Padded missing tax with `None`. | ⚠️ **Safe.** Correct syntax (`item.dict()`). Added protective `commit()`. | ⚠️ Requires Linting |
| **27B IQ4_XS** | 2.52 tok/s | **100% Exact.** Used only the 3 requested imports. Exact JSON schema. | ✅ **Perfect.** Correct syntax, defensive DB transaction (`commit()`, `close()`). | ✅ Highest Fidelity |

**Observation:** Even with the majority of the architecture explicitly dictated and autonomous reasoning suppressed, the IQ1 configuration still produced a basic executable-code failure. The model generated `item.__dict__()` even though `__dict__` is an attribute, not a callable method. This provides strong evidence that the IQ1_S configuration can severely degrade basic Python syntax reliability, in addition to higher-level generation instability.

---

# Part 3 — The Complexity Cliff

To test the algorithmic boundaries and constraint adherence across the configurations, they were subjected to an escalating series of tasks.

| Task | IQ1_S | 7B Q6_K | 27B ~4-bit | Result |
|---|---|---|---|---|
| `calculate_retry_delay` | ✅ Correct (~28 t/s) | ✅ Correct (~53 t/s) | ✅ Correct (~1.6 t/s) | All models succeed on simple logic. |
| `reserve_capacity` | ✅ Correct (~28 t/s) | ✅ Correct (~55 t/s) | ✅ Correct (~2.3 t/s) | All models handle basic math and checks. |
| `allocate_workers` | ⚠️ Passed on retry | ✅ Correct (~55 t/s) | ✅ Correct (~2.0 t/s) | IQ1 stumbles; Q6 & Q4 remain perfect. |
| `schedule_jobs` (Complex) | 💀 **Fatal Breakdown** | ❌ **Failed (Logic Collapse)** | ✅ Correct (~2.6 t/s) | **The Complexity Cliff:** Both IQ1 and Q6 collapse under extreme constraint density; only ~4-bit executes flawlessly. |

## The `schedule_jobs` Stress Test

The models were asked to implement a stateful job scheduler involving validation, duplicate ID filtering, priority fallbacks, stable sorting, capacity checks, and deterministic output.

**Behavioral Breakdown:**
- **27B IQ1_S (34.61 tok/s):** The generation completely collapsed. It added a forbidden helper function, reversed the required sorting order, hallucinated logic (`min_count/sorted(set(...))`), and failed to complete the assignment loop entirely.
- **7B Q6_K (57.79 tok/s):** Produced code rapidly but failed structurally. It mutated the original input dictionary (violating strict immutability constraints), implemented a fatal slicing error for duplicate checks that guarantees a `TypeError` for opaque IDs, failed the stable sorting logic (due to a variable scope leak in the lambda function), and completely ignored the lowest-load worker selection algorithm.
- **27B ~4-bit (2.60 tok/s):** Produced a complete, 327-token executable algorithm. All core constraints and sorting logic were perfectly maintained.

**Observation:** The benchmark reveals a "threshold effect" rather than a linear degradation. IQ1_S and 7B Q6_K are fully capable of writing simple, localized Python snippets. However, as the prompt lengthens and the "constraint density" (the number of rules the model must hold in its attention simultaneously) increases to 19 explicit rules, both lower-tier configurations suffer a catastrophic quality cliff. Only the ~4-bit model maintains the base model's true capability and holds all constraints in memory, albeit at a severe speed penalty.

---

# Part 4 — Independent Cross-Validation

## Independent `/orders` Constraint Test

A new, independently designed FastAPI/Pydantic task was used to test whether the observed behavior generalized beyond the `inventory` TaskBot prompt. 

The task required:
- Exactly two imports
- Pydantic Field constraints
- Validation implemented in the model rather than the endpoint
- Exact arithmetic ordering
- Conditional second-stage discount
- Exact response schema
- No additional framework logic

| Model | Latency | Throughput | Correct | Instruction Compliance |
|---|---:|---:|:---:|:---|
| **7B Q6_K** | 2.91 s | 55.72 tok/s | ✅ | Exact |
| **27B IQ1_S** | 4.62 s | 32.01 tok/s | ❌ | Failed |
| **27B IQ4_XS** | 68.09 s | 2.32 tok/s | ✅ | Exact |

**Behavioral Breakdown:**
- **7B Q6_K:** Produced a complete, executable implementation with all requested Pydantic constraints, arithmetic operations, conditional discount logic, and exact response fields. No unrequested imports or logic were added.
- **27B IQ1_S:** Omitted two explicitly required imports (`BaseModel` and `Field`) while still using both symbols in the code, producing a non-executable script that guarantees a `NameError`. It also emitted a `<think>` block despite the code-only instruction.
- **27B IQ4_XS:** Produced a complete and exact implementation with all requested imports, constraints, arithmetic, conditional logic, and response fields. No unrequested logic was introduced.

**Observation:** The independent test reproduces the same broad pattern seen in the earlier experiments: the IQ4_XS configuration retains strong code-generation capability but is severely throughput-limited on the available hardware, while IQ1_S remains substantially less reliable despite its higher generation speed. Crucially, the 7B model followed the specification closely on this mid-level complexity task and added no unrequested framework behavior in this run.

---

# Three-Way Benchmark Summary

| Model | Speed | Code Reliability | Role / Practical Result |
|---|---:|:---|:---|
| **27B IQ1_S** | ~29–37 t/s | ❌ Unstable; fails the "Complexity Cliff" | **Retired** |
| **7B Q6_K** | ~54–60 t/s | ⚠️ Reliable for most tasks, but fails extreme constraint density | **Production** |
| **27B ~4-bit** | ~1.6–2.6 t/s | 🟢 Highest observed fidelity & exactness | **Quality Reference** |

---

# Key Findings

## 1. Parameter count is not a sufficient indicator of quality
The 27B model was substantially larger than the 7B model, but extreme IQ1 quantization introduced severe generation instability, rendering it worse than a smaller 6-bit model on many tasks.

## 2. The "Complexity Cliff"
Quantization creates a non-linear degradation in performance. The 7B Q6_K and 27B IQ1_S models succeed on short, simple tasks, but rapidly break down as constraint density increases (e.g., 19 explicit rules in the `schedule_jobs` stress test). The 27B ~4-bit model was the only configuration to survive the cliff.

## 3. At the tested IQ1_S configuration, basic Python code reliability was severely degraded
Because the IQ4_XS and IQ1_S models are quantized variants of the same base model, the large reliability gap strongly implicates extreme quantization as the primary factor in the observed degradation. The IQ1_S model repeatedly produced non-executable code in the tested tasks, including missing required imports and treating `__dict__` as a callable method. 

## 4. Latency and token count are useless without correctness
The models failing the stress tests produced efficient-looking outputs rapidly (30-58 tok/s) that consistently violated strict architectural constraints and executable safety. A high throughput rate is irrelevant if the resulting code throws a `TypeError` or `NameError`.

## 5. Instruction deviations are task-dependent rather than invariant
The 7B Q6_K model deviated from restrictive instructions in some earlier tests by adding framework conventions such as `HTTPException`, and broke entirely under extreme algorithmic density (`schedule_jobs`). However, the `/orders` test proved it can follow strict architectural constraints exactly when the complexity is balanced. The 7B model acts as a highly capable specialist, capable of both strict and convention-driven behavior depending on the constraint density.

## 6. The ~4-bit tier demonstrates retained capability; hardware is the limiting factor
The 27B ~4-bit configuration produced correct, instruction-exact code in all complex tests (`TaskBot`, `/orders`, `schedule_jobs`). No fundamental code-generation failure was observed. Its practical limitation was throughput: approximately 1.6–2.6 tok/s with partial CPU offload on the 8GB VRAM / 16GB RAM test system. In this benchmark, the ~4-bit tier passed the correctness requirement but did not meet the throughput requirement for the intended high-volume workload on this hardware.

---

# Final Result

The benchmark does not show a simple relationship between parameter count, quantization level, and practical usefulness. 

Instead, the three configurations occupy fundamentally different operating points:

```text
                RELIABILITY / FIDELITY
                       ▲
                       │        27B ~4-bit
                       │          ██████████
                       │
                       │    7B Q6_K
                       │      ██████
                       │
                       │ IQ1_S
                       │ ██
                       └────────────────────────► PRACTICAL THROUGHPUT

                       2.3     55       33 tok/s
                     ~4-bit    Q6       IQ1
```

*   **27B ~4-bit:** Highest observed reliability and fidelity, surviving extreme constraint density and serving as the maximum capability reference, but fundamentally hardware-limited.
*   **7B Q6_K:** Best production throughput / reliability balance, delivering ~54–60 t/s. While it collapses under extreme algorithmic stress testing, it produces highly reliable and specification-compliant code for standard production workloads.
*   **27B IQ1_S:** Unacceptable reliability despite higher throughput, repeatedly failing strict code-generation tasks with non-executable output and collapsing under algorithmic complexity.

For Patchsmith's asynchronous batch requirement, the `Qwen2.5-Coder-7B-Instruct-Q6_K` is selected for production, provided tasks are scoped below its constraint density failure threshold. 

> **We optimize for correct code per second, not tokens per second.**
