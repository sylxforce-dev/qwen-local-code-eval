# Local Code Generation Inference Benchmark

## Overview

This experiment evaluated local Qwen models for Patchsmith's asynchronous batch code-generation workload across **three distinct quantization tiers of the same model family**, plus a dedicated smaller code specialist, to determine which configuration provides the best combination of:

- Code correctness
- Framework compliance (minimizing technical debt)
- Generation stability & resistance to repetition loops
- Latency & throughput
- VRAM/RAM efficiency and hardware constraints

All inference was performed locally through `llama_cpp`.

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

| Model | Quantization | Approx. size | Offload |
|---|---|---|---|
| Qwen 3.8-27B-UD-IQ1_S | ~1.56 bits/weight | ~6.2 GB | Full GPU |
| Qwen 2.5-Coder-7B-Instruct-Q6_K | 6-bit | — | Full GPU |
| Qwen 3.8-27B-UD-IQ4_XS | ~4.25 bits/weight | 14.3 GB | Partial GPU/CPU |

---

# Part 1 — Token Bucket (Algorithmic Logic)

Every model in this part received the same Token Bucket completion task:

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

## Experiment 1-3 — The IQ1_S Breakdown

The 1-bit quantized model (27B IQ1_S) was subjected to three different generation strategies to force a correct completion. All failed, but the failure modes shifted depending on the intervention.

| Exp | Strategy | Latency | tok/s | Result | Failure Mode |
|---|---|---:|---:|:---|:---|
| **1** | Forced Thinking | ~242 s | 32.95 | ❌ Failed | **Repetitive reasoning.** Model understood the algorithm but failed to converge, caught in an infinite loop of reconsidering its own plan. |
| **2** | HARD CLAMP | 39.43 s | 37.38 | ❌ Failed | **Copy-loop.** Thinking suppressed via prefix injection. Fast generation, but repeated the same incomplete class structure 11x until token limit. |
| **3** | Sampling Intervention | 47.46 s | 36.98 | ❌ Failed | **Semantic failure.** Increased penalties stopped exact copying but caused invalid constructs (`capacity if False else 0`). |

**Observation:** Controlling the generation process (clamps, sampling) altered *how* the model failed, but could not restore reliable completion. The underlying capability was too degraded.

## Experiment 4 — Dedicated Code Model (7B Q6_K)

Instruct mode, temperature 0.2, no HARD CLAMP, standard prompt. 

| Metric | Result |
|---|---:|
| Latency | 2.51 s |
| Generated tokens | 147 |
| Throughput | 58.52 tok/s |
| Correct completion | ✅ Yes |

**Observation:** Successful completion on the first generation. All six required behaviors implemented correctly, no repetition, no reasoning runaway, no prompt manipulation needed.

---

# Part 2 — Structural & Framework Constraints

## Experiment 5 — SQLAlchemy Structure (27B IQ1_S)

Task: generate a SQLAlchemy `User` model with an integer primary key, unique username, unique email, and a `created_at` default. 

| Metric | Result |
|---|---:|
| Latency | 2.35 s |
| Generated tokens | 82 |
| Throughput | 34.92 tok/s |
| Correct completion | ❌ No |

**Observation:** Output was syntactically plausible but violated multiple explicit requirements: `email` missing `unique=True`; `created_at` used `default=datetime.now()` (an execution-time trap); no declarative `Base` inheritance. Short, fast output is not the same as correct output.

## Experiment 6 — FastAPI + Pydantic (A/B Comparison)

Task: combine a FastAPI POST endpoint with Pydantic validation and an age restriction. 

| Model | Approach | Executable | Verdict |
|---|---|:---:|:---|
| **27B IQ1_S** | Applied `Field` constraint but hallucinated an unimported `Any` return type. | ❌ No | **NameError at runtime.** Looks sophisticated but fails instantly. |
| **7B Q6_K** | Ignored `Field`, used a simpler `if request.age < 18` + `HTTPException`. | ✅ Yes | **Functional.** Prioritized basic executability over framework elegance (1.65 s, 60.44 tok/s). |

**Observation:** This reinforces that executable output beats sophisticated-looking output that silently fails.

## Experiment 7-9 — "TaskBot" Architecture Dictation (Cross-Tier Evaluation)

Thinking disabled entirely. The exact same prompt was run across all three models. The full architecture (imports, DB connection line, response structure) was heavily scaffolded and explicitly dictated in the prompt, leaving the models no room to reason independently.

| Model | Speed | Instruction Adherence | Code Safety | Verdict |
|---|---:|:---|:---|:---|
| **27B IQ1_S** | 29.15 tok/s | **Failed.** Used dictated imports, but syntax broke. | ❌ **Fatal Error.** Treated attribute as method: `item.__dict__()`. Transaction left uncommitted. | ❌ Unusable |
| **7B Q6_K** | 57.05 tok/s | **"Rebellious".** Added unrequested `HTTPException`. Padded missing tax with `None`. | ⚠️ **Safe.** Correct syntax (`item.dict()`). Added protective `commit()` on its own. | ⚠️ Requires Linting |
| **27B IQ4_XS** | 2.52 tok/s | **100% Exact.** Used only the 3 requested imports. Exact JSON schema. | ✅ **Perfect.** Correct syntax, defensive DB transaction (`commit()`, `close()`). | ✅ Architecturally Gold |

### Behavioral Breakdown:
*   **The IQ1 Syntax Collapse (1-bit):** Even with 90%+ of the logic dictated, the IQ1 model still produced code with broken basic Python syntax knowledge (`item.__dict__()`). This proves that 1-bit quantization damages the model's fundamental language knowledge, not just its ability to plan.
*   **The "Survival Instinct" (6-bit):** The 7B model did not blindly obey the overly restrictive "exactly these imports" instruction. It fell back on internalized framework conventions (importing `HTTPException` for FastAPI) to keep the result robust.
*   **The Hardware Ceiling (4.25-bit):** At ~4.25 bits, the model exhibits maximal instruction compliance *and* independently defensive coding practice. The only cost is throughput: **2.52 tok/s**, driven entirely by partial CPU offload since the 14.3GB file does not fit in 8GB VRAM.

---

# Key Findings

1. **Parameter count is not a sufficient indicator of practical code-generation quality.** The 27B model was substantially larger than the 7B model, but IQ1 quantization introduced severe generation instability. 
2. **IQ1 introduced severe, distinct failure modes.** Uncontrolled reasoning, repetition collapse, and semantic instability all stemmed from the same underlying quantization damage.
3. **Controlling the generation process does not fix quality.** Prefix injection (HARD CLAMP) cut latency (~242s → ~39s) but could not produce a correct completion.
4. **Latency and token count are useless without correctness.** The IQ1 model produced efficient-looking 82-token outputs that violated strict architectural constraints.
5. **Below a certain bit-depth, syntax knowledge corrupts.** The IQ1 model failed basic Python syntax (`item.__dict__()`) even when fully prompted. No amount of prompt engineering recovers weights that no longer exist.
6. **"Rebel Syndrome" cannot be prompted away.** The 7B Q6 model's internalized framework conventions (adding unrequested protective imports) are baked into the weights and persist regardless of prompt strictness.
7. **Hardware currently dictates the ceiling, not architecture.** The 27B IQ4_XS model is precise and instruction-exact, but limited to ~2.5 tok/s due to 16GB system RAM / 8GB VRAM constraints.

---

# Production Decision & Compensation Protocol

**Selected engine for the current asynchronous batch pipeline:** `Qwen2.5-Coder-7B-Instruct-Q6_K`

**Reasoning:** Async batch processing over potentially hundreds of files is throughput-bound. 55–60 tok/s matches that requirement; 2.5 tok/s (Q4_XS tier) does not, regardless of its superior correctness.

**Documented, accepted technical debt:** The 7B model reliably adds small unrequested elements (defensive imports like `HTTPException`, `None`-padded optional fields). This cannot be reliably suppressed via system prompting.

**Compensation protocol (adopted):** Rather than fighting this behavior at the prompt level, accept it at generation time and remove it deterministically afterward. Pipe all generated code through a standard linter/import-cleaner (`ruff`, `isort`) as a mandatory post-processing step. This strips unused defensive imports at zero cost to generation throughput.

```text
Qwen2.5-Coder-7B-Instruct-Q6_K
        ↓
local GPU inference (~57 tok/s)
        ↓
raw code (correct, but with unrequested scaffolding)
        ↓
ruff / isort cleanup pass
        ↓
clean, lint-passing code
        ↓
Patchsmith pipeline
```

**Reference / Fallback Tier:** `Qwen 3.8-27B-UD-IQ4_XS`. Kept as the documented "gold standard" for cases where correctness matters more than latency, and as the reference against which the 7B model's technical debt is measured.

**Retired:** `Qwen 3.8-27B-UD-IQ1_S`. Confirmed unusable for code generation.
