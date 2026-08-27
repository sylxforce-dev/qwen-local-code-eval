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

- **GPU:** NVIDIA RTX 5060 Ti 8GB VRAM
- **System RAM:** 16GB
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

## Experiment 1 — Forced Thinking (27B IQ1_S)

The model was allowed to enter its normal reasoning path. It entered an extremely long reasoning sequence, repeatedly reconsidering the same interpretation without converging.

| Metric | Result |
|---|---:|
| Latency | ~242 s |
| Generated tokens | 7,973 |
| Throughput | 32.95 tok/s |
| Correct completion | No |

**Observation:** the model demonstrated enough knowledge to identify the intended algorithm but failed to converge efficiently. The primary failure mode was **repetitive reasoning**, not lack of basic programming knowledge.

## Experiment 2 — HARD CLAMP (27B IQ1_S)

A prefix was injected to terminate the `<think>` section before generation, forcing direct code generation. The first implementation was partially correct but omitted the required token consumption and timestamp update. On recognizing the incomplete implementation, generation entered a **copy-loop**, repeating the same class/method structure ~11 times until the 2048-token limit terminated the run.

| Metric | Result |
|---|---:|
| Latency | 39.43 s |
| Generated tokens | 1,474 |
| Throughput | 37.38 tok/s |
| Correct completion | No |

**Observation:** HARD CLAMP prevented the long reasoning phase but did not restore reliable completion — faster, but less coherent.

## Experiment 3 — Sampling Intervention (27B IQ1_S)

HARD CLAMP retained; `temperature=0.3`, `repeat_penalty=1.25`, `top_p=0.85`, `top_k=10`. The copy-loop changed into a **semantic failure mode** — invalid constructs like `capacity if False else 0` — before the model abandoned the code-only format and re-entered self-referential reasoning.

| Metric | Result |
|---|---:|
| Latency | 47.46 s |
| Generated tokens | 1,755 |
| Throughput | 36.98 tok/s |
| Correct completion | No |

**Observation:** sampling changes altered *how* the model failed, not *whether* it failed.

## Experiment 4 — Dedicated Code Model (7B Q6_K)

Instruct mode, temperature 0.2, no HARD CLAMP, standard prompt. **Successful completion on the first generation** — all six required behaviors implemented correctly, no repetition, no reasoning runaway, no prompt manipulation needed.

| Metric | Result |
|---|---:|
| Latency | 2.51 s |
| Generated tokens | 147 |
| Throughput | 58.52 tok/s |
| Correct completion | Yes |

---

# Part 2 — Structural & Framework Constraints

## Experiment 5 — SQLAlchemy Structure (27B IQ1_S)

Task: generate a SQLAlchemy `User` model with an integer primary key, unique username, unique email, and a `created_at` default. Output was syntactically plausible but violated multiple explicit requirements: `email` missing `unique=True`; `created_at` used `default=datetime.now()` (an execution-time trap) instead of the callable `default=datetime.now`; no declarative `Base` inheritance; `UniqueConstraint` imported but unused; an extraneous `<think>` block and markdown fence despite the code-only instruction.

| Metric | Result |
|---|---:|
| Latency | 2.35 s |
| Generated tokens | 82 |
| Throughput | 34.92 tok/s |
| Correct completion | No |

**Observation:** short, fast output is not the same as correct output — this test required no complex algorithm, only preserving several explicit constraints simultaneously, and still failed.

## Experiment 6 — FastAPI + Pydantic, first pass (27B IQ1_S vs 7B Q6_K)

Task: combine a FastAPI POST endpoint with Pydantic validation and an age restriction.

- **27B IQ1_S:** applied the Pydantic `Field` constraint but introduced an invalid `Any` return type without importing `Any` — a guaranteed `NameError` at runtime.
- **7B Q6_K:** chose a simpler `if request.age < 18: raise HTTPException(...)` implementation — functional and executable, though not using the `Field` constraint the task implied.

| Metric | 7B Q6_K Result |
|---|---:|
| Latency | 1.65 s |
| Throughput | 60.44 tok/s |
| Correct / executable | Yes |

**Observation:** this reinforced that executable output beats sophisticated-looking output that silently fails.

## Experiment 7 — Fully Scaffolded Registration Endpoint (27B IQ1_S, imports dictated)

Same registration-endpoint task, but this time the prompt explicitly pre-listed every required import as a "CRITICAL REQUIREMENT" ("including FastAPI, BaseModel, Field"). Under this heavy scaffolding, the model complied: it added the correct imports, built a `RegisterRequest` class, and applied all requested `Field` constraints (`min_length=3`, `min_length=8`, `ge=18`).

| Metric | Result |
|---|---:|
| Latency | 3.47 s |
| Generated tokens | 120 |
| Throughput | 34.61 tok/s |
| Correct completion | Yes, but only with full import scaffolding |

**Observation:** this is not a genuine recovery of capability. The model only succeeded because every required import was dictated in advance. Removing that explicit list (as in Experiment 6) caused the same failure mode to reappear immediately — the model did not "learn" the dependency, it was handed it.

## Experiment 8 — "TaskBot" Fully-Dictated Architecture (27B IQ1_S)

Thinking disabled entirely; the full architecture (imports, DB connection line, response structure) was dictated in the prompt, leaving the model no room to reason independently. This eliminated the repetition loop — generation completed in one pass.

| Metric | Result |
|---|---:|
| Latency | 6.55 s |
| Generated tokens | 191 |
| Throughput | 29.15 tok/s |
| Correct completion | **No — fatal runtime error** |

The model followed the dictated imports and used the exact required DB connection line (`conn = sqlite3.connect(':memory:')`), but the return statement called `item.__dict__()` — treating `__dict__`, a plain attribute, as if it were a callable method. This throws `TypeError: 'dict' object is not callable` the instant the endpoint is hit.

**Observation:** this is the most important IQ1 result in the whole benchmark. Even with 90%+ of the logic dictated and all autonomous reasoning suppressed, the model still produced code with **broken basic Python syntax knowledge** — not a reasoning failure, a corrupted-fact failure. 1-bit quantization damages the model's fundamental knowledge of the language itself, not just its ability to plan.

## Experiment 9 — Same TaskBot Prompt, 7B Q6_K

Same fully-dictated prompt as Experiment 8, run on the 7B code specialist instead.

| Metric | Result |
|---|---:|
| Latency | 3.52 s |
| Throughput | 57.05 tok/s |
| Correct completion | Yes |

The model used the correct `item.dict()` call (no syntax error). It also deviated from the literal instructions twice, in both cases constructively:

- Added `from fastapi import HTTPException` despite an explicit "exactly these imports and nothing else" instruction — because its internalized FastAPI conventions treat error handling as mandatory in an API context.
- The prompt only said "write the DB row insert"; the model added `cursor = conn.cursor()` and, critically, `conn.commit()` on its own initiative — the IQ1 model (Experiment 8, and separately) left an equivalent transaction hanging uncommitted, meaning data would silently never persist.

**Observation:** this is the "survival instinct" pattern — the 7B model does not blindly obey an incomplete or overly restrictive instruction; it falls back on internalized framework conventions to keep the result actually working, even when that means adding something not explicitly requested.

## Experiment 10 — Same TaskBot Prompt, 27B IQ4_XS

Same fully-dictated prompt, run on the mid-tier quantization (~4.25 bits/weight), with partial CPU offload due to VRAM limits.

| Metric | Result |
|---|---:|
| Latency | — |
| Throughput | 2.52 tok/s |
| Correct completion | **Yes — exact and complete** |

The model used **exactly** the three required imports — no more, no less (in contrast to 7B's unrequested `HTTPException` addition). It independently built a fully correct database interaction including cursor creation, `commit()`, and `close()` — all three, without being told to. The JSON response formatting was precise: an optional tax field was included in the response only when the underlying condition was actually true, rather than always present with a placeholder `None` (which 7B's output did).

**Observation:** at ~4.25 bits, the model exhibits maximal instruction compliance *and* independently correct, defensive coding practice — the combination that IQ1 could not produce even with full dictation (Experiment 8) and that 7B does not produce with full precision (Experiment 9, unrequested imports + `None`-padding). The cost is throughput: **2.52 tok/s**, driven by partial CPU offload since the ~14.3GB file does not fit in 8GB VRAM.

---

# Three-Way Comparison

| Model (Quantization) | Speed | Code Safety | Instruction Adherence | Verdict |
|---|---:|---|---|---|
| 27B IQ1_S (1-bit) | ~29–37 tok/s | **Fatal error** (`item.__dict__()`, uncommitted transaction) | Failed | ❌ Unusable |
| 7B Q6_K (6-bit) | ~57–60 tok/s | Safe, executable | "Rebellious" — adds unrequested imports/logic, `None`-padding | ⚠️ Requires review/lint |
| 27B IQ4_XS (4.25-bit) | ~2.5 tok/s | **Perfect** — exact, defensive, precise | 100% exact | ✅ Architecturally gold, throughput-limited |

---

# Key Findings

## 1. Model size alone was not the winning factor

The 27B model was substantially larger than the 7B model, but IQ1 quantization introduced severe generation instability. The 7B Q6 code-specialized model produced correct implementations dramatically faster.

> **Parameter count is not a sufficient indicator of practical code-generation quality.**

## 2. IQ1 introduced severe, distinct failure modes

Uncontrolled reasoning (Experiment 1), repetition collapse (Experiment 2), and semantic instability under sampling intervention (Experiment 3) — three different failure shapes from the same underlying instability.

## 3. HARD CLAMP improved latency but not correctness

Prefix injection cut generation time (~242s → ~39s) without producing a correct completion.

> **Controlling the generation process is not equivalent to improving the model's underlying generation quality.**

## 4. Sampling is not a substitute for model quality

Changing temperature/repeat-penalty/top-p/top-k changed the failure mode (copy-loop → semantic loop) but never produced a reliable completion.

## 5. Short output does not mean correct output

The SQLAlchemy test (Experiment 5) produced only 82 tokens in 2.35 seconds — efficient-looking, but violated several explicit requirements.

> **Latency and token count are useful only when correctness is measured alongside them.**

## 6. Executability matters more than apparent sophistication

The FastAPI/Pydantic test (Experiment 6) showed the 7B model's simpler, executable solution beating the 27B IQ1 model's more "sophisticated"-looking but broken one.

## 7. Scaffolding can mask, but not fix, a broken model

Experiment 7 showed IQ1 "succeeding" only once every required import was pre-dictated — and Experiment 8 then proved that even *full* dictation of architecture cannot prevent a basic Python syntax error (`item.__dict__()` called as a method). This is the benchmark's clearest evidence that:

> **Below a certain bit-depth, quantization damage extends to the model's basic knowledge of the language syntax itself, not just its higher-level reasoning or planning.**

No amount of prompt engineering recovers this, because the information is not present in the weights to be recovered.

## 8. "Rebel Syndrome" cannot be prompted away

Even a "zero autonomy," maximally restrictive system prompt (Experiment 9) could not stop the 7B model from adding an unrequested `HTTPException` import — its internalized FastAPI conventions are baked into the weights at 6-bit precision and persist regardless of instruction strictness. This is a *feature* of retained capability, not a prompt-following bug, but it does mean the model's output will reliably contain small unrequested additions that a stricter pipeline must account for.

## 9. Hardware currently dictates the ceiling, not model choice

With 16GB system RAM + 8GB VRAM, the 27B IQ4_XS model runs successfully and produces the most precise, correct, and instruction-exact code of any configuration tested (Experiment 10) — but at ~2.5 tok/s, driven by partial CPU offload. This is not a quantization-quality problem; it is a capacity problem. More RAM (a commonly cited threshold is ~32GB for comfortable full-GPU-adjacent operation at this model size) would likely make this tier practically usable for throughput-sensitive work. The current bottleneck is hardware, not architecture.

---

# Production Decision & Compensation Protocol

**Selected engine for the current asynchronous batch pipeline:** `Qwen2.5-Coder-7B-Instruct-Q6_K`

**Reasoning:** async batch processing over potentially hundreds of files is throughput-bound. 55–60 tok/s (2–4s per completion) matches that requirement; 2.5 tok/s (Q4/IQ4_XS tier) does not, regardless of its superior correctness.

**Documented, accepted technical debt:** the 7B model reliably adds small unrequested elements (defensive imports like `HTTPException`, `None`-padded optional fields) that a stricter specification did not ask for. Per Experiment 9, this cannot be reliably suppressed via system prompting — it is a stable property of the model's retained capability at this quantization level, not an inconsistent bug.

**Compensation protocol (adopted):** rather than fighting this behavior at the prompt level, accept it at generation time and remove it deterministically afterward — pipe all generated code through a standard linter/import-cleaner (e.g. `ruff`, `isort`) as a mandatory post-processing step before it enters the codebase. This strips unused defensive imports in a fraction of a second, at zero cost to generation throughput.

```text
Qwen2.5-Coder-7B-Instruct-Q6_K
        ↓
local GPU inference (~57 tok/s)
        ↓
raw code (correct, but with some unrequested scaffolding)
        ↓
ruff / isort cleanup pass
        ↓
clean, lint-passing code
        ↓
Patchsmith pipeline
```

**Reference / fallback tier — not currently in production rotation:** `Qwen 3.8-27B-UD-IQ4_XS`. Kept on file as the documented "gold standard" for cases where correctness matters more than latency (e.g. a final-pass review model, or a future high-RAM configuration), and as the benchmark reference against which the 7B model's technical debt is measured.

**Retired:** `Qwen 3.8-27B-UD-IQ1_S`. Confirmed unusable for code generation at any tested level of prompt control, including full architectural dictation — the failure is rooted in quantization damage to basic language-syntax knowledge, not a prompting problem.

---

# Final Result

The benchmark produced a counterintuitive, three-tier result:

**27B IQ4_XS (correctness) > 7B Q6 (throughput) > 27B IQ1 (unusable)**

for this specific local code-generation workload — with the ranking depending entirely on which axis (correctness vs. throughput) the workload actually optimizes for. For Patchsmith's asynchronous batch requirement, throughput wins today, with 7B Q6's technical debt handled by a deterministic lint pass rather than by further prompting. The 27B IQ4_XS result remains on record as proof that the *right* answer to "can this hardware run a 27B model well" is yes, precision-wise — the constraint is currently RAM capacity, not model architecture or quantization technique.
