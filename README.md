# FAAR: Forgetting-Aware Adaptive Replay on TRACE

Sequential QLoRA baseline vs. **FAAR** (Forgetting-Aware Adaptive Replay) on the [TRACE](https://arxiv.org/abs/2310.06762) continual-learning benchmark, using **Qwen2.5-7B-Instruct**, plus a lightweight **Qwen2.5-0.5B** baseline.

> **Status:** single-seed, directional results. See [Limitations](#limitations).

## Overview

A model fine-tuned on tasks one after another tends to forget earlier tasks. The sequential baseline never revisits old tasks. **FAAR** mixes replayed examples into each training step and spends a fixed replay budget where forgetting is measured:

```
f(t) = best(t) − latest(t)
w(t) = max(0, f(t)) / Σ_j max(0, f(j))
```

- `best(t)`: highest score seen so far on task `t`; `latest(t)`: its most recent score.
- Before each new task, 200 replay examples are drawn: each draw picks an earlier task `t` with probability `w(t)`, then a random training example of `t`.
- If no earlier task shows positive forgetting, replay is uniform.
- From task 2 on, each step trains on **500 current + 200 replayed** examples (baseline: 500).

## Results

### Aggregate (Qwen2.5-7B-Instruct, seed 42)

| Metric | Sequential QLoRA | FAAR | Change |
|---|---|---|---|
| OP ↑ | 0.1432 | **0.1485** | +0.0053 (+3.7%) |
| BWT (→ 0) | −0.0114 | **−0.0023** | +0.0092 |
| Mean forgetting ↓ | 0.0114 | **0.0023** | −80.2% |

### Per task

| Task | Final score (baseline) | Final score (FAAR) | Δ | Forgetting (baseline) | Forgetting (FAAR) |
|---|---|---|---|---|---|
| C-STANCE | 0.0855 | 0.1430 | +0.0575 | −0.0063 | −0.0569 |
| FOMC | 0.0309 | 0.0275 | −0.0034 | +0.0034 | +0.0060 |
| MeetingBank | 0.2230 | 0.1936 | −0.0294 | +0.0571 | +0.0840 |
| Py150 | 0.2508 | 0.2483 | −0.0025 | +0.0029 | +0.0061 |
| ScienceQA | 0.2681 | 0.3090 | +0.0409 | +0.0438 | −0.0180 |
| NumGLUE-cm | 0.0448 | 0.0528 | +0.0080 | +0.0044 | −0.0080 |
| NumGLUE-ds | 0.0397 | 0.0308 | −0.0089 | −0.0140 | +0.0049 |
| 20Minuten | 0.2029 | 0.1830 | −0.0199 | 0.0000 | 0.0000 |

Negative forgetting means the task kept improving after it was learned. FAAR is better on average but not on every task: it ends higher on 3 of 8 tasks, and MeetingBank forgetting gets worse.

### Qwen2.5-0.5B baseline (sanity check)

| Setup | OP | BWT | Mean forgetting |
|---|---|---|---|
| LoRA r=4, CPU, 1 epoch, seed 42, 3 built-in examples per task | 0.625 | 0.000 | 0.000 |

⚠️ **Not comparable with the 7B results.** Scores move in steps of 1/3, and the zero BWT/forgetting comes from cancellation (FOMC +0.33, Py150 −0.33), not from an absence of forgetting.

## Setup

| Item | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4, double quantization, bf16 compute |
| LoRA | r=16, α=32, dropout 0.05; q, k, v, o, gate, up, down (~40.4M trainable params, 0.53%) |
| Training | 1 epoch per task, LR 1e-4 (cosine), batch size 1 × 8 grad. accumulation, paged AdamW 8-bit |
| Data | 500 training instances per task, about 100 test items per task (81 for NumGLUE-cm) |
| Sequence limits | 768 input tokens, 64 generated tokens |
| Hardware | 1 × NVIDIA A40 (44 GB) |
| Seed | 42 |

**Task order:** C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten

## Metrics

With `R[i, j]` = score on task `j` after training on task `i` (T = 8):

- **OP** = mean of `R[T, j]` over all tasks
- **BWT** = mean of `R[T, j] − R[j, j]` (negative = forgetting)
- **Per-task forgetting** `F_j = R[j, j] − R[T, j]`; mean forgetting = −BWT

Scores come from a generic per-task scorer, **not** TRACE's official task-specific metrics. Compare runs within this repo; do not compare against published TRACE numbers.

## Repository structure

```
├── BASELINE-results/            # sequential QLoRA baseline (7B)
├── FAAR-results/                # FAAR run (7B)
├── TRACE_COMPARISON/            # baseline vs FAAR comparison
│   ├── baseline_vs_faar.csv
│   └── baseline_vs_faar_forgetting.png
└── Qwen2_5_0_5B_Baseline.ipynb  # 0.5B sanity-check notebook
```

Each results folder contains:

| File | Contents |
|---|---|
| `metrics.csv` | OP, BWT, mean forgetting |
| `performance_matrix.csv` | `R[i, j]` matrix |
| `performance_matrix_partial.csv` | matrix saved during the run |
| `forgetting.csv` | per-task forgetting |
| `performance_matrix_heatmap.png` | matrix heatmap |
| `manifest.json` | run config (model, seed, GPU, OP, BWT) |
| `after_XX_<task>.json` | evaluation outputs after each training step |

## Reproducing

```bash
# 1. Install dependencies (adjust to your environment)
pip install -r requirements.txt

# 2. Run the baseline (replace with your actual script/notebook)
python <baseline_script>.py --seed 42

# 3. Run FAAR
python <faar_script>.py --seed 42 --replay 200
```

The 0.5B baseline runs on CPU from `Qwen2_5_0_5B_Baseline.ipynb`.

## Limitations

- **Single seed** per method, so there are no error bars.
- **Not compute-matched:** FAAR trains on 700 examples per step vs. 500, so part of the gain may come from extra training rather than adaptive allocation.
- **No ablations yet:** adaptive allocation has not been separated from uniform replay at the same budget.
- **Small test sets and generic scorer:** with about 100 items per task, per-task differences of a few hundredths may be noise.
- **Skewed reference:** `best(t)` includes pre-training check scores, which pushes replay toward C-STANCE (estimated at about 88% of replay draws; reconstructed from the performance matrix, since weights were not logged).
- **Reactive allocation:** a task that has not yet lost ground gets no replay (e.g. MeetingBank at the final step).

## Next steps

- Multiple seeds for both methods
- Uniform-replay control with the same 200-example budget
- Replay-budget sweep (50 / 100 / 200 / 400)
- Compute `best(t)` from post-training scores only, and log replay weights
- Error analysis of MeetingBank outputs
- 0.5B run on the exact TRACE data

## References

- Wang et al., *TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models*, 2023.
