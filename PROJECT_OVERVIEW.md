# AutoResearch — Project Overview

## What Is AutoResearch?

AutoResearch is an autonomous AI research experimentation framework created by Andrej Karpathy. It gives an AI agent a real, single-GPU LLM training setup and lets it experiment **autonomously overnight**. The agent modifies code, trains for exactly 5 minutes, evaluates the result, keeps improvements or discards regressions, and repeats — yielding approximately 100 experiments while you sleep.

The training code is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The key innovation is that **humans don't touch Python files** — instead, they program `program.md` files that instruct AI agents, creating an autonomous research workflow.

## Core Architecture

### Three Files That Matter

| File | Purpose | Edited By |
|------|---------|-----------|
| `prepare.py` | Fixed constants, one-time data prep (downloads training data, trains BPE tokenizer), runtime utilities (dataloader, evaluation) | Nobody — this is fixed infrastructure |
| `train.py` | Full GPT model, optimizer (Muon + AdamW), and training loop. Architecture, hyperparameters, batch size — everything is fair game | The AI agent |
| `program.md` | Baseline instructions for one agent — the "research org code" | The human |

### The Experiment Loop

1. Agent analyzes current `train.py` and proposes a modification
2. Commits the change to git
3. Runs `uv run train.py` (exactly 5 minutes, wall clock)
4. Extracts `val_bpb` (validation bits per byte) from logs
5. If improved → keeps the change and advances the branch
6. If regressed or crashed → discards via `git reset`
7. Logs results to `results.tsv`
8. Repeats indefinitely without human intervention

## Technical Highlights

### Model Architecture (in `train.py`)
- **GPT-style transformer** with configurable depth, heads, and embedding dimensions
- **Flash Attention 3** kernels (with fallback for non-Hopper GPUs)
- **Rotary positional embeddings** (RoPE)
- **Sliding window attention** with configurable patterns (`SSSL` = 3 half-context + 1 full-context)
- **Value embeddings** (ResFormer-style) for enhanced attention
- **RMSNorm** normalization and **ReLU²** activation in MLPs
- **Softcap logits** for numerical stability

### Optimizer
- **MuonAdamW** — a hybrid combining:
  - **Muon** for 2D matrix parameters (orthogonalization + momentum)
  - **AdamW** for embeddings, unembedding, and scalar parameters
- Per-parameter-group learning rates
- Warmup/warmdown scheduling

### Data Pipeline
- Downloads shards from HuggingFace `climbmix-400b` dataset
- Trains a BPE tokenizer (~8K vocabulary) using `rustbpe`
- **Best-fit packing dataloader** — 100% token utilization with zero padding waste
- Pinned validation shard for consistent evaluation

### Evaluation Metric
- **val_bpb** (validation bits per byte) — lower is better
- Vocabulary-size-independent, enabling fair comparison across architectural changes
- Computed from cross-entropy in nats, normalized by actual byte lengths
- Special tokens excluded from calculation

## Quick Start

**Requirements:** Single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment manually (~5 min)
uv run train.py
```

### Running Autonomously

Spin up an AI agent (Claude, Codex, etc.) in this repo with permissions disabled, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent reads `program.md`, creates a git branch, and enters the autonomous loop.

## Key Hyperparameters

```python
# Architecture
DEPTH = 8                   # transformer layers (primary complexity knob)
ASPECT_RATIO = 64           # model_dim = depth * aspect_ratio
HEAD_DIM = 128              # attention head dimension
WINDOW_PATTERN = "SSSL"     # L=full context, S=half context window

# Optimization
TOTAL_BATCH_SIZE = 2**19    # tokens per gradient step
DEVICE_BATCH_SIZE = 128     # sequences per GPU forward pass
EMBEDDING_LR = 0.6          # learning rate for embeddings
MATRIX_LR = 0.04            # learning rate for Muon-optimized matrices
WARMDOWN_RATIO = 0.5        # fraction of training for LR decay

# Fixed (agent cannot change)
TIME_BUDGET = 300            # seconds (5 minutes)
MAX_SEQ_LEN = 2048          # context length
EVAL_TOKENS = 40 * 524288   # validation tokens
```

## Design Choices

- **Single file to modify** — Only `train.py` is touched by the agent, keeping scope manageable and diffs reviewable
- **Fixed time budget** — 5-minute wall clock makes experiments directly comparable regardless of architecture/batch changes. Finds the optimal model for *your specific hardware*
- **Self-contained** — No distributed training, no complex configs. One GPU, one file, one metric
- **Git-based tracking** — Full experimental history preserved; easy to analyze, rewind, or branch

## How Other Projects Can Benefit

### 1. As a Research Methodology
The "fixed-budget autonomous experimentation" pattern is not limited to LLM pretraining. Any ML project can adopt this approach:
- Fork the repo, replace `train.py` with your own training code
- Keep `prepare.py`'s evaluation harness pattern (fixed budget, single metric)
- Let agents explore your specific design space overnight

### 2. Reusable Components
Several components in the codebase are self-contained and valuable outside this project:
- **BPE tokenizer training** via `rustbpe` — fast, minimal-dependency tokenizer
- **Best-fit packing dataloader** — achieves 100% token utilization without padding
- **BPB evaluation** — vocab-size-independent metric, fairer than perplexity
- **MuonAdamW optimizer** — modern hybrid optimizer for transformer training

### 3. Agent-Driven Development Template
The `program.md` pattern is a reusable paradigm:
- Human writes high-level instructions in Markdown
- Agent executes autonomously, making decisions within defined constraints
- Results tracked via structured logs (`results.tsv`) and git history
- Applicable to any project where you want AI agents to iterate independently

### 4. Modern GPT Reference Implementation
`train.py` is a clean, single-file, well-tuned GPT implementation featuring:
- Flash Attention 3, RoPE, sliding window attention, value embeddings
- Muon optimizer with proper per-parameter-group configuration
- Fused compiled kernels for training speed
- Study it, borrow patterns, or use it as a teaching tool

### 5. Alternative to Traditional Hyperparameter Search
Instead of grid search, random search, or Bayesian optimization, this uses an LLM agent's **reasoning** to guide exploration — often more creative and context-aware than purely numerical methods.

## Platform Support

- **Primary:** NVIDIA GPU (H100 optimized)
- **Community forks:**
  - [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — MacOS
  - [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MacOS with MLX
  - [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) — Windows RTX

## License

MIT
