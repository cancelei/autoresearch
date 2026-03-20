# Parameter Golf Experiment Plan

## Competition Summary

**Goal:** Train the best language model that fits in **16 MB** and trains in **under 10 minutes on 8xH100 SXM GPUs**.
**Metric:** Bits-per-byte (BPB) on FineWeb validation set — lower is better.
**Deadline:** April 30, 2026.
**Compute:** OpenAI sponsoring $1M in credits; apply via [credit form](https://openai.com/index/parameter-golf/#credit-form).

### Current Leaderboard (as of March 19, 2026)

| Rank | BPB | Technique |
|------|-----|-----------|
| 1st | 1.1748 | Muon WD + 10L + spectral init + resid mix + sliding window eval + FP16 embed |
| 2nd | 1.1925 | Sliding window evaluation (stride=64) |
| 3rd | 1.1929 | LoRA test-time training |
| Baseline | 1.2244 | 9 layers, 512 dim, 1024 vocab, tied embeddings |

---

## Architecture Comparison: AutoResearch vs Parameter Golf

Both repos descend from the same lineage (modded-nanogpt/nanochat), sharing many components:

| Component | AutoResearch `train.py` | Parameter Golf `train_gpt.py` |
|-----------|------------------------|-------------------------------|
| Architecture | GPT + RoPE + RMSNorm + ReLU² MLP | GPT + RoPE + RMSNorm + ReLU² MLP |
| Attention | Flash Attention 3 + sliding window + value embeddings | F.scaled_dot_product_attention + GQA |
| Optimizer | MuonAdamW (Nesterov + NorMuon + cautious WD) | Muon + AdamW (simpler, no NorMuon) |
| Quantization | None (pretraining only) | int8 + zlib compression (critical for 16MB cap) |
| Evaluation | Simple val_bpb | val_bpb + LoRA TTT + sliding window eval |
| Distribution | Single GPU | 8xH100 DDP (torchrun) |
| Residual | resid_lambdas + x0_lambdas | resid_mix + skip connections (U-Net style) |
| Embedding | Separate wte + lm_head + value embeddings | Tied embeddings (critical for 16MB) |

### Key Advantages AutoResearch Brings

1. **Flash Attention 3 kernels** — PG baseline uses `F.scaled_dot_product_attention` (no FA3)
2. **Value embeddings (ResFormer)** — PG baseline has none
3. **Sliding window attention patterns** — PG baseline uses full attention only
4. **NorMuon optimizer** — More advanced Muon variant with variance reduction
5. **Cautious weight decay** — Masks WD based on gradient-parameter alignment
6. **Polar Express coefficients** — Optimized Newton-Schulz iteration

---

## Experiment Strategy

### Phase 1: Quick Wins (No Compute Needed — Local/MLX Development)

These are techniques already proven on the leaderboard that we should implement first:

#### 1.1 Sliding Window Evaluation (Expected: ~0.032 BPB improvement)
- **What:** Score each token with near-full context using overlapping windows at stride=64
- **Why:** Current SOTA #2 achieved this with zero architecture changes
- **Effort:** ~50 lines of eval code
- **From:** Matthew Li's submission

#### 1.2 FP16 Embedding Export (Expected: ~0.006 BPB improvement)
- **What:** Keep tied embedding weights in FP16 during int8 quantization instead of quantizing them
- **Why:** Tied embeddings serve dual duty (input + output); int8 errors compound
- **Effort:** ~10 lines in quantization code
- **From:** Renier Velazco's submission

#### 1.3 Aggressive Warmdown Schedule (Expected: ~0.005 BPB improvement)
- **What:** Set WARMDOWN_ITERS=20000 (beyond total steps) so LR decays throughout training
- **Why:** Produces tighter weight distributions → less int8 quantization damage
- **From:** samuellarson's submission

#### 1.4 Learning Rate Tuning (Expected: ~0.003 BPB improvement)
- **What:** Lower matrix/scalar LR to ~0.02 (half of default 0.04)
- **Why:** Systematic LR sweeps show defaults are too high
- **From:** Nan Liu's submission

**Phase 1 Total Expected Improvement: ~0.046 BPB → ~1.178 BPB (competitive with #1)**

### Phase 2: Architecture Innovations (Compute Grant Needed)

Port the best ideas from autoresearch's `train.py` into parameter-golf:

#### 2.1 Value Embeddings (ResFormer-style)
- **What:** Add input-dependent value residual connections from autoresearch
- **Risk:** Value embeddings add parameters → may not fit in 16MB
- **Mitigation:** Use mixed int6/int8 compression (from Nan Liu's technique) to free ~1.6MB
- **Expected:** 0.005-0.015 BPB (novel in this competition)

#### 2.2 Flash Attention 3 Integration
- **What:** Replace `F.scaled_dot_product_attention` with FA3 kernels
- **Why:** Faster attention → more training steps in 10 minutes → better model
- **Expected:** 0.003-0.008 BPB (from faster training throughput)

#### 2.3 Sliding Window Attention During Training
- **What:** Use `SSSL` pattern (3 short + 1 long window) from autoresearch
- **Why:** Reduces attention cost → can increase batch size or model size
- **Risk:** May hurt quality if model is too small to benefit
- **Expected:** 0.002-0.005 BPB

#### 2.4 NorMuon Optimizer
- **What:** Port the variance-reduction Muon variant from autoresearch
- **Why:** Better gradient normalization → more stable training → better convergence
- **Expected:** 0.002-0.005 BPB

#### 2.5 Longer Training Context (seq_len=4096)
- **What:** Increase from 1024 to 4096 tokens per sequence
- **Why:** Already proven (Spokane Way's submission, 0.023 BPB improvement)
- **Trade-off:** Slower per-step (71ms vs 43ms) but better signal per token
- **Expected:** 0.015-0.023 BPB

### Phase 3: Novel Approaches (High-Risk, High-Reward)

#### 3.1 Autonomous Agent Loop for Parameter Golf
- **What:** Adapt autoresearch's autonomous loop to iterate on `train_gpt.py`
- **How:**
  1. Fork parameter-golf
  2. Create a `program_pg.md` with parameter-golf-specific agent instructions
  3. Set time budget to 10 min on 8xH100
  4. Let agent modify `train_gpt.py` and track results
- **Expected:** ~6 experiments/hour × $4/hr on RunPod = $0.67/experiment

#### 3.2 Depth Recurrence / Weight Tying
- **What:** Share weights across transformer layers (double effective depth for free)
- **Why:** Competition description explicitly calls out "depth recurrence" as interesting
- **Risk:** May not converge well
- **Expected:** 0.005-0.020 BPB if it works

#### 3.3 Mixed Precision Quantization (int4/int6/int8)
- **What:** Use different quantization levels per layer based on sensitivity
- **Why:** Nan Liu showed int6 middle layers work; could push to int4 for least-sensitive layers
- **Expected:** Enables 12+ layers within 16MB budget

#### 3.4 Custom Tokenizer
- **What:** Train an optimized BPE tokenizer (vocab 2048-4096) for better byte efficiency
- **Why:** Competition allows custom tokenizers; could improve BPB directly
- **Risk:** "Submissions that edit the tokenizer will be examined much more carefully"
- **Expected:** 0.005-0.015 BPB

---

## Compute Grant Application Strategy

### Justification for Maximum Grant

**Approach:** Autonomous AI-agent-driven architecture search combining autoresearch methodology with parameter-golf constraints.

**Key Selling Points:**
1. **Efficiency:** Autonomous loop runs ~6 experiments/hour = ~100 experiments/overnight, far more than manual experimentation
2. **Novel techniques:** Porting Flash Attention 3, value embeddings, NorMuon optimizer, and sliding window training from autoresearch — none currently on the leaderboard
3. **Systematic:** Git-tracked experiments with structured results logging enable principled ablation studies
4. **Reproducible:** Every experiment committed and logged for full transparency

**Estimated Compute Needs:**
- Phase 1 (quick wins): ~$50 (10 hrs × 1xH100 for LR sweeps and validation)
- Phase 2 (architecture): ~$500 (25 hrs × 8xH100 for multi-seed validation)
- Phase 3 (autonomous loop): ~$2,000 (100 hrs × 8xH100 overnight runs)
- **Total request: ~$2,500-5,000**

### Application Template

```
Project: Autonomous Architecture Search for Parameter Golf

We're combining the autoresearch autonomous experimentation framework
(github.com/karpathy/autoresearch) with novel architectural components
to systematically explore the parameter-golf design space.

Our approach:
1. Port proven techniques from nanochat/autoresearch (Flash Attention 3,
   value embeddings, NorMuon optimizer, sliding window attention)
2. Run an autonomous AI agent loop that iterates on train_gpt.py,
   testing ~100 configurations per overnight run
3. Combine with proven leaderboard techniques (sliding window eval,
   FP16 embeddings, aggressive warmdown, mixed-precision quantization)

We estimate needing ~100 hours of 8xH100 time for systematic ablation
studies and autonomous search runs.
```

---

## Implementation Roadmap

### Week 1 (Now → March 27)
- [ ] Apply for compute grant with the above justification
- [ ] Implement Phase 1 quick wins on local hardware / 1xH100
- [ ] Submit first leaderboard entry (combining all Phase 1 techniques)
- [ ] Set up autonomous loop for parameter-golf

### Week 2 (March 27 → April 3)
- [ ] Port FA3 and value embeddings from autoresearch
- [ ] Run architecture ablations on 8xH100
- [ ] Submit improved entry

### Week 3 (April 3 → April 10)
- [ ] Run autonomous overnight experiments with agent loop
- [ ] Explore depth recurrence and custom tokenizers
- [ ] Submit best result

### Weeks 4-6 (April 10 → April 30)
- [ ] Polish best techniques
- [ ] Run 3+ seed validation for statistical significance
- [ ] Final submission with full documentation

---

## Submission Requirements Checklist

Each submission PR to `openai/parameter-golf` must include:
- [ ] New folder in `/records/track_10min_16mb/YYYY-MM-DD_SubmissionName/`
- [ ] `README.md` explaining the approach in detail
- [ ] `submission.json` with name, GitHub ID, val_bpb, metadata
- [ ] `train_gpt.py` that compiles and runs within the records folder
- [ ] Train logs demonstrating statistical significance (3+ seeds, p < 0.01)
- [ ] Must beat SOTA by ≥ 0.005 nats for record entry

## Key Constraints to Remember

1. **16,000,000 bytes** (decimal) total = code + compressed model weights
2. **1500 line limit** on `train_gpt.py`
3. **10 minutes training** + **10 minutes evaluation** on 8xH100 SXM
4. **No network calls** during evaluation
5. **No pretrained models** — train from scratch only
6. **New SOTA must beat current by ≥ 0.005 nats** with p < 0.01 significance
