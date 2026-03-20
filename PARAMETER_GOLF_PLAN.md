# Parameter Golf: Competitive Analysis & Winning Strategy

## Competition Summary

**Goal:** Train the best language model that fits in **16 MB** and trains in **under 10 minutes on 8xH100 SXM GPUs**.
**Metric:** Bits-per-byte (BPB) on FineWeb validation set — lower is better.
**Deadline:** April 30, 2026.
**Compute:** OpenAI sponsoring $1M in credits; apply via [credit form](https://openai.com/index/parameter-golf/#credit-form).

---

## Competitive Landscape (March 20, 2026)

### Official Leaderboard (merged records)

| Rank | BPB    | Author      | Technique Summary |
|------|--------|-------------|-------------------|
| 1st  | 1.1748 | notapplica  | 10L + Muon WD + spectral init + resid mix + sliding eval + FP16 embed |
| 2nd  | 1.1925 | Matthew Li  | Sliding window eval (stride=64) on baseline architecture |
| 3rd  | 1.1928 | samacqua    | LoRA test-time training |
| Base | 1.2244 | OpenAI      | 9L 512d 1024vocab tied embeddings |

### Unofficial SOTA (open PRs, unmerged — the REAL competition)

| BPB       | PR   | Author          | Key Techniques |
|-----------|------|-----------------|----------------|
| **1.1318** | #198 | jfprincz       | 11L + Int6 + WD=0.04 + SWA + FA3 + BigramHash + SmearGate + OrthoInit + muP |
| **1.1453** | #180 | thwu1          | 10L + Int5-MLP/Int6-Attn + WD=0.04 + SWA/50 + SmearGate + BigramHash + OrthoInit |
| **1.1458** | #162 | raahilshah     | 9L + Int6 + MLP3x + SmearGate + BigramHash + SWA + MuonWD |
| **1.1472** | #179 | devin-cog      | 11L + Int6+zstd + decoupled WD 0.038 + sliding eval |
| **1.1480** | #194 | baudrillardsgh0st | 11L + Int6 QAT + Per-Dim SmearGate + SWA + MLP3x |
| **1.1483** | #162 | raahilshah     | 9L + Int6 MLP3x + SmearGate + BigramHash + MuonWD + SWA (updated) |
| **1.1502** | #192 | baudrillardsgh0st | 11L + Int6 QAT + SmearGate + WD 0.038 |
| **1.1507** | #206 | dexhunter      | 9L + Int6 STE + SmearGate + OrthoInit + RoPE50K + SWA/100 + NorMuon |
| **1.1532** | #173 | tamoghnokandar | Int6 + MLP3x + FA3 + NorMuon |
| **1.1598** | #191 | chris-buckley  | 9L + Compression-funded MLP3x + Int6 |
| **1.1725** | #190 | newjordan      | 9L + BigramHash + SmearGate + OrthoInit + Int6 QAT |
| **1.1792** | #205 | xinpw8         | 10L + Int5/Int6 mixed + 2% pruning + BigramHash + SmearGate |
| **1.1855** | #187 | Idan3011       | 10L (15 effective) + Pre-Enrichment + Encoder Recurrence 2x |
| **1.0238** | #168 | spokane-way    | "Paid prefix" — likely rule-bending, 8.75MB artifact |

### What's Interesting: The "Paid Prefix" (PR #168)
Spokane Way claims 1.0238 BPB with a "paid prefix" technique and an 8.75MB artifact. This likely stores a fixed context prefix within the 16MB budget that provides useful conditioning for evaluation. At 1.02 BPB this is extraordinary — either a legitimate breakthrough or a rule exploit that will be disqualified. Worth studying regardless.

---

## Technique Frequency Analysis (Top 20 PRs)

Every top-20 submission uses a combination of these building blocks:

| Technique | Usage | BPB Impact | Status |
|-----------|-------|-----------|--------|
| **Sliding window eval (stride=64)** | 19/20 PRs | ~0.030 BPB | Table stakes |
| **Int6 quantization** | 18/20 PRs | Enables +2 layers or MLP3x | Table stakes |
| **FP16 tied embeddings** | 17/20 PRs | ~0.006 BPB | Table stakes |
| **zstd-22 compression** (vs zlib) | 16/20 PRs | ~10% smaller artifacts | Table stakes |
| **SmearGate** | 14/20 PRs | ~0.005 BPB | Near-universal |
| **BigramHash** | 11/20 PRs | ~0.005-0.010 BPB | Very common |
| **OrthoInit** | 10/20 PRs | ~0.003 BPB | Common |
| **SWA (checkpoint averaging)** | 10/20 PRs | ~0.003-0.005 BPB | Common |
| **Muon WD 0.02-0.04** | 10/20 PRs | ~0.005 BPB (quant-friendly) | Common |
| **MLP 3x expansion** | 8/20 PRs | ~0.005-0.010 BPB | Funded by int6 savings |
| **11 layers** | 7/20 PRs | ~0.005-0.010 BPB | Funded by int6 savings |
| **NorMuon optimizer** | 5/20 PRs | ~0.003-0.005 BPB | Emerging |
| **Flash Attention 3** | 3/20 PRs | ~0.005 BPB (throughput) | Underutilized |
| **Seq2048+ training** | 5/20 PRs | ~0.010-0.020 BPB | Underutilized |
| **Int6 QAT (STE)** | 6/20 PRs | ~0.003 vs post-hoc quant | Growing |
| **Encoder recurrence** | 2/20 PRs | ~0.005-0.015 BPB | Novel |
| **Value embeddings** | 0/20 PRs | Unknown | **Untapped** |
| **Int5-MLP / mixed quant** | 2/20 PRs | Enables +1 layer | Novel |
| **Paid prefix** | 1/20 PRs | Potentially huge | Unverified |
| **Depth recurrence** | 2/20 PRs | Mixed results so far | Risky |

---

## Our Competitive Edge: What Nobody Has Yet

### Tier 1: Techniques from AutoResearch That Are Absent or Rare in PG

1. **Value Embeddings (ResFormer)** — 0/20 PRs use this. AutoResearch has a proven implementation with gated value residuals. This is our single biggest potential differentiator.

2. **Advanced Muon (Polar Express + NorMuon + Cautious WD)** — AutoResearch's MuonAdamW is significantly more sophisticated than what anyone is using. Only 5/20 PRs use NorMuon, none use Polar Express coefficients or cautious weight decay.

3. **Flash Attention 3 + Sliding Window Training** — Only 3/20 PRs use FA3. None use the SSSL sliding window *training* pattern. This gives us ~15-30% faster step time → more training steps.

### Tier 2: Combinations Nobody Has Tried

4. **Value Embeddings + Int6 + MLP3x** — Value embeds add ~500K-1M params. With int6+zstd compression, they might fit. Nobody has tested this.

5. **Encoder Recurrence + Value Embeddings** — PR #187 showed encoder recurrence works (15 effective layers from 10). Combining with value embeddings could be very strong.

6. **FA3 + Seq4096 + Int6 QAT** — Longer training context is proven but expensive per-step. FA3 dramatically cuts attention cost, making seq4096 viable with more steps.

---

## Winning Recipe: The "Kitchen Sink" Build

Based on the competitive landscape, here is the target configuration for #1:

### Architecture
```
Layers:          11 (funded by int6 compression)
Model dim:       512
MLP expansion:   3x (hidden=1536, funded by int6 compression)
Attention heads:  8 (4 KV heads, GQA)
Vocab size:      1024 (SentencePiece BPE)
Sequence length: 2048 (training), stride=64 (eval)
Logit softcap:   30.0
```

### Novel Components (Our Edge)
```
SmearGate:       Per-dim (512 params) — token blending
BigramHash:      4096 buckets × 128 dim — token-pair features
Value Embeddings: Gated value residual from autoresearch (alternating layers)
Pre-Enrichment:  2-layer projection before transformer stack
```

### Training
```
Optimizer:       MuonAdamW (Polar Express + NorMuon + cautious WD)
Attention:       Flash Attention 3 with SSSL sliding window
Matrix LR:       0.02
Muon WD:         0.04 (decoupled, quant-friendly)
Momentum:        0.99 (warmup from 0.92 over 1500 steps)
Batch tokens:    786,432
Warmdown:        3000 iters
Grad clip:       0.3
Int6 QAT (STE): Enabled from step 0
OrthoInit:       All large matrices
```

### Quantization & Compression
```
Quantization:    Int6 per-row (MLP + attn), FP16 (embeddings)
Compression:     zstd-22
SWA:             30 checkpoints, every 50 steps during warmdown
Target artifact: <15.5 MB
```

### Evaluation
```
Sliding window:  stride=64, seq_len=1024
Document-aware:  Respect BOS boundaries
LoRA TTT:        Rank-8, lr=0.01, chunk=256
```

### Expected BPB Breakdown

Starting from unofficial SOTA (#198 at 1.1318):

| Technique | Expected Delta | Notes |
|-----------|---------------|-------|
| Match #198 baseline | 1.1318 | 11L + Int6 + all standard techniques |
| + Value Embeddings | -0.005 to -0.015 | Novel, untested in PG |
| + Polar Express Muon | -0.002 to -0.005 | Better optimizer than anyone has |
| + FA3 sliding window training | -0.003 to -0.008 | More steps from speed gain |
| + Pre-Enrichment | -0.002 to -0.005 | Proven in PR #187 |
| **Target** | **~1.10 - 1.12** | |

---

## Implementation Plan

### Phase 1: Build the Foundation (Days 1-3)

Build a `train_gpt.py` that matches the current PR SOTA (~1.13 BPB) by combining all proven techniques:

1. Start from the current baseline `train_gpt.py`
2. Add Int6 per-row quantization with STE QAT
3. Add zstd-22 compression (replace zlib)
4. Add SmearGate (per-dim variant)
5. Add BigramHash (4096×128)
6. Add OrthoInit for all large matrices
7. Increase to 11 layers + MLP 3x
8. Add SWA (every 50 steps during warmdown)
9. Set Muon WD=0.04, momentum=0.99, LR=0.02
10. Add sliding window eval (stride=64)
11. Keep embeddings in FP16

This is "table stakes" — replicating what the best PRs already do.

### Phase 2: Add Our Differentiators (Days 4-7)

Port autoresearch innovations:

1. **Flash Attention 3** — Replace `F.scaled_dot_product_attention` with FA3 kernel calls
2. **SSSL Sliding Window Training** — Add window_pattern support during training
3. **NorMuon + Polar Express** — Port the full MuonAdamW from autoresearch
4. **Value Embeddings** — Port gated value residuals (alternating layers)
5. **Pre-Enrichment Block** — 2-layer projection before transformer

### Phase 3: Optimize & Validate (Days 8-14)

1. Run ablation studies (each component on/off)
2. Hyperparameter sweep with autoresearch agent loop
3. 3-seed validation for statistical significance
4. Prepare submission PR

### Phase 4: Exotic Ideas (Days 15+)

If Phases 1-3 don't reach target:
1. Encoder recurrence (2x forward on encoder layers)
2. Mixed int5-MLP / int6-attn quantization
3. Custom tokenizer (vocab 2048-4096)
4. "Paid prefix" — if rules allow, store useful context in the artifact

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Value embeddings don't fit in 16MB | Medium | High | Use int5 for MLP, keep VE in int6 |
| FA3 not available on 8xH100 SXM | Low | High | FA3 works on Hopper (H100 = sm_90) |
| Int6 QAT destabilizes training | Low | Medium | Delay QAT start to 25% (PR #190 trick) |
| Competition moves faster than us | High | Medium | Focus on novel techniques (VE, Muon) |
| Sliding window eval disallowed | Very Low | High | Competition FAQ explicitly allows it |
| PR rejected for code quality | Low | Medium | Follow existing submission format exactly |

---

## Compute Budget

| Phase | Hardware | Hours | Cost Estimate |
|-------|----------|-------|---------------|
| Phase 1 (foundation) | 1xH100 | 20 hrs | ~$60 |
| Phase 2 (differentiators) | 8xH100 | 30 hrs | ~$600 |
| Phase 3 (validation) | 8xH100 | 20 hrs | ~$400 |
| Phase 4 (exotic) | 8xH100 | 30 hrs | ~$600 |
| **Total** | | **100 hrs** | **~$1,660** |

---

## Key Constraints

1. **16,000,000 bytes** (decimal) total = code + compressed model weights
2. **1500 line limit** on `train_gpt.py`
3. **10 minutes training** + **10 minutes evaluation** on 8xH100 SXM
4. **No network calls** during evaluation
5. **No pretrained models** — train from scratch only
6. **New SOTA must beat current by >= 0.005 nats** with p < 0.01

## Submission Checklist

Each PR to `openai/parameter-golf` must include:
- [ ] New folder in `/records/track_10min_16mb/YYYY-MM-DD_SubmissionName/`
- [ ] `README.md` explaining the approach in detail
- [ ] `submission.json` with name, GitHub ID, val_bpb, metadata
- [ ] `train_gpt.py` that compiles and runs within the records folder
- [ ] Train logs demonstrating statistical significance (3+ seeds, p < 0.01)
- [ ] Must beat SOTA by >= 0.005 nats for record entry
