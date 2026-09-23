# Continual V2 experimental branch

This branch is an evidence-driven extension of mini-AGI's continual-learning
architecture. It keeps the default behaviour unchanged while exposing two
mechanisms that Experiment1 identified as plausible causes of residual
catastrophic forgetting.

## Why change the original two-rate design?

Experiment1's faithful mini-AGI replication found a large causal benefit from
slowing the globally shared trunk, but forgetting remained even when the trunk
was frozen. That means the residual interference is downstream of the trunk.

The original read path uses two optimiser timescales:

```
slow trunk | fast pool
```

but "pool" contains two very different things:

- expert content: the w1/w3/w2 transformations that store local capability;
- routing: per-token routers, depth embeddings, the segment router and gates.

Changing expert content can overwrite a capability. Changing routing can hide a
capability without overwriting it at all: an old input simply stops reaching
the expert that learned it.

V2 therefore separates the update rates:

```
slow trunk | consolidating routing | fast expert content
```

`training.router_lr_mult=1.0` exactly preserves the original rate assignment.
Lower values are experimental and must be selected from measured retention and
acquisition, not assumed to be better.

## Bounded episodic replay

The real `read` path now optionally keeps a fixed-size reservoir of previously
seen passage references. A replay mark contains only:

```
(subject, file path, byte-token offset)
```

No text is copied into the reservoir. When replay fires, an old passage is
re-opened and learned through the same full-context `FileReader` path as its
first exposure.

Replay is task-free and cross-domain. A passage from an earlier subject may be
revisited even when that subject is no longer present in the current input
directories, provided the original file still exists.

Replay is also fixed-compute: it replaces a fraction of fresh visits instead of
adding extra optimiser steps. This makes the retention/acquisition tradeoff
visible rather than buying retention with unbounded training cost.

Persistence is opt-in via `training.replay_state`. It is deliberately kept out
of `weights/`, because a replay index contains local corpus file names and
should not silently become part of a shareable checkpoint.

Defaults preserve upstream behaviour:

```yaml
training:
  trunk_lr_mult: 0.1
  router_lr_mult: 1.0
  replay: 0.0
  replay_marks: 512
  replay_state: ""
```

## Experimental order

Do not tune all knobs at once.

1. **Router causal test:** hold replay at zero and compare router LR multipliers
   at matched checkpoints and data order. The strongest isolation freezes the
   trunk and varies only router plasticity while expert content remains fast.
2. **Replay test:** once a router regime is selected, compare replay fractions
   such as 0, 0.125 and 0.25 at fixed total processed characters.
3. **Interaction:** test whether router consolidation and replay are
   complementary or redundant.
4. **Tokenizer ablation:** compare the current immutable byte vocabulary to a
   frozen subword vocabulary, normalizing evaluation to bits per raw input byte
   and matched compute. Never change token IDs online during a model's life.

The goal is a Pareto frontier over retention, new-data acquisition, held-out
loss, compute and memory. No single forgetting number is sufficient.


## Existing Experiment1 evidence: localization beats scale

Two completed mini-AGI architecture sweeps narrow the next design substantially.

### More experts are not the retention mechanism

Milestone 3A varied the expert pool from 4 to 48 experts at fixed top-k=2.
At trunk alpha=0.01 the forgetting reductions versus each architecture's
alpha=1 baseline were:

| experts | forgetting reduction |
|---:|---:|
| 4 | 74.9% |
| 8 | 76.4% |
| 12 | 76.7% |
| 24 | 77.2% |
| 48 | 75.2% |

Spearman correlation between log2(pool size) and the reduction was only 0.40,
and E=48 was 1.47 percentage points *worse* than E=12. The preregistered scale
hypothesis failed. Capacity growth may still be useful when capacity is
genuinely exhausted, but increasing expert count is not supported as a
catastrophic-forgetting fix.

### Local writes are a retention mechanism

Milestone 3B held the pool at 24 experts and changed top-k. Under alpha=0.01:

| top-k | positive forgetting | target acquisition | final mean loss |
|---:|---:|---:|---:|
| **1** | **0.0889** | 0.0919 | **2.7826** |
| 2 | 0.1472 | 0.1199 | 2.7985 |
| 4 | 0.1511 | 0.1239 | 2.7919 |
| 8 | 0.1490 | 0.1106 | 2.7927 |

Top-k=1 had 40.3% lower forgetting than top-k=8 and won all six matched
seed-by-target comparisons. The preregistered localization hypothesis passed.

The interaction matters. At alpha=1, top-k=1 and top-k=8 had nearly identical
aggregate forgetting (0.6620 vs 0.6669). Sparse routing becomes protective when
the globally shared path is already slow.

The resulting V2 hypothesis is therefore:

```
slow global writes
+ maximally local expert writes
+ consolidating routing
+ bounded replay
```

not:

```
more experts
```

For a new-model V2 experiment, `pool.top_k=1` is now the evidence-backed
localization arm. The existing default stays unchanged in this PR so old
checkpoints and upstream behavior are not silently redefined.


## Tokenizer direction: keep bytes, learn compression inside the model

mini-AGI already has a tokenizer in the literal sense: a fixed byte alphabet
plus structural markers. It does not have a learned BPE/SentencePiece
vocabulary.

For a lifetime learner, stable byte IDs are attractive because the meaning of
the input/output IDs never changes as the data distribution changes. Replacing
or refitting a subword vocabulary later would move the representation
underneath every learned embedding and output weight.

Recent byte-model results also weaken the case that subword tokenization is
necessary for efficiency:

- **Byte Latent Transformer (BLT)** uses entropy-based variable-length byte
  patches and reports tokenized-LM-level performance at scale while retaining
  raw-byte robustness.
- **H-Net** learns content- and context-dependent byte chunking jointly with the
  language model; compute/data-matched byte H-Net outperforms a strong BPE
  Transformer in its reported setting.
- **ByteFlow** uses compression-driven adaptive byte segmentation and reports
  improvements over BPE and prior byte-level baselines.

The architectural target for V2 is therefore:

```
stable byte IDs
    -> local byte encoder / learned dynamic chunking
    -> shorter latent sequence
    -> recurrent global model + experts
    -> byte decoder
```

rather than:

```
mutable external tokenizer -> model
```

A frozen BPE tokenizer remains useful as an experimental baseline. Any
tokenization comparison must be normalized to **bits per raw byte** and matched
compute; loss per token is not comparable when one arm changes the unit of
sequence length.

References:

- Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens*,
  ACL 2025: https://aclanthology.org/2025.acl-long.453/
- Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence
  Modeling*: https://arxiv.org/abs/2507.07955
- Deng et al., *ByteFlow: Language Modeling through Adaptive Byte Compression
  without a Tokenizer*, ICLR 2026:
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/eaf5d2cdb582c058a078d4fdf52a20f9-Abstract-Conference.html
