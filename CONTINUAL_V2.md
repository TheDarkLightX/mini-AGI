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
