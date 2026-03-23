# Distributed Word2Vec (SGNS) Implementation Notes

## Goal

Implement a practical, interview-ready version of **Word2Vec Skip-gram with Negative Sampling (SGNS)** that explicitly models distributed training constraints.

This note focuses on system design decisions, not just model equations.

---

## 1) Problem framing

Classic SGNS is simple on one machine but expensive at scale because:

- embedding tables are huge (vocab can be millions)
- updates are sparse but frequent
- communication dominates compute in distributed settings

For ML Engineer/MLOps roles, the key is balancing **throughput**, **convergence quality**, and **operational simplicity**.

---

## 2) SGNS recap (minimal math)

For center word `w` and context word `c`, maximize:

- `log sigma(v_w · u_c)` for positive pairs
- `sum_k log sigma(-v_w · u_nk)` for negative samples

where:

- `v_*` are input embeddings
- `u_*` are output embeddings

Properties:

- sparse row updates (only touched tokens update)
- highly parallelizable
- communication hotspots on frequent words

---

## 3) Distributed design options

## A) Parameter server (PS), async updates

How it works:

- shard embedding rows across parameter servers
- workers pull required rows, compute gradients, push updates asynchronously

Pros:

- high throughput
- naturally supports sparse updates
- easy horizontal scaling for workers

Cons:

- stale gradients (workers read old params)
- hotspot shards for common tokens
- convergence sensitivity to lag/skew

Best when:

- latency/throughput prioritized
- slight optimization noise is acceptable

## B) Synchronous all-reduce

How it works:

- each worker computes gradients on local batch
- global gradient synchronization every step (or micro-step)

Pros:

- cleaner optimization semantics
- deterministic-ish across runs
- easier debugging of convergence

Cons:

- expensive synchronization barriers
- inefficient for ultra-sparse updates unless heavily optimized
- straggler sensitivity

Best when:

- model quality reproducibility matters most
- cluster/network is stable and high-bandwidth

---

## 4) Recommendation for Word2Vec at production scale

For SGNS-style sparse embedding training, start with:

- **sharded parameter server + bounded staleness**
- token-frequency-aware partitioning
- adaptive learning rate per shard
- optional periodic sync checkpoints to reduce drift

Reason: sparse updates benefit more from asynchronous sharded access than strict step-level all-reduce.

---

## 5) Data and sampling pipeline

Pipeline stages:

1. tokenize corpus and build vocab with min-frequency threshold
2. subsample very frequent words (`t / f(w)` heuristic)
3. generate (center, context) pairs via sliding window
4. negative sample by smoothed unigram (`f(w)^0.75`)
5. dispatch batches to workers

Distributed concerns:

- keep negative sampler statistically consistent across workers
- cache alias tables per worker and refresh periodically
- track per-worker token skew to detect imbalance

---

## 6) Sharding strategy details

Base strategy:

- consistent hash by token id -> shard

Improvements:

- separate heavy hitters into dedicated shard classes
- replicate top-K frequent output rows for read-heavy paths
- use asynchronous write coalescing for frequent tokens

Watch metrics:

- per-shard QPS
- queue depth
- update staleness window
- hot-token concentration

---

## 7) Correctness and quality risks

- stale reads create optimizer noise
- delayed updates can overfit local minibatch distribution
- uneven shard load slows tail workers
- negative sampler drift changes effective objective

Mitigations:

- bounded staleness limit
- gradient clipping
- periodic global eval on analogy/similarity set
- shard rebalancing by observed load

---

## 8) What to measure (must-have)

System metrics:

- tokens/sec per worker
- network bytes/sec per shard
- p95 pull/push latency
- staleness distribution (steps behind)

Model metrics:

- training loss trend
- nearest-neighbor quality checks
- downstream task proxy (if available)

Decision trigger example:

- if throughput high but quality plateaus early, reduce staleness or increase sync frequency.

---

## 9) What this folder contains

- `sgns_distributed.py`
- compact SGNS skeleton
- sharded embeddings
- pluggable sync modes: `async_ps`, `sync_allreduce`
- simple instrumentation for update counts and remap skew

Use this as a systems-thinking artifact, not a final production trainer.