# Distributed ResNet Training (DDP) - Practical Implementation Notes

## Why this topic now

ResNet is a foundational vision architecture, and distributed data-parallel training is the production default for scaling it.
This artifact connects model understanding (residual blocks) with distributed systems realities (synchronization cost, process coordination, throughput bottlenecks).

For ML Engineer / MLOps roles, this is a high-signal combination: model + systems.

---

## 1) ResNet refresher (what matters in practice)

Core idea from *Deep Residual Learning for Image Recognition*:

- learn residual function `F(x)` and add skip path: `y = F(x) + x`
- easier optimization for deep networks
- better gradient flow compared to plain deep stacks

Operationally important:

- residual connections allow deeper models without severe degradation
- deeper model = more compute + potentially more communication under DDP

---

## 2) DDP topology and process model

Definitions:

- **world size**: total training processes
- **rank**: unique process id
- **local rank**: GPU index on node

DDP mechanics:

1. each rank holds full model replica
2. each rank gets unique mini-batch shard (`DistributedSampler`)
3. backward pass triggers gradient all-reduce
4. optimizer step stays consistent due to synchronized grads

Important implication:
- communication cost grows with model size and gradient volume

---

## 3) Correct baseline checklist

- initialize `torch.distributed` process group
- set CUDA device from local rank
- wrap model with `DistributedDataParallel`
- use `DistributedSampler` and call `set_epoch(epoch)`
- keep logging/checkpoint writes to rank 0
- destroy process group on clean exit

Without these, results are often incorrect or unstable.

---

## 4) Global batch and learning-rate scaling

Global batch:

`global_batch = per_gpu_batch * world_size`

Common practical rule:
- if global batch increases `k` times, start by scaling LR by `k` (then tune)
- add warmup for stability in larger-batch training

Watchouts:
- blindly scaling LR can diverge
- gradient accumulation may be needed when per-GPU memory is tight

---

## 5) Throughput bottlenecks in distributed ResNet

Most common bottlenecks:

1. data pipeline too slow (CPU decode/augmentation)
2. communication overhead (all-reduce) dominates step
3. sync points from frequent logging or host-device transfers
4. unbalanced workers causing stragglers

High-ROI fixes:

- mixed precision (AMP)
- larger per-rank batch if memory permits
- optimized dataloader workers + pinned memory
- reduce unnecessary synchronization in hot loop

---

## 6) Reliability and failure modes

### A) Deadlocks/hangs
Causes:
- uneven number of batches across ranks
- one rank crashes while others wait in collectives

Mitigations:
- use `DistributedSampler(drop_last=True)` for strict alignment
- add timeout-aware launch and robust error propagation

### B) Checkpoint corruption/inconsistency
Mitigations:
- only rank 0 writes checkpoints
- atomic write (temp file then rename)
- store model/optimizer/scaler/epoch metadata

### C) Numerical instability
Mitigations:
- AMP with GradScaler
- gradient clipping when needed
- monitor loss for NaN/Inf and fail fast

---

## 7) What to measure (must-have)

System metrics:

- samples/sec (per rank + global)
- step time p50/p95
- GPU utilization
- dataloader wait time
- communication overhead proxy (step time increase with more GPUs)

Model metrics:

- training/validation loss
- top-1 accuracy
- convergence speed per wall-clock hour

If speedup is poor, diagnose data/comm bottleneck before changing model depth.

---

## 8) Suggested experiment matrix

Minimal but convincing:

1. 1 GPU baseline (no DDP)
2. 2 GPU DDP, same per-GPU batch
3. 2 GPU DDP + AMP
4. compare:
- throughput gain
- final accuracy parity
- resource utilization

This gives interview-ready evidence of systems thinking.

---

## 9) Scope of this repo implementation

`train_resnet_ddp.py` intentionally keeps model/data simple so the distributed mechanics are clear and runnable on limited hardware.

Next upgrades:

- switch to torchvision ResNet18/34 on CIFAR-10 or ImageNet subset
- add profiler traces
- add multi-node launch documentation
- add experiment tracking integration (MLflow/W&B)