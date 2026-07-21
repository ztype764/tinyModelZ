# TinyModelZ GPU-Only Execution Mode Design

## Objective

Implement a new execution mode named **GPU_ONLY** where the complete
forward pass, backward pass, optimizer update, and tensor lifecycle
remain entirely on the GPU.

The CPU should only: - Load and tokenize datasets. - Transfer each batch
to the GPU once. - Start the training step. - Read back scalar metrics
(loss, perplexity, learning rate). - Save checkpoints when required.

Everything else should remain resident on the GPU.

## 1. Execution Modes

``` java
enum ExecutionMode {
    CPU,
    HYBRID,
    GPU_ONLY
}
```

Current behavior remains **HYBRID**.

## 2. Device-Resident Tensors

Each Tensor should maintain: - Device - Host buffer (optional) - GPU
buffer - Dirty flags

Expose:

``` java
tensor.toGPU();
tensor.toCPU();
tensor.isOnGPU();
tensor.isDirtyOnGPU();
```

Use lazy synchronization.

## 3. GPU Computation Graph

Execute entirely on GPU:

-   Embedding
-   RoPE
-   QKV
-   Attention
-   Softmax
-   Output projection
-   LayerNorm
-   FeedForward
-   CrossEntropy
-   Backward

No gradient copies to CPU.

## 4. GPU AdamW

Keep weights, gradients, first/second moments on GPU.

Optimizer updates should execute entirely on CUDA/OpenCL.

## 5. GPU CrossEntropy

Compute loss on GPU.

Return only scalar loss.

## 6. Persistent GPU Memory

Allocate once: - Model parameters - Optimizer state - Activation buffers

Reuse buffers and avoid repeated allocations.

## 7. Minimize Transfers

Only transfer: - Input batch → GPU - Loss scalar ← GPU

Everything else remains on GPU.

## 8. Kernel Fusion

Fuse operations where practical: - LayerNorm + Linear - Linear + Bias -
Bias + GeLU - Residual + LayerNorm - Attention pipeline

## 9. Reduce JNI Overhead

Batch operations into larger JNI calls.

Avoid one JNI call per tensor operation.

## 10. Asynchronous Execution

Synchronize only for: - Loss retrieval - Checkpoints - Inference
output - Explicit sync

## 11. Profiling

Report: - CPU prep - Host→Device - Device→Host - Kernel execution -
Synchronization - JNI overhead - Optimizer - Checkpoint

## 12. Benchmarks

Compare: - CPU - OpenCL Hybrid - CUDA Hybrid - OpenCL GPU_ONLY - CUDA
GPU_ONLY

Metrics: - Tokens/sec - GFLOPS - Kernel launches - GPU utilization -
VRAM - Transfer time - JNI overhead

## 13. Checkpointing

Remain backward compatible.

Resume directly in GPU_ONLY mode.

## 14. Portability

Support: - CPU - OpenCL - CUDA

GPU_ONLY must work for CUDA and OpenCL.

## 15. Target Architecture

``` text
Batch
  │
  ▼
Upload once
  │
  ▼
GPU
  Embedding
  RoPE
  Transformer
  LayerNorm
  MLP
  CrossEntropy
  Backward
  AdamW
  │
  ▼
Return loss only
```

## Success Criteria

-   Minimize host-device transfers.
-   Minimize JNI crossings.
-   Maximize GPU residency.
-   Reuse GPU memory.
-   Preserve numerical correctness.
-   Keep all existing tests passing.
-   Improve training throughput.
