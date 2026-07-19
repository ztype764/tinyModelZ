Your task is to perform a complete code review of the training pipeline and implement every improvement that is beneficial WITHOUT changing the overall architecture.

Do not rewrite the framework.
Do not introduce PyTorch, TensorFlow, DJL or HuggingFace.
Keep everything implemented from scratch.

----------------------------------------
PHASE 1 – PROFILE THE TRAINER
----------------------------------------

Implement a detailed profiler.

Measure separately:

- Dataset loading
- Tokenization
- Batch preparation
- Host → GPU transfer
- GPU kernel execution
- GPU → Host transfer
- Forward pass
- Attention
- Feed Forward
- Embedding lookup
- Softmax
- CrossEntropy
- Backward pass
- Optimizer
- Checkpoint writing

Print a timing summary after every epoch.

----------------------------------------
PHASE 2 – GPU PERFORMANCE
----------------------------------------

Review every OpenCL kernel.

Improve:

- Global memory access
- Local memory usage
- Kernel launch sizes
- Workgroup sizes
- Memory coalescing
- Vectorized loads
- Buffer reuse

Avoid unnecessary GPU allocations.

Avoid unnecessary GPU synchronization.

Avoid CPU↔GPU transfers unless absolutely necessary.

The entire forward pass, backward pass and optimizer should remain on GPU whenever possible.

----------------------------------------
PHASE 3 – MEMORY OPTIMIZATION
----------------------------------------

Review the Tensor engine.

Reduce:

- temporary Tensor creation
- unnecessary cloning
- repeated allocations
- garbage generation

Implement reusable tensor pools where appropriate.

Reuse GPU buffers whenever possible.

----------------------------------------
PHASE 4 – TRAINING IMPROVEMENTS
----------------------------------------

Implement:

✓ Gradient clipping

✓ Learning rate scheduler
    - cosine decay
    - warmup

✓ Validation pass

✓ Best checkpoint saving

✓ Early stopping (optional)

✓ Resume training automatically

----------------------------------------
PHASE 5 – TRAINING METRICS
----------------------------------------

Improve training logs.

Every epoch print:

Epoch
Average Loss
Validation Loss
Perplexity
Learning Rate
Tokens/sec
Samples/sec
GPU compute time
GPU utilization (if available)
Peak RAM
Peak GPU memory
Checkpoint path
Elapsed time
ETA

----------------------------------------
PHASE 6 – INFERENCE
----------------------------------------

Improve generation.

Implement configurable:

Temperature

Top-K

Top-P

Repetition penalty

Maximum generation length

Streaming generation

Generation statistics

----------------------------------------
PHASE 7 – MODEL QUALITY
----------------------------------------

Review TinyGPT implementation.

Verify:

✓ Causal mask correctness

✓ LayerNorm

✓ Attention scaling

✓ Weight initialization

✓ Dropout placement

✓ Optimizer correctness

✓ Loss implementation

✓ Backprop correctness

Report any mathematical issues found.

----------------------------------------
PHASE 8 – TOKENIZER
----------------------------------------

Character tokenizer remains for testing.

Prepare the project for Byte Pair Encoding.

Do NOT remove the current tokenizer.

Instead create interfaces allowing multiple tokenizer implementations.

----------------------------------------
PHASE 9 – CHECKPOINTS
----------------------------------------

Improve checkpoint format.

Store:

Model weights

Optimizer state

Learning rate

Epoch

Global step

Vocabulary

Tokenizer metadata

Random seed

Allow exact training resume.

----------------------------------------
PHASE 10 – TESTING
----------------------------------------

Add tests for:

Gradient correctness

Numerical stability

GPU vs CPU consistency

Checkpoint restore

Generation consistency

Loss convergence

Tokenizer encode/decode

Dataset correctness

----------------------------------------
PHASE 11 – CODE QUALITY
----------------------------------------

Review the complete project.

Identify:

Dead code

Duplicate code

Inefficient algorithms

Memory leaks

GPU synchronization issues

OpenCL resource leaks

Race conditions

Potential NaN generation

Integer overflow

Floating point instability

Improve documentation.

Every major class should contain JavaDoc explaining:

Purpose

Math

Tensor shapes

Time complexity

Memory complexity

----------------------------------------
PHASE 12 – FINAL REPORT
----------------------------------------

After completing all improvements produce a report containing:

Implemented improvements

Performance gains

Remaining bottlenecks

Future optimization opportunities

Recommended roadmap toward:

10M parameter model

100M parameter model

BPE tokenizer

RoPE

KV Cache

Flash Attention

Mixed Precision

Distributed training

Do NOT change APIs unless necessary.

Explain every significant optimization before implementing it.

Prioritize correctness over speed.

Keep TinyModelZ educational, readable and modular.
