# 🎯 TinyModelZ Project Goals & Roadmap

---

## 🧮 Math & GPU Acceleration Engine
- [x] Matrix operations (2D dot product, transpose, bias vector addition)
- [x] Vector operations (1D dot product, norm, softmax)
- [x] Random weight initializers (Xavier / He normal & uniform)
- [x] File serialization (binary TMA1 / TMAT tensor persistence)
- [x] OpenCL GPU Hardware Acceleration (`DeviceManager`, `GPUMathEngine`, `libtinymodelz_gpu.so`, automatic CPU fallback)
- [x] Persistent OpenCL memory buffer allocation pooling ($4.1\times$ speedup, reducing GPU suite runtime from 385ms to 93ms)
- [x] Tiled local memory OpenCL matrix multiplication kernels

## 🧠 Tensor Engine & Autograd
- [x] N-Dimensional Tensors with shape, stride, and offset tracking
- [x] Contiguous memory layout conversions (`toContiguous`)
- [x] Implicit and explicit tensor broadcasting
- [x] Automatic differentiation (Autograd computation graph with backward pass gradient accumulation)

## 🕸️ Neural Network Modules
- [x] Abstract base `Module` parameter management
- [x] Fully connected `Linear` layer
- [x] Lookup `Embedding` layer
- [x] Activation functions (`GeLU`, `ReLU`, `Sigmoid`, `Tanh`)
- [x] `LayerNorm` (Layer Normalization with learnable scale $\gamma$ and shift $\beta$)
- [x] `Dropout` stochastic regularizer layer

## 🤖 Transformer Architecture
- [x] Scaled dot-product causal self-attention
- [x] `MultiHeadAttention` with parallel query, key, and value linear projections
- [x] Position-wise Feed-Forward Network (`FeedForward`)
- [x] Residual skip connections and Pre-LN normalization strategy
- [x] Transformer Decoder block (`TransformerBlock`)
- [x] Complete `TinyGPT` Model ($Token + Positional Embedding \rightarrow N \times TransformerBlock \rightarrow LayerNorm \rightarrow LM Head$)

## 🔤 Tokenizer Subsystem
- [x] Character-level tokenizer (`CharacterTokenizer`)
- [x] Trie-based subword tokenizer (`TrieTokenizer` with WordPiece MaxMatch algorithm)
- [x] **Byte-Pair Encoding (BPE) Tokenizer** (`BPETokenizer.java` with subword pair merging and special control tokens)
- [x] **Modular Tokenizer Factory & Serialization** (`TokenizerFactory.java` supporting dynamic strategy selection and lifecycle management)

## 🏋️ Training Infrastructure & Optimization
- [x] `TextDataset` text-to-token stream wrapper
- [x] `DataLoader` mini-batching with shifted next-token target labels ($y_t = x_{t+1}$)
- [x] `CrossEntropyLoss` with log-sum-exp numerical stabilization and autograd backward pass
- [x] `AdamW` optimizer with decoupled weight decay, momentum ($\beta_1$), RMSprop variance ($\beta_2$), and bias correction
- [x] L2 Gradient Clipping (`Trainer.clipGradients`) for global gradient norm stabilization ($\le 1.0$)
- [x] `LRScheduler` with 5% Linear Warmup, Cosine Annealing decay, and step synchronization upon resume
- [x] Granular nanosecond execution profiler (`Profiler.java` tracking Forward, Backward, Optimizer, Batch Prep, Checkpointing, and Peak RAM)
- [x] Validation evaluation pass tracking validation loss and perplexity
- [x] **Structured Timestamped Checkpoint Subdirectories** (`checkpoints/<dataset>_<tokenizer>_<timestamp>/`) retaining model runs and `run_info.properties`
- [x] **Lossless Training Resumption** (`Checkpoint.java` & `TrainTinyStories` restoring parameters, AdamW $m,v$ state vectors, step counts, and starting from specific epochs)

## 🔮 Autoregressive Inference & Sampling
- [x] Argmax Greedy decoding
- [x] Temperature logit scaling
- [x] Top-K logit filtering
- [x] Top-P (Nucleus) cumulative probability sampling
- [x] Repetition Penalty ($1.15\times$ logit scaling) to prevent token loops
- [x] **Key-Value (KV) Cache** (`MultiHeadAttention.KVCache`) for fast autoregressive inference
- [x] **Rotary Position Embeddings (RoPE)** (`RotaryEmbedding.java`) for relative positional encoding

## 🖥️ API, CLI & GraalVM Native Image
- [x] Interactive terminal prompt runner & run discovery tool (`PromptRunner` listing available checkpoints and loading target tokenizers)
- [x] CLI argument parser for training & resumption (`TrainTinyStories --tokenizer --resume --start-epoch`)
- [x] Spring Boot REST API endpoints (`/api/generate`, `/api/tokenize`)
- [x] Interactive web visualizer UI
- [x] GraalVM Native Image compilation (<85MB RSS, sub-100ms cold startup)

## 📦 CI/CD & Version Control
- [x] **Semantic Versioning** (`v1.0.0` SemVer compliance in `pom.xml` & `Version.java`)
- [x] **GitHub Actions Continuous Integration** (`.github/workflows/test.yml` running unit tests on PRs & main push)
- [x] **GitHub Actions Release Publisher** (`.github/workflows/release.yml` building JARs, native `.so`, and creating GitHub Releases on `v*.*.*` version tags)

---

## 🔮 Future Scaling & Architectural Upgrades (10M–100M Parameters)
- [ ] **FlashAttention OpenCL/CUDA/Metal Kernel**: Tiled online softmax attention reducing memory complexity from $O(N^2)$ to $O(N)$
- [ ] **Mixed Precision Training (FP16 / BF16)**: Half-precision matrix multiplication for $2\times$ memory throughput
- [ ] **Distributed Multi-GPU Training**: Data parallel training across multiple OpenCL/CUDA/Metal devices