# tinyModelZ - Custom Tokenizer, Math Engine, and Training/Inference Loop

A custom, from-scratch Java implementation of a WordPiece/MaxMatch tokenizer, BPE tokenizer, character-level tokenizer, tensor-based autograd engine, transformer block with KV Cache and RoPE, training optimizer with checkpoint resumption, and autoregressive text generator.

---

## 📁 Project Directory Structure

```
tinyModelZ/
├── src/
│   ├── main/java/com/tinymodelz/
│   │   ├── math/
│   │   │   ├── Vector.java             # 1D vector operations (norm, dot, softmax)
│   │   │   ├── Matrix.java             # 2D matrix operations (dot product, transpose, bias, GPU dispatch)
│   │   │   ├── RandomInitializer.java  # Xavier/He parameter initializers
│   │   │   ├── Tensor.java             # N-dimensional autograd tensor with shape/stride tracking
│   │   │   ├── Device.java             # Compute device target enum (CPU, GPU)
│   │   │   ├── DeviceManager.java      # Hardware compute device state manager
│   │   │   └── MathIO.java             # TMA1/TMAT binary weight saving/loading
│   │   ├── gpu/
│   │   │   ├── GPUMathEngine.java      # OpenCL JNI native GPU hardware acceleration engine
│   │   │   └── CUDAMathEngine.java     # CUDA Driver PTX native acceleration engine
│   │   ├── nn/
│   │   │   ├── Module.java             # Abstract base neural network module
│   │   │   ├── Linear.java             # Fully connected linear layer
│   │   │   ├── Embedding.java          # Input lookup embedding layer
│   │   │   ├── GeLU.java               # Gaussian Error Linear Unit activation
│   │   │   ├── LayerNorm.java          # Layer normalization layer
│   │   │   ├── Dropout.java            # Stochastic regularizer layer
│   │   │   ├── MultiHeadAttention.java # Scaled dot-product causal attention
│   │   │   ├── KVCache.java            # Key-Value cache for $O(1)$ decoding
│   │   │   ├── RotaryEmbedding.java    # Rotary Position Embeddings (RoPE)
│   │   │   ├── FeedForward.java        # Transformer MLP feed-forward block
│   │   │   ├── TransformerBlock.java   # Complete encoder/decoder transformer block
│   │   │   └── CrossEntropyLoss.java   # Log-Softmax loss with autograd backprop
│   │   ├── tokenizer/
│   │   │   ├── Tokenizer.java          # Base tokenizer interface contract
│   │   │   ├── TokenizerFactory.java   # Dynamic factory for Character, BPE, and Trie tokenizers
│   │   │   ├── CharacterTokenizer.java # Lossless character-level tokenizer
│   │   │   ├── BPETokenizer.java       # Byte-Pair Encoding subword tokenizer
│   │   │   ├── TrieTokenizer.java      # WordPiece subword splitter & pre-tokenization
│   │   │   └── Trie.java               # Insertion, lookup, and longest prefix matching (MaxMatch)
│   │   ├── train/
│   │   │   ├── TextDataset.java        # Wraps text to list of token IDs
│   │   │   ├── DataLoader.java         # Minibatching and shifted labels for next-token prediction
│   │   │   ├── Trainer.java            # Training loop with validation & gradient clipping
│   │   │   ├── TrainTinyStories.java   # CLI entry point for full pipeline execution & resumption
│   │   │   ├── AdamW.java              # Decoupled weight decay Adam optimizer
│   │   │   ├── LRScheduler.java        # Warmup and Cosine Annealing scheduler with step sync
│   │   │   └── Checkpoint.java         # Model checkpoints save, load, and resumption
│   │   ├── inference/
│   │   │   ├── Generator.java          # Autoregressive decoder with Greedy, Temp, Top-K, Top-P sampling
│   │   │   └── PromptRunner.java       # Interactive CLI prompt runner with run & epoch selection
│   │   └── Application.java            # Spring Boot main class
│   └── test/java/com/tinymodelz/
│       ├── TestReporter.java           # Visual HTML5 test report generator
│       ├── TestRunner.java             # Entry point to execute all unit tests
│       ├── tokenizer/
│       │   ├── CharacterTokenizerTest.java
│       │   ├── BPETokenizerTest.java
│       │   └── TrieTokenizerTest.java
│       ├── train/
│       │   ├── TrainingTest.java
│       │   └── TokenizerAndResumeTest.java # Multi-tokenizer & epoch 2 resumption unit tests
│       └── nn/
│           └── KVCacheTest.java
├── checkpoints/                        # Model run subdirectories
│   └── tinystories/                    # Base directory for TinyStories training runs
│       └── <run_name>/                 # Isolated run folder (e.g. TinyStories_valid_reduced_bpe_20260720_234500)
│           ├── epoch_1/                # Per-epoch model checkpoints
│           ├── epoch_2/
│           ├── run_info.properties     # Training metadata (dataset, tokenizer, hyperparameters)
│           ├── tokenizer_config.properties
│           ├── vocab.txt               # Serialized vocabulary
│           └── bpe_merges.txt          # BPE merge rules (for BPE tokenizer)
├── RULES.md                            # Project rules and engineering constraints
├── GOALS.md                            # Project goals checklist
├── pom.xml                             # Maven project configuration file
└── build.sh                            # Command-line build and test runner script
```

---

## 🔤 Modular Tokenizer Infrastructure (`com.tinymodelz.tokenizer`)
TinyModelZ includes a centralized `TokenizerFactory` supporting 3 tokenizer types:
1. **BPETokenizer (`bpe`)**: Byte-Pair Encoding subword algorithm with iterative merge ranking and special control token support (`<|endoftext|>`). Default for training.
2. **CharacterTokenizer (`character`)**: Lossless character-level encoding with automatic character set extraction.
3. **TrieTokenizer (`trie`)**: WordPiece subword tokenizer using a Trie prefix tree for MaxMatch greedy tokenization.

The factory handles serialization and re-loading of tokenizer configurations (`tokenizer_config.properties`, `vocab.txt`, `bpe_merges.txt`) into checkpoint directories.

---

## 🏋️ Training Infrastructure & Checkpoint Resumption (`com.tinymodelz.train`)
*   **Default Tokenizer**: Training pipeline uses **BPE** (`bpe`) by default.
*   **Isolated Run Subdirectories**: For each non-resumed training run, a dedicated subfolder is created inside `checkpoints/tinystories/<dataset>_<tokenizer>_<timestamp>/`.
*   **Shifted Target Labels**: `DataLoader` automatically produces input sequences and target sequences shifted by exactly one index position ($y_t = x_{t+1}$) for training next-token prediction models.
*   **Decoupled Weight Decay**: `AdamW` features decoupled momentum ($\beta_1$), RMSprop-like variance ($\beta_2$), bias corrections ($\hat{m}_t, \hat{v}_t$), and isolated parameter tracking tables.
*   **L2 Gradient Clipping**: Limits global gradient norms ($\|g\|_2 \le 1.0$) to stabilize deep transformer training.
*   **Linear Warmup & Cosine Decay**: `LRScheduler` linearly warms up learning rate over 5% of training steps before cosine decaying to minimum floor, with support for step synchronization during training resumption.
*   **Robust Training Resumption**: Allows interrupting and restarting training at any epoch (e.g. resuming from epoch 2 onwards). Re-loads model parameters, optimizer momentum/variance, and total step count automatically.

---

## 🔮 Inference & Sampling Generator (`com.tinymodelz.inference`)
The `Generator` class supports autoregressive generation using:
1.  **Key-Value (KV) Cache**: Caches Key and Value attention matrices for $O(1)$ decoding throughput.
2.  **Greedy Decoding**: Selects the argmax token index.
3.  **Temperature Scaling**: Modifies logit distribution entropy before soft-max scaling.
4.  **Top-K Filtering**: Truncates search space to top $K$ probability tokens.
5.  **Top-P (Nucleus) Sampling**: Dynamically selects tokens exceeding cumulative probability threshold $P$.
6.  **Repetition Penalty**: Applies $1.15\times$ logit scaling to previously generated tokens to prevent repetitive loops.

---

## ⚡ GPU Powered Training (`com.tinymodelz.gpu`)
* **OpenCL & CUDA Backends**: Native hardware acceleration via OpenCL (`libtinymodelz_gpu.so`) or NVIDIA Driver API (`libtinymodelz_cuda.so`).
* **Persistent Buffer Pooling**: Reuses memory buffers across iterations, delivering a **4x speedup** on matrix operations.
* **Automatic Hardware Probing**: Detects GPUs at startup with zero-downtime multi-threaded CPU fallback.

---

## 🤖 Interactive & CLI Prompt Runner (`PromptRunner`)

`PromptRunner` scans `checkpoints/tinystories/` for saved runs and enables both interactive and CLI selection of training data/runs and specific epoch checkpoints:

### 1. Interactive Selection Mode
Simply launch `PromptRunner` without arguments:
```bash
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java \
  -Dexec.mainClass="com.tinymodelz.inference.PromptRunner"
```
You will be prompted to:
1. Select which **Training Run / Dataset** to use.
2. Select which **Epoch Checkpoint** (`epoch_1`, `epoch_2`, `best_checkpoint`, etc.) to load.

### 2. CLI Direct Selection & Listing Mode
```bash
# List all available runs and epoch checkpoints
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java \
  -Dexec.mainClass="com.tinymodelz.inference.PromptRunner" \
  -Dexec.args="--list"

# Select training run by index/name and epoch by number/name
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java \
  -Dexec.mainClass="com.tinymodelz.inference.PromptRunner" \
  -Dexec.args="--run 1 --epoch 3 --prompt 'Once upon a time' --device cuda"
```

---

## 🚀 Commands & Detailed Usage Guide

### 🧪 1. Running the Test Suite
Run all unit and end-to-end integration tests (Tokenizers, Autograd, Layers, Checkpoints, Resumption, OpenCL & CUDA GPU precision checks):
```bash
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn test-compile exec:java \
  -Dexec.mainClass="com.tinymodelz.TestRunner" \
  -Dexec.classpathScope="test"
```
* **HTML Dashboard Report**: Generates an interactive test report at `test_report.html`.

---

### 🏋️ 2. Training TinyGPT Models (`TrainTinyStories`)

#### Standard Training Run (BPE Tokenizer Default)
```bash
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java \
  -Dexec.mainClass="com.tinymodelz.train.TrainTinyStories" \
  -Dexec.args="--dataset TinyStories-valid-reduced.txt --tokenizer bpe --epochs 5 --batch-size 16 --seq-len 64 --device cuda"
```

#### Resuming Training from Epoch 2 (or latest epoch checkpoint)
```bash
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java \
  -Dexec.mainClass="com.tinymodelz.train.TrainTinyStories" \
  -Dexec.args="--resume checkpoints/tinystories/TinyStories_valid_reduced_bpe_20260720_234500 --epochs 5 --device cuda"
```

#### CLI Option Flags:
* `--tokenizer [bpe|character|trie]`: Tokenization algorithm (default: `bpe`)
* `--resume [folder]`: Path to existing run folder or specific `epoch_N` directory to resume training
* `--start-epoch [N]`: Explicitly set starting epoch index when resuming
* `--device [cuda|opencl|cpu]`: Execution target hardware engine

---

## 📝 Compliance with project RULES.md

1. **No External ML Libraries**: Core mathematical, tensor, tokenizer, and backpropagation logic is built completely from scratch.
2. **From-Scratch Algorithm Implementation**: Custom Trie tree, BPE merger, character tokenizer, weight initializers, cross-entropy loss, AdamW, and binary serialization format.
3. **Visual Reporting**: Automatic generation of HTML test reports (`test_report.html`).
