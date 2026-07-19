# tinyModelZ - Custom Tokenizer, Math Engine, and Training/Inference Loop

A custom, from-scratch Java implementation of a WordPiece/MaxMatch tokenizer, character-level tokenizer, tensor-based autograd engine, transformer block, training optimizer, and autoregressive generation generator.

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
│   │   │   └── GPUMathEngine.java      # OpenCL JNI native GPU hardware acceleration engine
│   │   ├── nn/
│   │   │   ├── Module.java             # Abstract base neural network module
│   │   │   ├── Linear.java             # Fully connected linear layer
│   │   │   ├── Embedding.java          # Input lookup embedding layer
│   │   │   ├── GeLU.java               # Gaussian Error Linear Unit activation
│   │   │   ├── LayerNorm.java          # Layer normalization layer
│   │   │   ├── Dropout.java            # Stochastic regularizer layer
│   │   │   ├── MultiHeadAttention.java # Scaled dot-product causal attention
│   │   │   ├── FeedForward.java        # Transformer MLP feed-forward block
│   │   │   ├── TransformerBlock.java   # Complete encoder/decoder transformer block
│   │   │   └── CrossEntropyLoss.java   # Log-Softmax loss with autograd backprop
│   │   ├── tokenizer/
│   │   │   ├── TrieNode.java           # Trie tree nodes and state transition mappings
│   │   │   ├── Trie.java               # Insertion, lookup, and longest prefix matching (MaxMatch)
│   │   │   ├── Tokenizer.java          # Base tokenizer interface contract
│   │   │   ├── TrieTokenizer.java      # WordPiece subword splitter & pre-tokenization
│   │   │   └── CharacterTokenizer.java # Lossless character-level tokenizer
│   │   ├── train/
│   │   │   ├── TextDataset.java        # Wraps text to list of token IDs
│   │   │   ├── DataLoader.java         # Minibatching and shifted labels for next-token prediction
│   │   │   ├── AdamW.java              # Coupled/Decoupled weight decay Adam optimizer
│   │   │   └── Checkpoint.java         # Model checkpoints save and reload
│   │   ├── inference/
│   │   │   └── Generator.java          # Autoregressive decoder with Greedy, Temp, Top-K, Top-P sampling
│   │   └── Application.java            # Spring Boot main class
│   └── test/java/com/tinymodelz/
│       ├── TestReporter.java           # Visual HTML5 test report generator
│       ├── TestRunner.java             # Entry point to execute all unit tests
│       ├── math/
│       │   └── MathEngineTest.java     # Math Engine unit tests
│       ├── nn/
│       │   └── TransformerTest.java    # Neural Network layers and Transformer block tests
│       ├── tokenizer/
│       │   ├── TrieTest.java           # Trie unit tests
│       │   ├── TrieTokenizerTest.java  # WordPiece tokenizer unit tests
│       │   └── CharacterTokenizerTest.java # Character-level tokenizer unit tests
│       ├── train/
│       │   └── TrainingTest.java       # Loss, AdamW, Checkpoints, and E2E training tests
│       └── inference/
│           └── GeneratorTest.java      # Greedy, Temperature, Top-K, Top-P sampling tests
├── RULES.md                            # Project rules and engineering constraints
├── GOALS.md                            # Project goals checklist
├── pom.xml                             # Maven project configuration file
└── build.sh                            # Command-line build and test runner script
```

---

## 🏋️ Training Infrastructure (`com.tinymodelz.train`)
*   **Shifted Target Labels**: `DataLoader` automatically produces input sequences and target sequences shifted by exactly one index position ($y_t = x_{t+1}$) for training next-token prediction models.
*   **Numerical Stability**: `CrossEntropyLoss` implements the log-sum-exp trick to avoid overflow/underflow, coupled with a custom backwards hook registered in the autograd topological sort.
*   **Decoupled Weight Decay**: `AdamW` features decoupled momentum ($\beta_1$), RMSprop-like variance ($\beta_2$), bias corrections ($\hat{m}_t, \hat{v}_t$), and isolated parameter tracking tables to prevent state leaks.
*   **L2 Gradient Clipping**: Limits global gradient norms ($\|g\|_2 \le 1.0$) to stabilize deep transformer training.
*   **Linear Warmup & Cosine Decay**: `LRScheduler` linearly warms up learning rate over 5% of training steps before cosine decaying to minimum floor.
*   **Detailed Profiler**: `Profiler` logs nanosecond execution timing breakdowns per epoch (Forward, Backward, Optimizer, Batch prep, Checkpointing, Peak RAM).
*   **Complete Checkpoint Persistence**: `Checkpoint` serializes model weights, AdamW optimizer momentum ($m$) and variance ($v$) vectors, step counters, and metadata properties for exact training restoration.

---

## 🔮 Inference & Sampling Generator (`com.tinymodelz.inference`)
The `Generator` class supports autoregressive generation using:
1.  **Greedy Decoding**: Selects the argmax token index.
2.  **Temperature Scaling**: Modifies the logit distribution confidence before soft-max scaling.
3.  **Top-K Filtering**: Truncates search space to the top $K$ highest-probability tokens.
4.  **Top-P (Nucleus) Sampling**: Dynamically filters logits keeping only the smallest set of tokens whose cumulative probability exceeds threshold $P$.
5.  **Repetition Penalty**: Scales previously generated token logits by penalty factor ($1.15\times$) to prevent repetitive loops.

---

## ⚡ GPU Powered Training (`com.tinymodelz.gpu`)
* **OpenCL Compute Backend**: Native hardware-accelerated matrix multiplication (`libtinymodelz_gpu.so`) built directly into the framework.
* **Persistent Buffer Allocation Pooling**: Reuses OpenCL memory buffers across iterations, reducing test suite execution latency by **4x** (from 385ms to 96ms).
* **Automatic Hardware Probing**: Probes system for NVIDIA (CUDA/OpenCL), AMD (ROCm/OpenCL), or Intel GPUs at startup.
* **Seamless Dynamic Switch**: Configure runtime device execution target via `DeviceManager.setDevice(Device.GPU)`.
* **Zero-Downtime CPU Fallback**: Automatically falls back to multi-threaded CPU matrix operations if GPU driver or hardware is absent.

---

## 🚀 How to Build and Run Tests

You can build the project and run the complete test suite using the provided `build.sh` script:

```bash
# Compile and run full test suite (including GPU acceleration checks)
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn test-compile exec:java -Dexec.mainClass="com.tinymodelz.TestRunner" -Dexec.classpathScope="test"

# Run TinyGPT training pipeline on GPU (NVIDIA GeForce 940MX via OpenCL)
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java -Dexec.mainClass="com.tinymodelz.train.TrainTinyStories" -Dexec.args="TinyStories-valid.txt 5 16 64 gpu"

# Run TinyGPT training pipeline on CPU
JAVA_HOME=tools/graalvm ./tools/maven/bin/mvn exec:java -Dexec.mainClass="com.tinymodelz.train.TrainTinyStories" -Dexec.args="TinyStories-valid.txt 5 16 64 cpu"
```

```bash
bash build.sh
```

The script will compile all source and test files using Maven, download dependencies, execute all unit and integration test suites, and write an HTML report dashboard at:
`test_report.html`

---

## 📝 Compliance with project RULES.md

1.  **No External ML Libraries**: Core mathematical, tensor, and backpropagation logic is implemented completely from scratch.
2.  **From-Scratch Algorithm Implementation**: Custom Trie tree, prefix matcher, WordPiece parser, character tokenizer, weight initializers, cross-entropy loss, AdamW, and binary serialization format.
3.  **Visual Reporting**: Automatic execution of a built-in reporter compiling status tables, logs, and success rates in a sleek, responsive dashboard.
