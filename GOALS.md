# 🎯 Project Goals & Roadmap

## 🧮 Math Engine
- [x] Matrix operations
- [x] Vector operations
- [x] Random initialization
- [x] File serialization

## 🧠 Tensor Engine
- [x] Tensors
- [x] Broadcasting
- [x] Basic operations
- [x] (Optional) Automatic differentiation

## 🕸️ Neural Network
- [x] Linear layer
- [x] Embeddings
- [x] Activation functions
- [x] LayerNorm
- [x] Dropout

## 🤖 Transformer
- [x] Self-attention
- [x] Multi-head attention
- [x] Feed-forward network
- [x] Residual connections
- [x] Decoder block

## 🔤 Tokenizer
- [x] Character tokenizer
- [x] Trie-based subword tokenizer (WordPiece)

## 🏋️ Training
- [x] Dataset loader
- [x] Batching
- [x] Cross-entropy loss
- [x] Adam optimizer (AdamW style with momentum & weight decay)
- [x] Checkpoints (binary TMAT serialization)

## 🔮 Inference
- [x] Greedy decoding
- [x] Temperature
- [x] Top-k and Top-p sampling

## 🖥️ CLI/Web API
- [x] Prompt input & Interactive UI visualizer
- [x] Generate response (REST API endpoints for tokenization and model execution)
- [x] GraalVM Native Image compilation for low memory usage (<85MB RSS) and sub-100ms startup