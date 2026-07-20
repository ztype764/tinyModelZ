# 📚 Loading and Preparing External Datasets in TinyModelZ

This guide explains how to prepare, load, tokenize, and batch external text datasets for training TinyModelZ language models.

---

## 🛠️ Step 1: Pre-process Raw Text Data
Before feeding text data to the tokenizer, it is recommended to clean the input (e.g., standardizing line breaks, resolving special character representations, etc.). 

For a simple character-level or subword tokenizer, read the file as a UTF-8 string:

```java
import java.nio.file.Files;
import java.nio.file.Path;

String rawText = Files.readString(Path.of("path/to/dataset.txt"));
```

---

## 🔤 Step 2: Initialize the Tokenizer
You can construct and initialize tokenizers using `TokenizerFactory`, supporting Byte-Pair Encoding (`BPE`), Character-level (`CHARACTER`), or WordPiece (`TRIE`).

### Option A: Byte-Pair Encoding (BPE) Tokenizer (Recommended)
Automatically extracts character base vocabularies, learns merge frequencies, and preserves special control tokens like `<|endoftext|>`.

```java
import com.tinymodelz.tokenizer.TokenizerFactory;
import com.tinymodelz.tokenizer.TokenizerFactory.TokenizerType;
import com.tinymodelz.tokenizer.Tokenizer;
import java.util.List;

List<String> specialTokens = List.of("<|endoftext|>");
Tokenizer tokenizer = TokenizerFactory.createTokenizer(TokenizerType.BPE, rawText, specialTokens);
```

### Option B: Character-level Tokenizer
Lossless character-level tokenizer extracting unique characters from the corpus.

```java
Tokenizer tokenizer = TokenizerFactory.createTokenizer(TokenizerType.CHARACTER, rawText, specialTokens);
```

### Option C: WordPiece Trie Tokenizer
Trie prefix tree based tokenizer for MaxMatch subword tokenization.

```java
Tokenizer tokenizer = TokenizerFactory.createTokenizer(TokenizerType.TRIE, rawText, specialTokens);
```

---

## 🏋️ Step 3: Wrap in TextDataset and DataLoader
The `TextDataset` maps the text to token IDs, while the `DataLoader` manages mini-batching, random shuffling, and shifted next-token prediction targets.

```java
import com.tinymodelz.train.TextDataset;
import com.tinymodelz.train.DataLoader;

// 1. Wrap the text corpus
TextDataset dataset = new TextDataset(rawText, tokenizer);

// 2. Configure DataLoader:
// - batchSize: 32
// - seqLen: 128 (context length)
// - shuffle: true (randomize start offsets per epoch)
DataLoader loader = new DataLoader(dataset, 32, 128, true);

System.out.println("Total sequence tokens: " + dataset.getLength());
System.out.println("Total batches per epoch: " + loader.getNumBatches());
```

---

## 🔄 Step 4: Integrations in Training Loop
Use the `DataLoader` within your training loop to retrieve inputs `X` and labels `Y`, run forward/backward passes, and update weights.

```java
import com.tinymodelz.math.Tensor;
import com.tinymodelz.train.AdamW;
import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.nn.Module;

// Initialize model, optimizer and loss function
Module model = ... // Your TinyGPT architecture
AdamW optimizer = new AdamW(model.getParameters(), 3e-4f);
CrossEntropyLoss lossFn = new CrossEntropyLoss();

int epochs = 10;
for (int epoch = 0; epoch < epochs; epoch++) {
    loader.reset(); // Reset iterator state & shuffle if enabled
    
    while (loader.hasNext()) {
        Tensor[] batch = loader.nextBatch();
        Tensor x = batch[0]; // Input shape: [batchSize, seqLen]
        Tensor y = batch[1]; // Target labels shape: [batchSize, seqLen]
        
        // Zero gradients
        optimizer.zeroGrad();
        
        // Forward pass
        Tensor logits = model.forward(x); // Output shape: [batchSize, seqLen, vocabSize]
        Tensor loss = lossFn.forward(logits, y);
        
        // Backward pass
        loss.backward();
        
        // Update model weights
        optimizer.step();
        
        System.out.printf("Epoch %d - Loss: %.4f\n", epoch, loss.getData()[0]);
    }
}
```
