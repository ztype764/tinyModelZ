# tinyModelZ - Custom Tokenizer & Math Engine

A custom, from-scratch Java implementation of a WordPiece/MaxMatch tokenizer, character-level tokenizer, and basic Math Engine designed for language model development.

## 📁 Project Directory Structure

```
tinyModelZ/
├── src/
│   ├── main/java/com/tinymodelz/
│   │   ├── math/
│   │   │   ├── Vector.java             # 1D vector operations (norm, dot, softmax)
│   │   │   ├── Matrix.java             # 2D matrix operations (dot product, transpose, bias)
│   │   │   ├── RandomInitializer.java  # Xavier/He parameter initializers
│   │   │   └── MathIO.java             # TMA1 binary weight saving/loading
│   │   └── tokenizer/
│   │       ├── TrieNode.java           # Trie tree nodes and state transition mappings
│   │       ├── Trie.java               # Insertion, lookup, and longest prefix matching (MaxMatch)
│   │       ├── Tokenizer.java          # Base tokenizer interface contract
│   │       ├── TrieTokenizer.java      # WordPiece subword splitter & pre-tokenization
│   │       └── CharacterTokenizer.java # Lossless character-level tokenizer
│   └── test/java/com/tinymodelz/
│       ├── TestReporter.java           # Visual HTML5 test report generator
│       ├── TestRunner.java             # Entry point to execute all unit tests
│       ├── math/
│       │   └── MathEngineTest.java     # Math Engine unit tests
│       └── tokenizer/
│           ├── TrieTest.java           # Trie unit tests
│           ├── TrieTokenizerTest.java  # WordPiece tokenizer unit tests
│           └── CharacterTokenizerTest.java # Character-level tokenizer unit tests
├── RULES.md                            # Project rules and engineering constraints
├── GOALS.md                            # Project goals checklist
├── pom.xml                             # Maven project configuration file
├── build.sh                            # Command-line build and test runner script
└── .gitignore                          # Git untracked file configurations
```

## 🚀 How to Build and Run Tests

You can build the project and run the complete test suite using the provided `build.sh` script:

```bash
bash build.sh
```

The script will:
1. Clean the build directory (`bin/`).
2. Compile all source and test Java files.
3. Execute the `TestRunner` test suite, which runs all tests and generates a visual report.

## 📊 Visualizing Test Results
After running the tests, an interactive HTML5 dashboard is generated at the root of the project:
* **File**: `test_report.html`
* Open it directly in any web browser to view overall pass rates, detailed suite statistics, test execution times, and collapsible log summaries.

## 📝 Compliance with project RULES.md

1. **No External Packages**: Built using native Java collections and standard library utilities only.
2. **From-Scratch Algorithm Implementation**: Custom Trie tree, prefix matcher, WordPiece parser, character tokenizer, weight initializers, and binary serialization format.
3. **Comprehensive Unit Tests**: Built-in test suite verifying all tokenization paths and mathematical operations.
4. **Mathematical Documentation**: Javadocs containing LaTeX-styled math formulas and algorithmic complexity profiles for every major class.
