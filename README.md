# tinyModelZ - Custom Trie Tokenizer

A custom, from-scratch Java implementation of a WordPiece/MaxMatch tokenizer designed for language model development.

## 📁 Project Directory Structure

```
tinyModelZ/
├── src/
│   ├── main/java/com/tinymodelz/tokenizer/
│   │   ├── TrieNode.java       # Trie tree nodes and state transition mappings
│   │   ├── Trie.java           # Insertion, lookup, and longest prefix matching (MaxMatch)
│   │   ├── Tokenizer.java      # Base tokenizer interface contract
│   │   └── TrieTokenizer.java  # WordPiece subword splitter & pre-tokenization
│   └── test/java/com/tinymodelz/tokenizer/
│       ├── TrieTest.java       # Unit tests for the Trie data structure
│       ├── TrieTokenizerTest.java # Unit tests for the WordPiece tokenizer
│       └── TestRunner.java     # Entry point to execute all unit tests
├── RULES.md                    # Project rules and engineering constraints
├── pom.xml                     # Maven project configuration file
├── build.sh                    # Command-line build and test runner script
└── .gitignore                  # Git untracked file configurations
```

## 🚀 How to Build and Run Tests

You can build the project and run the complete test suite using the provided `build.sh` script:

```bash
bash build.sh
```

The script will:
1. Clean the build directory (`bin/`).
2. Compile all source and test Java files.
3. Execute the `TestRunner` test suite.

## 📝 Compliance with project RULES.md

1. **No External Packages**: Built using native Java collections and standard library utilities only.
2. **From-Scratch Algorithm Implementation**: Custom Trie tree, prefix matcher, and WordPiece parser.
3. **Comprehensive Unit Tests**: Built-in test suite verifying all tokenization paths.
4. **Mathematical Documentation**: Elaborate math formulations and complexity profiles included in each class's Javadoc.
