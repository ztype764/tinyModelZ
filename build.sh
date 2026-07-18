#!/bin/bash
set -e

# Setup clean build environment
echo "Cleaning build directory 'bin'..."
rm -rf bin
mkdir -p bin

# Compile all Java files in src
echo "Compiling Java files..."
javac -cp "lib/*" -d bin $(find src -name "*.java")

# Run the test suite
# Note: Change -Dorg.slf4j.simpleLogger.defaultLogLevel to "debug" or "trace" to see detailed step-by-step tokenizer matching logs.
echo "Running tests..."
java -Dorg.slf4j.simpleLogger.defaultLogLevel=info -cp "bin:lib/*" com.tinymodelz.tokenizer.TestRunner
