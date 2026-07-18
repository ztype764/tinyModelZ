package com.tinymodelz;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.tinymodelz.math.MathEngineTest;
import com.tinymodelz.math.TensorTest;
import com.tinymodelz.nn.NeuralNetworkTest;
import com.tinymodelz.tokenizer.CharacterTokenizerTest;
import com.tinymodelz.tokenizer.TrieTest;
import com.tinymodelz.tokenizer.TrieTokenizerTest;

/**
 * TestRunner execution entry point.
 * Runs all unit tests for the tokenizer components and exits with a non-zero
 * code
 * if any assertions fail.
 */
public class TestRunner {

    private static final Logger logger = LoggerFactory.getLogger(TestRunner.class);

    public static void main(String[] args) {
        logger.info("==================================================");
        logger.info("Starting TinyModelZ Tokenizer Test Suite");
        logger.info("==================================================");

        try {
            TestReporter.startSuite("Trie Prefix Tree Operations");
            TrieTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("WordPiece Trie Tokenizer");
            TrieTokenizerTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Character-Level Tokenizer");
            CharacterTokenizerTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Math Engine Operations");
            MathEngineTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Tensor Engine Operations");
            TensorTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Neural Network Layers");
            NeuralNetworkTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Transformer Components");
            com.tinymodelz.nn.TransformerTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Language Model Training & Optimization");
            com.tinymodelz.train.TrainingTest.runTests();
            TestReporter.endSuite();

            TestReporter.startSuite("Autoregressive Inference & Sampling");
            com.tinymodelz.inference.GeneratorTest.runTests();
            TestReporter.endSuite();

            // Generate visual report
            TestReporter.generateReport("test_report.html");

            if (TestReporter.hasFailures()) {
                logger.error("==================================================");
                logger.error("TEST SUITE FAILED! Check test_report.html for details.");
                logger.error("==================================================");
                System.exit(1);
            }

            logger.info("==================================================");
            logger.info("ALL TESTS PASSED SUCCESSFULLY! Check test_report.html for visualization.");
            logger.info("==================================================");
            System.exit(0);
        } catch (Throwable e) {
            logger.error("==================================================");
            logger.error("UNEXPECTED ERROR RUNNING TESTS: ", e);
            logger.error("==================================================");
            System.exit(1);
        }
    }
}
