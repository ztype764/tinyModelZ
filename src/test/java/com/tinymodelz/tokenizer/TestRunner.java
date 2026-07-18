package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TestRunner execution entry point.
 * Runs all unit tests for the tokenizer components and exits with a non-zero code
 * if any assertions fail.
 */
public class TestRunner {

    private static final Logger logger = LoggerFactory.getLogger(TestRunner.class);

    public static void main(String[] args) {
        logger.info("==================================================");
        logger.info("Starting TinyModelZ Tokenizer Test Suite");
        logger.info("==================================================");

        try {
            TrieTest.runTests();
            TrieTokenizerTest.runTests();
            logger.info("==================================================");
            logger.info("ALL TESTS PASSED SUCCESSFULLY!");
            logger.info("==================================================");
            System.exit(0);
        } catch (AssertionError e) {
            logger.error("==================================================");
            logger.error("TEST FAILURE: ", e);
            logger.error("==================================================");
            System.exit(1);
        } catch (Exception e) {
            logger.error("==================================================");
            logger.error("UNEXPECTED ERROR RUNNING TESTS: ", e);
            logger.error("==================================================");
            System.exit(1);
        }
    }
}
