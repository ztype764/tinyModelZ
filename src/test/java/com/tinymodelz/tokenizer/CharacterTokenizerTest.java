package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Unit tests for the {@link CharacterTokenizer} class.
 */
public class CharacterTokenizerTest {

    private static final Logger logger = LoggerFactory.getLogger(CharacterTokenizerTest.class);

    public static void runTests() {
        com.tinymodelz.TestReporter.runTest("Testing Basic Character Splitting", () -> testBasicCharacterSplitting());
        com.tinymodelz.TestReporter.runTest("Testing Character Encode and Decode Roundtrip", () -> testEncodeAndDecodeRoundtrip());
        com.tinymodelz.TestReporter.runTest("Testing Special and Unknown Characters", () -> testSpecialAndUnkTokens());
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void testBasicCharacterSplitting() {
        logger.info("Testing Basic Character Splitting...");
        List<String> vocab = Arrays.asList(
            "h", "e", "l", "o", " ", "w", "r", "d", "!"
        );
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);

        List<String> tokens = tokenizer.tokenize("hello world!");
        List<String> expected = Arrays.asList("h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "!");
        com.tinymodelz.TestReporter.logMetric("Input text", "hello world!");
        com.tinymodelz.TestReporter.logMetric("Split tokens", tokens);
        assertEquals(expected, tokens, "Character-level splitting failed");
    }

    private static void testEncodeAndDecodeRoundtrip() {
        logger.info("Testing Character Encode and Decode Roundtrip...");
        List<String> vocab = Arrays.asList(
            "h", "e", "l", "o", " ", "w", "r", "d", "!"
        );
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);

        String originalText = "hello world!";
        List<Integer> ids = tokenizer.encode(originalText);
        
        List<Integer> expectedIds = Arrays.asList(
            tokenizer.tokenToId("h"),
            tokenizer.tokenToId("e"),
            tokenizer.tokenToId("l"),
            tokenizer.tokenToId("l"),
            tokenizer.tokenToId("o"),
            tokenizer.tokenToId(" "),
            tokenizer.tokenToId("w"),
            tokenizer.tokenToId("o"),
            tokenizer.tokenToId("r"),
            tokenizer.tokenToId("l"),
            tokenizer.tokenToId("d"),
            tokenizer.tokenToId("!")
        );
        com.tinymodelz.TestReporter.logMetric("Input text", originalText);
        com.tinymodelz.TestReporter.logMetric("Encoded IDs", ids);
        assertEquals(expectedIds, ids, "Character encoding to IDs failed");

        String decodedText = tokenizer.decode(ids);
        com.tinymodelz.TestReporter.logMetric("Decoded text", decodedText);
        assertEquals(originalText, decodedText, "Character decoding (roundtrip) failed");
    }

    private static void testSpecialAndUnkTokens() {
        logger.info("Testing Special and Unknown Characters...");
        List<String> vocab = Arrays.asList("a", "b", "c");
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);

        // 'z' is not in vocab, should become [UNK]
        List<String> tokens = tokenizer.tokenize("abz");
        List<String> expected = Arrays.asList("a", "b", "[UNK]");
        com.tinymodelz.TestReporter.logMetric("Vocab defined", vocab);
        com.tinymodelz.TestReporter.logMetric("Tokenized 'abz'", tokens);
        assertEquals(expected, tokens, "Unknown character mapping failed");

        int unkId = tokenizer.getUnkId();
        assertEquals(unkId, tokenizer.encode("z").get(0), "Unknown character ID mapping failed");

        // Special tokens registration check
        assertEquals("[PAD]", tokenizer.idToToken(tokenizer.getPadId()), "PAD token retrieval failed");
        assertEquals("[UNK]", tokenizer.idToToken(tokenizer.getUnkId()), "UNK token retrieval failed");
        assertEquals("[CLS]", tokenizer.idToToken(tokenizer.getClsId()), "CLS token retrieval failed");
        assertEquals("[SEP]", tokenizer.idToToken(tokenizer.getSepId()), "SEP token retrieval failed");
        assertEquals("[MASK]", tokenizer.idToToken(tokenizer.getMaskId()), "MASK token retrieval failed");
    }
}
