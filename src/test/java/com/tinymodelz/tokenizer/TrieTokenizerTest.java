package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Unit tests for the {@link TrieTokenizer} class.
 */
public class TrieTokenizerTest {

    private static final Logger logger = LoggerFactory.getLogger(TrieTokenizerTest.class);

    public static void runTests() {
        com.tinymodelz.TestReporter.runTest("Testing Basic Tokenization", () -> testBasicTokenization());
        com.tinymodelz.TestReporter.runTest("Testing WordPiece subword tokenization", () -> testWordPieceTokenization());
        com.tinymodelz.TestReporter.runTest("Testing Encode and Decode", () -> testEncodeAndDecode());
        com.tinymodelz.TestReporter.runTest("Testing Special Tokens registration", () -> testSpecialTokens());
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void testBasicTokenization() {
        logger.info("Testing Basic Tokenization...");
        List<String> vocab = Arrays.asList(
            "hello", "world", ",", "!"
        );
        TrieTokenizer tokenizer = new TrieTokenizer(vocab);

        List<String> tokens = tokenizer.tokenize("hello, world!");
        List<String> expected = Arrays.asList("hello", ",", "world", "!");
        assertEquals(expected, tokens, "Basic tokenization with punctuation failed");
    }

    private static void testWordPieceTokenization() {
        logger.info("Testing WordPiece subword tokenization...");
        List<String> vocab = Arrays.asList(
            "play", "##er", "##ing", "walk", "##s", "##ed"
        );
        TrieTokenizer tokenizer = new TrieTokenizer(vocab);

        // Test normal root + suffix subword split
        assertEquals(
            Arrays.asList("play", "##ing"),
            tokenizer.tokenize("playing"),
            "WordPiece split 'playing' failed"
        );

        assertEquals(
            Arrays.asList("play", "##er"),
            tokenizer.tokenize("player"),
            "WordPiece split 'player' failed"
        );

        // Test word not fully coverable by vocab (should become [UNK])
        assertEquals(
            Arrays.asList("[UNK]"),
            tokenizer.tokenize("playerx"),
            "OOV word 'playerx' should resolve to [UNK]"
        );

        // Test mixed word sequence
        assertEquals(
            Arrays.asList("walk", "##s", "play", "##ed"),
            tokenizer.tokenize("walks played"),
            "Multi-word subword tokenization failed"
        );
    }

    private static void testEncodeAndDecode() {
        logger.info("Testing Encode and Decode...");
        List<String> vocab = Arrays.asList(
            "hello", "world", "play", "##er", "##ing", ",", "!"
        );
        TrieTokenizer tokenizer = new TrieTokenizer(vocab);

        String text = "hello playing!";
        List<Integer> ids = tokenizer.encode(text);
        
        int helloId = tokenizer.tokenToId("hello");
        int playId = tokenizer.tokenToId("play");
        int ingId = tokenizer.tokenToId("##ing");
        int exclId = tokenizer.tokenToId("!");
        
        List<Integer> expectedIds = Arrays.asList(helloId, playId, ingId, exclId);
        assertEquals(expectedIds, ids, "Encoding text to IDs failed");

        String decoded = tokenizer.decode(ids);
        assertEquals("hello playing !", decoded, "Decoding IDs back to string failed");
    }

    private static void testSpecialTokens() {
        logger.info("Testing Special Tokens registration...");
        List<String> vocab = Arrays.asList("token1", "token2");
        TrieTokenizer tokenizer = new TrieTokenizer(vocab);

        assertEquals(true, tokenizer.tokenToId("[PAD]") >= 0, "PAD token must exist");
        assertEquals(true, tokenizer.tokenToId("[UNK]") >= 0, "UNK token must exist");
        assertEquals(true, tokenizer.tokenToId("[CLS]") >= 0, "CLS token must exist");
        assertEquals(true, tokenizer.tokenToId("[SEP]") >= 0, "SEP token must exist");
        assertEquals(true, tokenizer.tokenToId("[MASK]") >= 0, "MASK token must exist");

        int padId = tokenizer.getPadId();
        assertEquals("[PAD]", tokenizer.idToToken(padId), "padId must map back to [PAD]");
    }
}
