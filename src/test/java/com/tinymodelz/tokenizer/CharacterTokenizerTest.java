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
        logger.info("Running CharacterTokenizerTest...");
        testBasicCharacterSplitting();
        testEncodeAndDecodeRoundtrip();
        testSpecialAndUnkTokens();
        logger.info("CharacterTokenizerTest passed successfully!");
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
        assertEquals(expectedIds, ids, "Character encoding to IDs failed");

        String decodedText = tokenizer.decode(ids);
        assertEquals(originalText, decodedText, "Character decoding (roundtrip) failed");
    }

    private static void testSpecialAndUnkTokens() {
        logger.info("Testing Special and Unknown Characters...");
        List<String> vocab = Arrays.asList("a", "b", "c");
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);

        // 'z' is not in vocab, should become [UNK]
        List<String> tokens = tokenizer.tokenize("abz");
        List<String> expected = Arrays.asList("a", "b", "[UNK]");
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
