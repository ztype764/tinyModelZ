package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Unit tests for the {@link Trie} class.
 */
public class TrieTest {

    private static final Logger logger = LoggerFactory.getLogger(TrieTest.class);

    public static void runTests() {
        logger.info("Running TrieTest...");
        testEmptyTrie();
        testInsertionAndSearch();
        testLongestPrefixMatch();
        logger.info("TrieTest passed successfully!");
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void testEmptyTrie() {
        logger.info("Testing Empty Trie...");
        Trie trie = new Trie();
        assertEquals(-1, trie.search("hello"), "Empty trie should return -1 for search");
        Trie.Match match = trie.longestPrefixMatch("hello", 0);
        assertEquals(0, match.getLength(), "Empty trie longest prefix match should have length 0");
        assertEquals(-1, match.getTokenId(), "Empty trie longest prefix match should have token ID -1");
    }

    private static void testInsertionAndSearch() {
        logger.info("Testing Insertion and Search...");
        Trie trie = new Trie();
        trie.insert("play", 10);
        trie.insert("player", 11);
        trie.insert("playing", 12);

        assertEquals(10, trie.search("play"), "Should find 'play'");
        assertEquals(11, trie.search("player"), "Should find 'player'");
        assertEquals(12, trie.search("playing"), "Should find 'playing'");
        assertEquals(-1, trie.search("playe"), "Should not find 'playe'");
        assertEquals(-1, trie.search("playings"), "Should not find 'playings'");
    }

    private static void testLongestPrefixMatch() {
        logger.info("Testing Longest Prefix Match...");
        Trie trie = new Trie();
        trie.insert("play", 10);
        trie.insert("player", 11);
        trie.insert("ing", 12);
        trie.insert("##ing", 13);

        // Test matching from start index 0
        Trie.Match match1 = trie.longestPrefixMatch("players", 0);
        assertEquals(6, match1.getLength(), "Longest prefix should match 'player' of length 6");
        assertEquals(11, match1.getTokenId(), "Token ID should be 11");

        Trie.Match match2 = trie.longestPrefixMatch("playing", 0);
        assertEquals(4, match2.getLength(), "Longest prefix should match 'play' of length 4");
        assertEquals(10, match2.getTokenId(), "Token ID should be 10");

        // Test matching from start index shifting
        Trie.Match match3 = trie.longestPrefixMatch("playing", 4);
        assertEquals(3, match3.getLength(), "Longest prefix of 'ing' should match 'ing' of length 3");
        assertEquals(12, match3.getTokenId(), "Token ID should be 12");

        // Test subword prefix match
        Trie.Match match4 = trie.longestPrefixMatch("##ing", 0);
        assertEquals(5, match4.getLength(), "Longest prefix of '##ing' should match '##ing' of length 5");
        assertEquals(13, match4.getTokenId(), "Token ID should be 13");
    }
}
