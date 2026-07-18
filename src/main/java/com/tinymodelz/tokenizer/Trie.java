package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <h3>Trie</h3>
 * 
 * <p>A Trie (prefix tree) data structure optimized for fast vocabulary lookup and longest prefix matching.
 * This implementation is designed specifically for language model tokenization (e.g., WordPiece or MaxMatch algorithms).</p>
 * 
 * <h4>Mathematical Formulation of Operations:</h4>
 * 
 * <h5>1. Insertion</h5>
 * <p>Let $W = c_1 c_2 \dots c_L$ be a word of length $L$, and $\tau \in \mathbb{Z}_{\ge 0}$ be its associated token ID.
 * The insertion operation recursively traverses the state transition function $\delta$ starting from the root node $v_0 = r$:</p>
 * 
 * $$v_i = \delta(v_{i-1}, c_i) \quad \text{for } i = 1, 2, \dots, L$$
 * 
 * <p>If a transition $\delta(v_{i-1}, c_i)$ does not exist, a new node is created. 
 * At the terminal node $v_L$, we set:</p>
 * 
 * $$isEndOfWord(v_L) = \text{true}, \quad \text{tokenId}(v_L) = \tau$$
 * 
 * <h5>2. Longest Prefix Match (MaxMatch)</h5>
 * <p>Given an input string $S$ and a starting index $s \in [0, |S|-1]$, we seek the longest prefix $S[s \dots s + k - 1]$ 
 * of length $k \ge 1$ such that the prefix is in the vocabulary $V_{vocab}$.
 * We trace the path in the Trie starting at the root $r$:</p>
 * 
 * $$v_0 = r$$
 * $$v_j = \delta(v_{j-1}, S[s + j - 1]) \quad \text{for } j = 1, 2, \dots$$
 * 
 * <p>The search terminates at step $M$ where either $s + M = |S|$ or $\delta(v_M, S[s + M])$ is undefined.
 * The longest prefix match is given by the node $v_{k}$ ($1 \le k \le M$) that maximizes $k$ subject to 
 * $isEndOfWord(v_k) = \text{true}$.</p>
 * 
 * <h4>Complexity Analysis:</h4>
 * <ul>
 *   <li><b>Insertion Time Complexity:</b> $O(L)$ where $L$ is the length of the word.</li>
 *   <li><b>Search/Lookup Time Complexity:</b> $O(L)$ where $L$ is the length of the query word.</li>
 *   <li><b>Longest Prefix Match Time Complexity:</b> $O(M)$ where $M \le |S| - s$ is the length of the path traversed.</li>
 *   <li><b>Space Complexity:</b> $O(\sum_{w \in V_{vocab}} |w|)$ in the worst-case, where each character of every word 
 *       corresponds to a unique node. In practice, common prefixes share nodes, reducing actual memory footprint.</li>
 * </ul>
 */
public class Trie {
    private static final Logger logger = LoggerFactory.getLogger(Trie.class);
    private final TrieNode root;

    /**
     * Represents the result of a longest prefix match operation.
     */
    public static class Match {
        private final int length;
        private final int tokenId;

        /**
         * Constructs a Match result.
         * 
         * @param length the length of the matched prefix
         * @param tokenId the token ID of the matched prefix
         */
        public Match(int length, int tokenId) {
            this.length = length;
            this.tokenId = tokenId;
        }

        /**
         * Gets the length of the matched prefix.
         * 
         * @return the matched prefix length
         */
        public int getLength() {
            return length;
        }

        /**
         * Gets the token ID of the matched prefix.
         * 
         * @return the token ID, or -1 if no match was found
         */
        public int getTokenId() {
            return tokenId;
        }

        @Override
        public String toString() {
            return "Match{length=" + length + ", tokenId=" + tokenId + "}";
        }
    }

    /**
     * Initializes an empty Trie.
     */
    public Trie() {
        this.root = new TrieNode();
    }

    /**
     * Inserts a word and its associated token ID into the Trie.
     * 
     * @param word the word to insert
     * @param tokenId the unique token ID for the word
     * @throws IllegalArgumentException if the word is null or empty, or token ID is negative
     */
    public void insert(String word, int tokenId) {
        if (word == null || word.isEmpty()) {
            throw new IllegalArgumentException("Word cannot be null or empty");
        }
        if (tokenId < 0) {
            throw new IllegalArgumentException("Token ID must be non-negative");
        }

        logger.trace("Inserting word '{}' with token ID {}", word, tokenId);
        TrieNode current = root;
        for (int i = 0; i < word.length(); i++) {
            current = current.getOrCreateChild(word.charAt(i));
        }
        current.setEndOfWord(true);
        current.setTokenId(tokenId);
    }

    /**
     * Searches for a word in the Trie and returns its token ID.
     * 
     * @param word the word to search for
     * @return the token ID, or -1 if the word is not in the Trie
     */
    public int search(String word) {
        if (word == null || word.isEmpty()) {
            return -1;
        }

        logger.trace("Searching Trie for exact word: '{}'", word);
        TrieNode current = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (!current.hasChild(c)) {
                logger.trace("Search mismatch: character '{}' not found at depth {}", c, i);
                return -1;
            }
            current = current.getChild(c);
        }

        if (current.isEndOfWord()) {
            logger.trace("Search found exact word: '{}' with ID {}", word, current.getTokenId());
            return current.getTokenId();
        } else {
            logger.trace("Search prefix match only (not a terminal word): '{}'", word);
            return -1;
        }
    }

    /**
     * Finds the longest prefix of the substring text[start...] that exists in the Trie.
     * 
     * @param text the full input text string
     * @param start the starting index of the substring to match
     * @return a Match object containing the length of the longest matching prefix and its token ID.
     *         If no match is found, returns a Match with length 0 and token ID -1.
     * @throws IllegalArgumentException if text is null or start is out of bounds
     */
    public Match longestPrefixMatch(String text, int start) {
        if (text == null) {
            throw new IllegalArgumentException("Text cannot be null");
        }
        if (start < 0 || start > text.length()) {
            throw new IllegalArgumentException("Start index out of bounds: " + start);
        }

        logger.trace("Performing longestPrefixMatch on text from index {}: '{}'", start, text.substring(start));
        TrieNode current = root;
        int longestLength = 0;
        int matchedTokenId = -1;

        for (int i = start; i < text.length(); i++) {
            char c = text.charAt(i);
            if (!current.hasChild(c)) {
                break;
            }
            current = current.getChild(c);
            if (current.isEndOfWord()) {
                longestLength = i - start + 1;
                matchedTokenId = current.getTokenId();
                logger.trace("Found intermediate prefix match: '{}' (len={}, ID={})",
                        text.substring(start, i + 1), longestLength, matchedTokenId);
            }
        }

        logger.trace("Longest prefix match result: length={}, ID={}", longestLength, matchedTokenId);
        return new Match(longestLength, matchedTokenId);
    }

    /**
     * Gets the root node of the Trie.
     * 
     * @return the root TrieNode
     */
    public TrieNode getRoot() {
        return root;
    }
}
