package com.tinymodelz.tokenizer;

import java.util.HashMap;
import java.util.Map;

/**
 * <h3>TrieNode</h3>
 * 
 * <p>Represents a single node in a Trie (prefix tree) data structure.
 * Mathematically, a Trie is represented as a directed graph $G = (V, E)$ where:</p>
 * <ul>
 *   <li>$V$ is the set of nodes (vertices), where each node $v \in V$ corresponds to a unique prefix $p(v) \in \Sigma^*$.</li>
 *   <li>$\Sigma$ is the alphabet of characters.</li>
 *   <li>$E \subseteq V \times \Sigma \times V$ is the set of labeled, directed edges. 
 *       For each node $u \in V$, and each character $c \in \Sigma$, there exists at most one outgoing edge $(u, c, v) \in E$.</li>
 * </ul>
 * 
 * <p>For each node $v \in V$, we define a function $f: V \to \{0, 1\}$ indicating whether $p(v)$ is a valid word 
 * in the vocabulary $V_{vocab}$:</p>
 * 
 * $$f(v) = \begin{cases} 
 * 1 & \text{if } p(v) \in V_{vocab} \\ 
 * 0 & \text{otherwise} 
 * \end{cases}$$
 * 
 * <p>Each node in this implementation stores:</p>
 * <ol>
 *   <li>A transition map $\delta: \Sigma \to V$, mapping a character $c$ to a child node $v = \delta(u, c)$.</li>
 *   <li>A boolean flag $isEndOfWord$ representing $f(v)$.</li>
 *   <li>A token ID $\tau(v) \in \mathbb{Z}_{\ge 0}$ representing the vocabulary index of the word $p(v)$ when $f(v) = 1$.</li>
 * </ol>
 * 
 * <h4>Complexity:</h4>
 * <ul>
 *   <li><b>Space Complexity:</b> $O(|\Sigma|)$ per node in the worst case. In practice, using a hash map, 
 *       the space complexity is proportional to the number of active child transitions, i.e., $O(d)$ where $d$ 
 *       is the out-degree of the node.</li>
 *   <li><b>Transition Time Complexity:</b> $O(1)$ on average for hash map lookups.</li>
 * </ul>
 */
public class TrieNode {
    private final Map<Character, TrieNode> children;
    private boolean isEndOfWord;
    private int tokenId;

    /**
     * Constructs a new TrieNode.
     */
    public TrieNode() {
        this.children = new HashMap<>();
        this.isEndOfWord = false;
        this.tokenId = -1;
    }

    /**
     * Checks if a transition exists for the given character.
     * 
     * @param c the character transition to check
     * @return true if a child node exists for character c, false otherwise
     */
    public boolean hasChild(char c) {
        return children.containsKey(c);
    }

    /**
     * Retrieves the child node associated with the given character transition.
     * 
     * @param c the character transition
     * @return the child TrieNode, or null if no transition exists
     */
    public TrieNode getChild(char c) {
        return children.get(c);
    }

    /**
     * Adds or retrieves a child node for the given character.
     * 
     * @param c the character transition
     * @return the child TrieNode (either existing or newly created)
     */
    public TrieNode getOrCreateChild(char c) {
        return children.computeIfAbsent(c, k -> new TrieNode());
    }

    /**
     * Returns whether this node represents the end of a valid word in the vocabulary.
     * 
     * @return true if the node represents a word, false otherwise
     */
    public boolean isEndOfWord() {
        return isEndOfWord;
    }

    /**
     * Sets whether this node represents the end of a valid word in the vocabulary.
     * 
     * @param endOfWord true if the node represents a word, false otherwise
     */
    public void setEndOfWord(boolean endOfWord) {
        isEndOfWord = endOfWord;
    }

    /**
     * Gets the token ID associated with this node.
     * 
     * @return the token ID, or -1 if the node is not the end of a word
     */
    public int getTokenId() {
        return tokenId;
    }

    /**
     * Sets the token ID associated with this node.
     * 
     * @param tokenId the token ID to assign
     */
    public void setTokenId(int tokenId) {
        this.tokenId = tokenId;
    }

    /**
     * Returns the transitions (children) of this node.
     * 
     * @return a map of character transitions to child nodes
     */
    public Map<Character, TrieNode> getChildren() {
        return children;
    }
}
