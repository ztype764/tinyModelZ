package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * <h3>TrieTokenizer</h3>
 * 
 * <p>A Java-based WordPiece/MaxMatch tokenizer implemented from scratch.
 * This class uses a {@link Trie} to partition words into subwords. It is designed 
 * for language model tokenization tasks, adhering to strict computational constraints 
 * and providing full transparency of the underlying algorithms.</p>
 * 
 * <h4>Mathematical Formulation of WordPiece Tokenization:</h4>
 * 
 * <p>Let $W = c_1 c_2 \dots c_N$ be a string representing a word of length $N$ over alphabet $\Sigma$.
 * Let $V$ be our vocabulary containing two types of subwords:</p>
 * <ol>
 *   <li>Root subwords $s \in V$ (e.g., "play")</li>
 *   <li>Suffix subwords prefixed with "##", i.e., $\text{"\#\#"} \oplus s \in V$ (e.g., "##ing")</li>
 * </ol>
 * 
 * <p>We define the segmentation of $W$ into $k$ subwords $T = (t_1, t_2, \dots, t_k)$ recursively. 
 * Let $idx_j$ be the character index in $W$ after emitting token $t_j$. We initialize $idx_0 = 0$.</p>
 * 
 * <p>At step $j \ge 1$, we define the search candidate suffix $S_j$ as:</p>
 * 
 * $$S_j = \begin{cases} 
 * W[idx_{j-1} \dots N-1] & \text{if } idx_{j-1} = 0 \\
 * \text{"\#\#"} \oplus W[idx_{j-1} \dots N-1] & \text{if } idx_{j-1} > 0 
 * \end{cases}$$
 * 
 * <p>We query the Trie for the longest prefix match of $S_j$. Let the matched token in the Trie be $t_j \in V$ 
 * and its length be $L_j$.</p>
 * 
 * <p>The transition for the index sequence is:</p>
 * 
 * $$idx_j = \begin{cases}
 * idx_{j-1} + L_j & \text{if } idx_{j-1} = 0 \\
 * idx_{j-1} + (L_j - 2) & \text{if } idx_{j-1} > 0
 * \end{cases}$$
 * 
 * <p>The algorithm terminates successfully at step $k$ if $idx_k = N$.
 * If at any step $j$ no prefix match is found (i.e., $L_j = 0$ or, for $idx_{j-1} > 0$, $L_j \le 2$), 
 * the entire word $W$ is declared untokenizable and is mapped to the unknown token:</p>
 * 
 * $$T = (t_{unk})$$
 * 
 * <h4>Complexity Analysis:</h4>
 * <ul>
 *   <li><b>Pre-tokenization:</b> $O(|S|)$ to split text by whitespace and separate punctuation.</li>
 *   <li><b>Tokenization:</b> For a word $W$, the worst-case time complexity is $O(|W|^2)$ if we backtrack, 
 *       but using the Trie, each step takes $O(L_j)$ which yields a linear time complexity $O(|W|)$ 
 *       with respect to the word length.</li>
 *   <li><b>Space Complexity:</b> $O(|S|)$ to store the token lists and generated subwords.</li>
 * </ul>
 */
public class TrieTokenizer implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(TrieTokenizer.class);

    private final List<String> idToToken;
    private final Map<String, Integer> tokenToId;
    private final Trie trie;

    private final String padToken = "[PAD]";
    private final String unkToken = "[UNK]";
    private final String clsToken = "[CLS]";
    private final String sepToken = "[SEP]";
    private final String maskToken = "[MASK]";

    private final int padId;
    private final int unkId;
    private final int clsId;
    private final int sepId;
    private final int maskId;

    /**
     * Constructs a TrieTokenizer with the provided vocabulary.
     * Special tokens ([PAD], [UNK], [CLS], [SEP], [MASK]) will be verified and added if not present.
     * 
     * @param vocab the vocabulary as a list of strings
     * @throws IllegalArgumentException if the vocabulary is null or empty
     */
    public TrieTokenizer(List<String> vocab) {
        if (vocab == null || vocab.isEmpty()) {
            throw new IllegalArgumentException("Vocabulary cannot be null or empty");
        }

        this.idToToken = new ArrayList<>(vocab);
        Set<String> vocabSet = new HashSet<>(vocab);

        String[] specialTokens = {padToken, unkToken, clsToken, sepToken, maskToken};
        for (String special : specialTokens) {
            if (!vocabSet.contains(special)) {
                idToToken.add(special);
            }
        }

        this.tokenToId = new HashMap<>(idToToken.size());
        this.trie = new Trie();

        for (int i = 0; i < idToToken.size(); i++) {
            String token = idToToken.get(i);
            tokenToId.put(token, i);
            trie.insert(token, i);
        }

        this.padId = tokenToId.get(padToken);
        this.unkId = tokenToId.get(unkToken);
        this.clsId = tokenToId.get(clsToken);
        this.sepId = tokenToId.get(sepToken);
        this.maskId = tokenToId.get(maskToken);

        logger.info("Initializing TrieTokenizer. Vocab size: {}, Special tokens: PAD={}, UNK={}, CLS={}, SEP={}, MASK={}",
                idToToken.size(), padId, unkId, clsId, sepId, maskId);
    }

    @Override
    public List<String> tokenize(String text) {
        if (text == null) {
            throw new IllegalArgumentException("Text cannot be null");
        }

        logger.debug("Tokenizing text: '{}'", text);
        List<String> words = preTokenize(text);
        List<String> allTokens = new ArrayList<>();

        for (String word : words) {
            // If the word itself is directly in the vocabulary, preserve it.
            if (tokenToId.containsKey(word)) {
                logger.trace("Word '{}' found directly in vocabulary.", word);
                allTokens.add(word);
                continue;
            }

            List<String> wordTokens = new ArrayList<>();
            boolean isBad = false;
            int start = 0;
            int wordLen = word.length();

            while (start < wordLen) {
                String searchStr;
                if (start == 0) {
                    searchStr = word.substring(start);
                } else {
                    searchStr = "##" + word.substring(start);
                }

                Trie.Match match = trie.longestPrefixMatch(searchStr, 0);
                if (match.getLength() == 0) {
                    logger.trace("No prefix match found for suffix '{}' of word '{}'", searchStr, word);
                    isBad = true;
                    break;
                }

                int matchedLength = match.getLength();
                if (start > 0) {
                    if (matchedLength <= 2) { // Matched only "##" or subset, which is invalid
                        logger.trace("Suffix match was only the prefix '##' for suffix '{}' of word '{}'", searchStr, word);
                        isBad = true;
                        break;
                    }
                    matchedLength -= 2;
                }

                String token = searchStr.substring(0, match.getLength());
                logger.trace("Matched subword: '{}' (len={}, id={})", token, matchedLength, match.getTokenId());
                wordTokens.add(token);
                start += matchedLength;
            }

            if (isBad) {
                logger.warn("Word '{}' could not be fully tokenized; falling back to [UNK]", word);
                allTokens.add(unkToken);
            } else {
                logger.debug("Word '{}' successfully tokenized into: {}", word, wordTokens);
                allTokens.addAll(wordTokens);
            }
        }

        logger.debug("Final tokenized list: {}", allTokens);
        return allTokens;
    }

    @Override
    public List<Integer> encode(String text) {
        logger.info("Encoding text: '{}'", text);
        List<String> tokens = tokenize(text);
        List<Integer> ids = new ArrayList<>(tokens.size());
        for (String token : tokens) {
            ids.add(tokenToId.getOrDefault(token, unkId));
        }
        logger.debug("Encoded token IDs: {}", ids);
        return ids;
    }

    @Override
    public String decode(List<Integer> ids) {
        return decode(ids, true);
    }

    /**
     * Decodes a list of token IDs back into a single string, with optional control over special tokens.
     * 
     * @param ids the list of token IDs to decode
     * @param skipSpecialTokens if true, special tokens are omitted from the output
     * @return the decoded string
     */
    public String decode(List<Integer> ids, boolean skipSpecialTokens) {
        if (ids == null || ids.isEmpty()) {
            return "";
        }

        logger.info("Decoding token IDs: {} (skipSpecialTokens={})", ids, skipSpecialTokens);
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String token = idToToken(id);
            if (skipSpecialTokens && (token.equals(padToken) || token.equals(clsToken) || 
                                     token.equals(sepToken) || token.equals(maskToken))) {
                logger.trace("Skipping special token '{}' (id={}) during decode", token, id);
                continue;
            }

            if (token.startsWith("##")) {
                sb.append(token.substring(2));
            } else {
                if (sb.length() > 0) {
                    sb.append(" ");
                }
                sb.append(token);
            }
        }
        String result = sb.toString();
        logger.debug("Decoded result string: '{}'", result);
        return result;
    }

    @Override
    public int getVocabSize() {
        return idToToken.size();
    }

    @Override
    public String idToToken(int id) {
        if (id < 0 || id >= idToToken.size()) {
            return unkToken;
        }
        return idToToken.get(id);
    }

    @Override
    public int tokenToId(String token) {
        return tokenToId.getOrDefault(token, unkId);
    }

    /**
     * Helper to split text by whitespace and separate punctuation.
     */
    private List<String> preTokenize(String text) {
        List<String> words = new ArrayList<>();
        StringBuilder currentWord = new StringBuilder();

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isWhitespace(c)) {
                if (currentWord.length() > 0) {
                    words.add(currentWord.toString());
                    currentWord.setLength(0);
                }
            } else if (isPunctuation(c)) {
                if (currentWord.length() > 0) {
                    words.add(currentWord.toString());
                    currentWord.setLength(0);
                }
                words.add(String.valueOf(c));
            } else {
                currentWord.append(c);
            }
        }

        if (currentWord.length() > 0) {
            words.add(currentWord.toString());
        }

        return words;
    }

    /**
     * Determines whether a character should be treated as a punctuation/symbol boundary.
     */
    private boolean isPunctuation(char c) {
        int type = Character.getType(c);
        return type == Character.CONNECTOR_PUNCTUATION ||
               type == Character.DASH_PUNCTUATION ||
               type == Character.START_PUNCTUATION ||
               type == Character.END_PUNCTUATION ||
               type == Character.INITIAL_QUOTE_PUNCTUATION ||
               type == Character.FINAL_QUOTE_PUNCTUATION ||
               type == Character.OTHER_PUNCTUATION ||
               type == Character.MATH_SYMBOL ||
               type == Character.CURRENCY_SYMBOL ||
               type == Character.MODIFIER_SYMBOL ||
               type == Character.OTHER_SYMBOL;
    }

    // Special token accessors

    public String getPadToken() { return padToken; }
    public String getUnkToken() { return unkToken; }
    public String getClsToken() { return clsToken; }
    public String getSepToken() { return sepToken; }
    public String getMaskToken() { return maskToken; }

    public int getPadId() { return padId; }
    public int getUnkId() { return unkId; }
    public int getClsId() { return clsId; }
    public int getSepId() { return sepId; }
    public int getMaskId() { return maskId; }
}
