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
 * <h3>CharacterTokenizer</h3>
 * 
 * <p>A Java-based character-level tokenizer implemented from scratch.
 * This class partitions text strings into sequences of individual characters (or Unicode code points), 
 * providing simple, clean, and lossless text decomposition.</p>
 * 
 * <h4>Mathematical Formulation:</h4>
 * 
 * <p>Let $\Sigma$ be our character alphabet, and $\Sigma^*$ be the set of all finite strings.
 * Let $V_{vocab} = \{t_0, t_1, \dots, t_{M-1}\}$ be our vocabulary of size $M$, where each token $t_i$ 
 * represents either a single character or a special control token.</p>
 * 
 * <p>For an input string $S = c_1 c_2 \dots c_N$ of length $N$ where each character $c_i \in \Sigma$:</p>
 * <ol>
 *   <li><b>Character Tokenization mapping ($\phi_{char}$):</b> Splits the string $S$ into individual tokens:
 *       $$\phi_{char}: \Sigma^* \to V_{vocab}^*$$
 *       $$S \mapsto (t_1, t_2, \dots, t_N) \quad \text{where } t_i = \begin{cases} c_i & \text{if } c_i \in V_{vocab} \\ t_{unk} & \text{otherwise} \end{cases}$$</li>
 *   <li><b>Index mapping ($\psi$):</b> Maps a token to its vocabulary index:
 *       $$\psi: V_{vocab} \to I$$</li>
 *   <li><b>Encoding mapping ($E$):</b> Composites the tokenization and indexing:
 *       $$E(S) = (\psi(t_1), \psi(t_2), \dots, \psi(t_N))$$</li>
 *   <li><b>Decoding mapping ($D$):</b> Reconstructs the original string by string concatenation:
 *       $$D(y_1, y_2, \dots, y_N) = \bigoplus_{i=1}^N \psi^{-1}(y_i)$$
 *       where $\bigoplus$ represents the string concatenation operator.</li>
 * </ol>
 * 
 * <p>If the alphabet of $S$ is a subset of the vocabulary ($\{c_1, \dots, c_N\} \subseteq V_{vocab}$), 
 * then character-level encoding and decoding is perfectly lossless, satisfying $D(E(S)) = S$ exactly.</p>
 * 
 * <h4>Complexity Analysis:</h4>
 * <ul>
 *   <li><b>Tokenization/Encoding Time Complexity:</b> $O(N)$ where $N$ is the character length of the input text.</li>
 *   <li><b>Decoding Time Complexity:</b> $O(K)$ where $K$ is the number of token IDs to decode.</li>
 *   <li><b>Space Complexity:</b> $O(N)$ auxiliary space to allocate token sequences.</li>
 * </ul>
 */
public class CharacterTokenizer implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(CharacterTokenizer.class);

    private final List<String> idToToken;
    private final Map<String, Integer> tokenToId;

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
     * Constructs a CharacterTokenizer with the provided vocabulary of character strings.
     * Special tokens ([PAD], [UNK], [CLS], [SEP], [MASK]) will be verified and added if not present.
     * 
     * @param vocab the vocabulary as a list of strings (each typically a single character)
     * @throws IllegalArgumentException if the vocabulary is null or empty
     */
    public CharacterTokenizer(List<String> vocab) {
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
        for (int i = 0; i < idToToken.size(); i++) {
            tokenToId.put(idToToken.get(i), i);
        }

        this.padId = tokenToId.get(padToken);
        this.unkId = tokenToId.get(unkToken);
        this.clsId = tokenToId.get(clsToken);
        this.sepId = tokenToId.get(sepToken);
        this.maskId = tokenToId.get(maskToken);

        logger.info("Initializing CharacterTokenizer. Vocab size: {}, Special tokens: PAD={}, UNK={}, CLS={}, SEP={}, MASK={}",
                idToToken.size(), padId, unkId, clsId, sepId, maskId);
    }

    @Override
    public List<String> tokenize(String text) {
        if (text == null) {
            throw new IllegalArgumentException("Text cannot be null");
        }

        logger.debug("Tokenizing text at character level: '{}'", text);
        List<String> tokens = new ArrayList<>(text.length());
        
        for (int i = 0; i < text.length(); i++) {
            String cStr = String.valueOf(text.charAt(i));
            if (tokenToId.containsKey(cStr)) {
                logger.trace("Matched character: '{}' -> ID={}", cStr, tokenToId.get(cStr));
                tokens.add(cStr);
            } else {
                logger.trace("Unrecognized character: '{}' -> mapping to {}", cStr, unkToken);
                tokens.add(unkToken);
            }
        }

        logger.debug("Final character token list: {}", tokens);
        return tokens;
    }

    @Override
    public List<Integer> encode(String text) {
        logger.info("Encoding text at character level: '{}'", text);
        List<String> tokens = tokenize(text);
        List<Integer> ids = new ArrayList<>(tokens.size());
        for (String token : tokens) {
            ids.add(tokenToId.getOrDefault(token, unkId));
        }
        logger.debug("Encoded character IDs: {}", ids);
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

        logger.info("Decoding character token IDs: {} (skipSpecialTokens={})", ids, skipSpecialTokens);
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String token = idToToken(id);
            if (skipSpecialTokens && (token.equals(padToken) || token.equals(clsToken) || 
                                     token.equals(sepToken) || token.equals(maskToken))) {
                logger.trace("Skipping special token '{}' (id={}) during decode", token, id);
                continue;
            }
            sb.append(token);
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
