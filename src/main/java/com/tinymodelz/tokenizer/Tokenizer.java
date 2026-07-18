package com.tinymodelz.tokenizer;

import java.util.List;

/**
 * <h3>Tokenizer</h3>
 * 
 * <p>The base interface for language model tokenizers.
 * A tokenizer acts as the interface between raw textual strings and discrete numerical representations 
 * suitable for machine learning models.</p>
 * 
 * <h4>Mathematical Formulation:</h4>
 * 
 * <p>Let $\Sigma^*$ be the set of all finite-length strings over the character alphabet $\Sigma$.
 * Let $V_{vocab} = \{t_0, t_1, \dots, t_{M-1}\}$ be the vocabulary of size $M = |V_{vocab}|$.
 * Let $I = \{0, 1, \dots, M-1\}$ be the set of token indices (IDs).</p>
 * 
 * <p>We define the following mappings:</p>
 * <ol>
 *   <li><b>Tokenization mapping ($\phi$):</b> Decomposes a string $S \in \Sigma^*$ into a sequence of vocabulary tokens:
 *       $$\phi: \Sigma^* \to V_{vocab}^*$$
 *       where $S \mapsto (x_1, x_2, \dots, x_n)$ such that the reconstructed text $\bigoplus_{i=1}^n x_i$ is close or identical to $S$ 
 *       (under the tokenizer's merge or split logic).</li>
 *   <li><b>Index mapping ($\psi$):</b> Maps a token to its vocabulary index:
 *       $$\psi: V_{vocab} \to I$$
 *       In case of an unknown token, $\psi(t_{unk}) = \text{ID}_{unk}$.</li>
 *   <li><b>Encoding mapping ($E$):</b> Composites the tokenization and indexing:
 *       $$E: \Sigma^* \to I^*$$
 *       where $E(S) = (\psi(x_1), \psi(x_2), \dots, \psi(x_n))$.</li>
 *   <li><b>Decoding mapping ($D$):</b> Reconstructs the textual representation from index sequences:
 *       $$D: I^* \to \Sigma^*$$
 *       where $D(y_1, y_2, \dots, y_m) = \text{detokenize}(\psi^{-1}(y_1), \psi^{-1}(y_2), \dots, \psi^{-1}(y_m))$.</li>
 * </ol>
 * 
 * <p>Ideally, $D(E(S)) \approx S$. In lossless tokenizers (e.g. byte-level tokenizers), $D(E(S)) = S$ holds exactly.</p>
 */
public interface Tokenizer {

    /**
     * Decomposes raw text into a list of token strings.
     * 
     * @param text the input string to tokenize
     * @return a list of tokens
     */
    List<String> tokenize(String text);

    /**
     * Encodes raw text into a list of vocabulary token IDs.
     * 
     * @param text the input string to encode
     * @return a list of token IDs
     */
    List<Integer> encode(String text);

    /**
     * Decodes a list of token IDs back into a single human-readable string.
     * 
     * @param ids the list of token IDs to decode
     * @return the decoded string
     */
    String decode(List<Integer> ids);

    /**
     * Gets the total size of the vocabulary.
     * 
     * @return the vocabulary size
     */
    int getVocabSize();

    /**
     * Converts a token ID to its corresponding string token.
     * 
     * @param id the token ID
     * @return the token string, or the unknown token representation if the ID is invalid
     */
    String idToToken(int id);

    /**
     * Converts a token string to its corresponding token ID.
     * 
     * @param token the token string
     * @return the token ID, or the unknown token ID if the token is not in the vocabulary
     */
    int tokenToId(String token);
}
