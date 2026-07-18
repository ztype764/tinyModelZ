package com.tinymodelz.train;

import com.tinymodelz.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * <h3>Text Dataset</h3>
 * 
 * <p>Encapsulates a textual dataset by encoding a raw text string or file
 * into a contiguous sequence of token IDs using a {@link Tokenizer}.</p>
 */
public class TextDataset {

    private final int[] tokenIds;

    /**
     * Constructs a TextDataset from a raw string.
     * 
     * @param text the raw input text
     * @param tokenizer the tokenizer to use for encoding
     */
    public TextDataset(String text, Tokenizer tokenizer) {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }
        if (tokenizer == null) {
            throw new IllegalArgumentException("Tokenizer cannot be null");
        }
        List<Integer> encoded = tokenizer.encode(text);
        this.tokenIds = new int[encoded.size()];
        for (int i = 0; i < encoded.size(); i++) {
            this.tokenIds[i] = encoded.get(i);
        }
    }

    /**
     * Constructs a TextDataset by reading from a file path.
     * 
     * @param filePath the path to the text file
     * @param tokenizer the tokenizer to use for encoding
     * @throws IOException if the file cannot be read
     */
    public TextDataset(Path filePath, Tokenizer tokenizer) throws IOException {
        this(Files.readString(filePath), tokenizer);
    }

    /**
     * Returns the array of token IDs.
     * 
     * @return the token IDs array
     */
    public int[] getTokenIds() {
        return tokenIds;
    }

    /**
     * Returns the number of tokens in the dataset.
     * 
     * @return the total token count
     */
    public int size() {
        return tokenIds.length;
    }
}
