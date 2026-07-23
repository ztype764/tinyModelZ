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
     * Constructs a TextDataset by streaming line-by-line from a file path.
     * This avoids allocating multi-gigabyte String objects and prevents NegativeArraySizeException / OOM on large files.
     * 
     * @param filePath the path to the text file
     * @param tokenizer the tokenizer to use for encoding
     * @throws IOException if the file cannot be read
     */
    public TextDataset(Path filePath, Tokenizer tokenizer) throws IOException {
        if (filePath == null || !Files.exists(filePath)) {
            throw new IllegalArgumentException("File path cannot be null or non-existent: " + filePath);
        }
        if (tokenizer == null) {
            throw new IllegalArgumentException("Tokenizer cannot be null");
        }

        int capacity = 1024 * 1024;
        int[] buffer = new int[capacity];
        int size = 0;

        try (java.io.BufferedReader reader = Files.newBufferedReader(filePath, java.nio.charset.StandardCharsets.UTF_8)) {
            String line;
            boolean firstLine = true;
            while ((line = reader.readLine()) != null) {
                if (!firstLine) {
                    List<Integer> nlEnc = tokenizer.encode("\n");
                    for (int id : nlEnc) {
                        if (size >= buffer.length) {
                            buffer = java.util.Arrays.copyOf(buffer, buffer.length * 2);
                        }
                        buffer[size++] = id;
                    }
                }
                firstLine = false;

                if (line.isEmpty()) continue;

                List<Integer> encoded = tokenizer.encode(line);
                for (int id : encoded) {
                    if (size >= buffer.length) {
                        buffer = java.util.Arrays.copyOf(buffer, buffer.length * 2);
                    }
                    buffer[size++] = id;
                }
            }
        }

        if (size == 0) {
            throw new IllegalArgumentException("Dataset file contains no tokens: " + filePath);
        }

        this.tokenIds = java.util.Arrays.copyOf(buffer, size);
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
