package com.tinymodelz.train;

import com.tinymodelz.math.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * <h3>DataLoader</h3>
 * 
 * <p>Generates mini-batches from a {@link TextDataset}.</p>
 * <p>Produces input tensors [B, T] and target tensors [B, T] shifted by one token
 * for next-token prediction training. Supports random shuffling per epoch.</p>
 */
public class DataLoader {

    private final TextDataset dataset;
    private final int batchSize;
    private final int seqLen;
    private final boolean shuffle;

    private List<Integer> startOffsets;
    private int cursor = 0;

    /**
     * Constructs a DataLoader.
     * 
     * @param dataset the dataset to load from
     * @param batchSize number of sequences per batch (B)
     * @param seqLen length of each sequence (T)
     * @param shuffle whether to shuffle starting indices at the start of epoch
     */
    public DataLoader(TextDataset dataset, int batchSize, int seqLen, boolean shuffle) {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset cannot be null");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        if (seqLen <= 0) {
            throw new IllegalArgumentException("Sequence length must be positive");
        }
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.seqLen = seqLen;
        this.shuffle = shuffle;

        reset();
    }

    /**
     * Resets the cursor and reshuffles the start offsets if shuffle mode is enabled.
     */
    public void reset() {
        int[] ids = dataset.getTokenIds();
        int maxStartOffset = ids.length - seqLen - 1;

        startOffsets = new ArrayList<>();
        for (int i = 0; i <= maxStartOffset; i++) {
            startOffsets.add(i);
        }

        if (shuffle) {
            Collections.shuffle(startOffsets);
        }
        cursor = 0;
    }

    /**
     * Checks if there are enough remaining samples to form a complete batch.
     * 
     * @return true if a next batch exists, false otherwise
     */
    public boolean hasNext() {
        return cursor + batchSize <= startOffsets.size();
    }

    /**
     * Retrieves the next batch of input and target tensors.
     * 
     * @return an array of two Tensors: [inputs, targets], each of shape [batchSize, seqLen]
     */
    public Tensor[] nextBatch() {
        if (!hasNext()) {
            reset();
        }

        float[] xData = new float[batchSize * seqLen];
        float[] yData = new float[batchSize * seqLen];
        int[] ids = dataset.getTokenIds();

        for (int b = 0; b < batchSize; b++) {
            int startIdx = startOffsets.get(cursor + b);
            for (int t = 0; t < seqLen; t++) {
                xData[b * seqLen + t] = ids[startIdx + t];
                yData[b * seqLen + t] = ids[startIdx + t + 1];
            }
        }

        cursor += batchSize;

        Tensor x = new Tensor(xData, new int[]{batchSize, seqLen});
        Tensor y = new Tensor(yData, new int[]{batchSize, seqLen});

        return new Tensor[]{x, y};
    }

    /**
     * Returns the total number of complete batches available in the dataset.
     * 
     * @return number of batches
     */
    public int getNumBatches() {
        return startOffsets.size() / batchSize;
    }

    /**
     * Gets the current batch size.
     * 
     * @return batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Gets the sequence length.
     * 
     * @return sequence length
     */
    public int getSeqLen() {
        return seqLen;
    }
}
