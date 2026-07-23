package com.tinymodelz.train;

import com.tinymodelz.math.Tensor;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * <h3>DataLoader</h3>
 * 
 * <p>
 * Generates mini-batches from a {@link TextDataset}.
 * </p>
 * <p>
 * Produces input tensors [B, T] and target tensors [B, T] shifted by one token
 * for next-token prediction training. Supports random shuffling per epoch.
 * </p>
 * <p>
 * Optimized to use zero memory allocations when shuffle is disabled, and primitive
 * integer arrays when shuffle is enabled, avoiding OutOfMemoryError on multi-hundred-million token datasets.
 * </p>
 */
public class DataLoader {

    private final int[] tokenIds;
    private final int batchSize;
    private final int seqLen;
    private final int stride;
    private final boolean shuffle;

    private final int totalSamples;
    private final int[] startOffsets;
    private int cursor = 0;

    /**
     * Constructs a DataLoader with default sequence stride of 1.
     * 
     * @param dataset   the dataset to load from
     * @param batchSize number of sequences per batch (B)
     * @param seqLen    length of each sequence (T)
     * @param shuffle   whether to shuffle starting indices at the start of epoch
     */
    public DataLoader(TextDataset dataset, int batchSize, int seqLen, boolean shuffle) {
        this(dataset, batchSize, seqLen, 1, shuffle);
    }

    /**
     * Constructs a DataLoader with a custom sampling stride between sequence start positions.
     * 
     * @param dataset   the dataset to load from
     * @param batchSize number of sequences per batch (B)
     * @param seqLen    length of each sequence (T)
     * @param stride    step distance between starting sequence indices (e.g. 1 or seqLen)
     * @param shuffle   whether to shuffle starting indices at the start of epoch
     */
    public DataLoader(TextDataset dataset, int batchSize, int seqLen, int stride, boolean shuffle) {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset cannot be null");
        }
        if (dataset.getTokenIds() == null || dataset.getTokenIds().length == 0) {
            throw new IllegalArgumentException("Dataset contains no tokens");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        if (seqLen <= 0) {
            throw new IllegalArgumentException("Sequence length must be positive");
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("Stride must be positive");
        }
        this.tokenIds = dataset.getTokenIds();
        this.batchSize = batchSize;
        this.seqLen = seqLen;
        this.stride = stride;
        this.shuffle = shuffle;

        if (tokenIds.length < seqLen + 1) {
            throw new IllegalArgumentException("Dataset token count (" + tokenIds.length
                    + ") is insufficient for sequence length " + seqLen + " (requires at least " + (seqLen + 1) + " tokens)");
        }

        int maxStartOffset = tokenIds.length - seqLen - 1;
        this.totalSamples = (maxStartOffset / stride) + 1;
        if (totalSamples < batchSize) {
            throw new IllegalArgumentException("Dataset available sequence samples (" + totalSamples
                    + ") is smaller than batch size (" + batchSize + ")");
        }

        if (shuffle) {
            this.startOffsets = new int[totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                this.startOffsets[i] = i * stride;
            }
        } else {
            this.startOffsets = null;
        }

        reset();
    }

    /**
     * Resets the cursor and reshuffles the start offsets if shuffle mode is enabled.
     */
    public void reset() {
        if (shuffle && startOffsets != null) {
            Random rnd = ThreadLocalRandom.current();
            for (int i = startOffsets.length - 1; i > 0; i--) {
                int index = rnd.nextInt(i + 1);
                int temp = startOffsets[index];
                startOffsets[index] = startOffsets[i];
                startOffsets[i] = temp;
            }
        }
        cursor = 0;
    }

    /**
     * Checks if there are enough remaining samples to form a complete batch.
     * 
     * @return true if a next batch exists, false otherwise
     */
    public boolean hasNext() {
        return cursor + batchSize <= totalSamples;
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

        for (int b = 0; b < batchSize; b++) {
            int sampleIdx = cursor + b;
            int startIdx = shuffle ? startOffsets[sampleIdx] : sampleIdx * stride;
            for (int t = 0; t < seqLen; t++) {
                xData[b * seqLen + t] = tokenIds[startIdx + t];
                yData[b * seqLen + t] = tokenIds[startIdx + t + 1];
            }
        }

        cursor += batchSize;

        Tensor x = new Tensor(xData, new int[] { batchSize, seqLen });
        Tensor y = new Tensor(yData, new int[] { batchSize, seqLen });

        return new Tensor[] { x, y };
    }

    /**
     * Returns the total number of complete batches available in the dataset.
     * 
     * @return number of batches
     */
    public int getNumBatches() {
        return totalSamples / batchSize;
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

    /**
     * Gets the sequence sampling stride.
     * 
     * @return stride
     */
    public int getStride() {
        return stride;
    }

    /**
     * Gets the total number of valid sequence start positions.
     * 
     * @return total samples
     */
    public int getTotalSamples() {
        return totalSamples;
    }
}
