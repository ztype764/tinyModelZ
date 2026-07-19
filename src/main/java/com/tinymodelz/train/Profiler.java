package com.tinymodelz.train;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <h3>Profiler</h3>
 *
 * <p>Provides high-precision timing, memory utilization tracking, and granular
 * performance profiling for the TinyModelZ training pipeline.</p>
 */
public class Profiler {

    private static final Logger logger = LoggerFactory.getLogger(Profiler.class);

    private long forwardTimeNs = 0;
    private long backwardTimeNs = 0;
    private long optimizerTimeNs = 0;
    private long batchPrepTimeNs = 0;
    private long checkpointTimeNs = 0;
    private long totalEpochTimeNs = 0;

    /**
     * Resets timing accumulators at the start of an epoch.
     */
    public void resetEpoch() {
        forwardTimeNs = 0;
        backwardTimeNs = 0;
        optimizerTimeNs = 0;
        batchPrepTimeNs = 0;
        checkpointTimeNs = 0;
        totalEpochTimeNs = 0;
    }

    public void addForwardTime(long nanos) {
        forwardTimeNs += nanos;
    }

    public void addBackwardTime(long nanos) {
        backwardTimeNs += nanos;
    }

    public void addOptimizerTime(long nanos) {
        optimizerTimeNs += nanos;
    }

    public void addBatchPrepTime(long nanos) {
        batchPrepTimeNs += nanos;
    }

    public void addCheckpointTime(long nanos) {
        checkpointTimeNs += nanos;
    }

    public void setTotalEpochTime(long nanos) {
        totalEpochTimeNs = nanos;
    }

    /**
     * Measures current JVM Heap RAM usage in Megabytes.
     *
     * @return peak allocated memory in MB
     */
    public static float getUsedMemoryMb() {
        Runtime runtime = Runtime.getRuntime();
        long usedBytes = runtime.totalMemory() - runtime.freeMemory();
        return usedBytes / (1024.0f * 1024.0f);
    }

    /**
     * Prints a detailed timing and performance breakdown after an epoch completes.
     *
     * @param epoch current epoch number
     * @param epochs total epoch count
     */
    public void printSummary(int epoch, int epochs) {
        float forwardMs = forwardTimeNs / 1_000_000.0f;
        float backwardMs = backwardTimeNs / 1_000_000.0f;
        float optMs = optimizerTimeNs / 1_000_000.0f;
        float batchMs = batchPrepTimeNs / 1_000_000.0f;
        float chkMs = checkpointTimeNs / 1_000_000.0f;
        float totalMs = totalEpochTimeNs / 1_000_000.0f;
        float usedRamMb = getUsedMemoryMb();

        logger.info(String.format("--- Profiling Summary [Epoch %d/%d] ---", epoch, epochs));
        logger.info(String.format("  Forward Pass:    %8.2f ms (%4.1f%%)", forwardMs, totalMs > 0 ? (forwardMs / totalMs) * 100 : 0));
        logger.info(String.format("  Backward Pass:   %8.2f ms (%4.1f%%)", backwardMs, totalMs > 0 ? (backwardMs / totalMs) * 100 : 0));
        logger.info(String.format("  Optimizer Step:  %8.2f ms (%4.1f%%)", optMs, totalMs > 0 ? (optMs / totalMs) * 100 : 0));
        logger.info(String.format("  Batch Prep:      %8.2f ms (%4.1f%%)", batchMs, totalMs > 0 ? (batchMs / totalMs) * 100 : 0));
        if (chkMs > 0) {
            logger.info(String.format("  Checkpointing:   %8.2f ms (%4.1f%%)", chkMs, totalMs > 0 ? (chkMs / totalMs) * 100 : 0));
        }
        logger.info(String.format("  Peak RAM Usage:  %8.2f MB", usedRamMb));
        logger.info("----------------------------------------");
    }
}
