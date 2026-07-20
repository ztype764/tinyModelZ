package com.tinymodelz.train;

import com.tinymodelz.inference.Generator;
import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.nn.TinyGPT;
import com.tinymodelz.tokenizer.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * <h3>Trainer</h3>
 *
 * <p>
 * Orchestrates training, evaluation, overfit testing, gradient clipping,
 * learning rate scheduling, profiling, checkpointing, and text generation.
 * </p>
 */
public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    private final TinyGPT model;
    private final Tokenizer tokenizer;
    private final AdamW optimizer;
    private final CrossEntropyLoss lossFn;
    private final Generator generator;
    private final Profiler profiler;

    private float maxGradNorm = 1.0f; // Default L2 gradient clipping threshold

    /**
     * Constructs a Trainer instance.
     *
     * @param model     the TinyGPT model
     * @param tokenizer the dataset tokenizer
     * @param optimizer the AdamW optimizer
     * @param lossFn    the CrossEntropyLoss criterion
     */
    public Trainer(TinyGPT model, Tokenizer tokenizer, AdamW optimizer, CrossEntropyLoss lossFn) {
        if (model == null || tokenizer == null || optimizer == null || lossFn == null) {
            throw new IllegalArgumentException("Trainer components cannot be null.");
        }
        this.model = model;
        this.tokenizer = tokenizer;
        this.optimizer = optimizer;
        this.lossFn = lossFn;
        this.generator = new Generator(42L);
        this.profiler = new Profiler();
    }

    /**
     * Sets the maximum gradient L2 norm for gradient clipping.
     *
     * @param maxGradNorm clipping threshold (e.g. 1.0f)
     */
    public void setMaxGradNorm(float maxGradNorm) {
        this.maxGradNorm = maxGradNorm;
    }

    /**
     * Performs L2 gradient clipping across all trainable model parameters.
     *
     * @param maxNorm maximum allowed global gradient norm
     * @return original unclipped global gradient norm
     */
    public float clipGradients(float maxNorm) {
        if (maxNorm <= 0.0f)
            return 0.0f;

        List<Tensor> params = model.getParameters();
        double sumSq = 0.0;

        for (Tensor p : params) {
            if (!p.requiresGrad())
                continue;
            float[] grad = p.getGrad();
            if (grad == null)
                continue;
            for (float g : grad) {
                sumSq += g * g;
            }
        }

        float totalNorm = (float) Math.sqrt(sumSq);
        if (totalNorm > maxNorm) {
            float scale = maxNorm / (totalNorm + 1e-6f);
            for (Tensor p : params) {
                if (!p.requiresGrad())
                    continue;
                float[] grad = p.getGrad();
                if (grad == null)
                    continue;
                for (int i = 0; i < grad.length; i++) {
                    grad[i] *= scale;
                }
            }
        }
        return totalNorm;
    }

    /**
     * Evaluates the model loss and perplexity on a validation dataset.
     *
     * @param valLoader DataLoader configured with validation split
     * @return validation loss value
     */
    public float evaluate(DataLoader valLoader) {
        if (valLoader == null)
            return Float.MAX_VALUE;

        model.eval();
        valLoader.reset();

        float totalLoss = 0.0f;
        int count = 0;

        while (valLoader.hasNext()) {
            Tensor[] batch = valLoader.nextBatch();
            Tensor x = batch[0];
            Tensor y = batch[1];

            Tensor logits = model.forward(x);
            Tensor loss = lossFn.forward(logits, y);
            totalLoss += loss.getData()[0];
            count++;
        }

        float avgValLoss = count > 0 ? totalLoss / count : Float.MAX_VALUE;
        float valPpl = (float) Math.exp(avgValLoss);
        logger.info(String.format("=== Validation Evaluation === Loss: %.4f | PPL: %.2f (%d batches)", avgValLoss,
                valPpl, count));
        return avgValLoss;
    }

    /**
     * Executes an overfit test on a single batch.
     */
    public float trainOverfit(DataLoader loader, int maxSteps, float targetLoss) {
        logger.info("==================================================");
        logger.info("Starting Single-Batch Overfit Test (targetLoss <= {}, maxSteps = {})", targetLoss, maxSteps);
        logger.info("==================================================");

        model.train();
        loader.reset();
        if (!loader.hasNext()) {
            throw idleDatasetException();
        }

        Tensor[] fixedBatch = loader.nextBatch();
        Tensor x = fixedBatch[0];
        Tensor y = fixedBatch[1];

        float currentLoss = Float.MAX_VALUE;
        long startTime = System.currentTimeMillis();

        for (int step = 1; step <= maxSteps; step++) {
            optimizer.zeroGrad();
            Tensor logits = model.forward(x);
            Tensor loss = lossFn.forward(logits, y);
            loss.backward();

            clipGradients(maxGradNorm);
            optimizer.step();

            currentLoss = loss.getData()[0];
            float perplexity = (float) Math.exp(currentLoss);

            if (step == 1 || step % 10 == 0 || currentLoss <= targetLoss || step == maxSteps) {
                logger.info(String.format("[Overfit Step %d/%d] loss = %.4f, PPL = %.2f", step, maxSteps, currentLoss,
                        perplexity));
            }

            if (currentLoss <= targetLoss) {
                logger.info(String.format("Overfit test SUCCESS: achieved loss %.4f <= target %.4f in %d steps (%d ms)",
                        currentLoss, targetLoss, step, System.currentTimeMillis() - startTime));
                break;
            }
        }
        return currentLoss;
    }

    /**
     * Executes full multi-epoch training over a dataset with profiling, learning
     * rate scheduling,
     * validation evaluation, and best checkpoint retention.
     */
    public void train(DataLoader loader, DataLoader valLoader, int epochs, File checkpointDir, List<String> prompts,
            int generateTokens, int logInterval) {
        train(loader, valLoader, 1, epochs, checkpointDir, prompts, generateTokens, logInterval);
    }

    /**
     * Executes multi-epoch training starting from a specified initial epoch number (supports resuming).
     */
    public void train(DataLoader loader, DataLoader valLoader, int startEpoch, int epochs, File checkpointDir, List<String> prompts,
            int generateTokens, int logInterval) {
        logger.info("==================================================");
        logger.info(String.format("Starting Training: Epoch %d to %d, Batch Size: %d, Context Length: %d",
                startEpoch, epochs, loader.getBatchSize(), loader.getSeqLen()));
        logger.info("Tokenizer in use: " + tokenizer.getClass().getSimpleName() + " (vocabSize: " + tokenizer.getVocabSize() + ")");
        logger.info("Model Config: " + model.summary());
        logger.info("==================================================");

        int totalBatchesPerEpoch = loader.getNumBatches();
        int totalTrainingSteps = totalBatchesPerEpoch * epochs;
        int warmupSteps = (int) (totalTrainingSteps * 0.05f); // 5% warmup steps

        LRScheduler scheduler = new LRScheduler(optimizer, optimizer.getLearningRate(), 1e-5f, warmupSteps,
                totalTrainingSteps);
        scheduler.setCurrentStep(optimizer.getStepCount());
        float bestValLoss = Float.MAX_VALUE;

        for (int epoch = startEpoch; epoch <= epochs; epoch++) {
            profiler.resetEpoch();
            model.train();
            loader.reset();

            long epochStartTimeNanos = System.nanoTime();
            float epochLossSum = 0.0f;
            int batchesProcessed = 0;
            int totalEpochTokens = 0;

            while (loader.hasNext()) {
                long tPrepStart = System.nanoTime();
                Tensor[] batch = loader.nextBatch();
                Tensor x = batch[0];
                Tensor y = batch[1];
                profiler.addBatchPrepTime(System.nanoTime() - tPrepStart);

                optimizer.zeroGrad();

                long tFwdStart = System.nanoTime();
                Tensor logits = model.forward(x);
                Tensor loss = lossFn.forward(logits, y);
                profiler.addForwardTime(System.nanoTime() - tFwdStart);

                long tBwdStart = System.nanoTime();
                loss.backward();
                profiler.addBackwardTime(System.nanoTime() - tBwdStart);

                long tOptStart = System.nanoTime();
                clipGradients(maxGradNorm);
                optimizer.step();
                scheduler.step();
                profiler.addOptimizerTime(System.nanoTime() - tOptStart);

                float lossVal = loss.getData()[0];
                epochLossSum += lossVal;
                batchesProcessed++;

                int batchTokens = x.getShape()[0] * x.getShape()[1];
                totalEpochTokens += batchTokens;

                if (batchesProcessed == 1 || batchesProcessed % logInterval == 0
                        || batchesProcessed == totalBatchesPerEpoch) {
                    float avgLossSoFar = epochLossSum / batchesProcessed;
                    float ppl = (float) Math.exp(avgLossSoFar);

                    long elapsedMs = (System.nanoTime() - epochStartTimeNanos) / 1_000_000;
                    long remainingBatches = totalBatchesPerEpoch - batchesProcessed;
                    float avgBatchTimeMs = (float) elapsedMs / batchesProcessed;
                    long etaSec = (long) ((remainingBatches * avgBatchTimeMs) / 1000.0f);
                    float tokPerSec = (totalEpochTokens / (Math.max(1, elapsedMs) / 1000.0f));

                    logger.info(String.format(
                            "[Epoch %d/%d | Batch %d/%d] loss = %.4f | PPL = %.2f | lr = %.6f | tok/s = %.0f | ETA = %ds",
                            epoch, epochs, batchesProcessed, totalBatchesPerEpoch,
                            avgLossSoFar, ppl, scheduler.getLearningRate(), tokPerSec, etaSec));
                }
            }

            long totalEpochTimeNanos = System.nanoTime() - epochStartTimeNanos;
            profiler.setTotalEpochTime(totalEpochTimeNanos);

            float avgEpochLoss = epochLossSum / batchesProcessed;
            float epochPpl = (float) Math.exp(avgEpochLoss);
            float overallTokPerSec = (totalEpochTokens / ((totalEpochTimeNanos / 1_000_000.0f) / 1000.0f));

            logger.info("==================================================");
            logger.info(String.format("=== Epoch %d/%d Complete ===", epoch, epochs));
            logger.info(String.format("  Train Loss:    %.4f", avgEpochLoss));
            logger.info(String.format("  Perplexity:    %.2f", epochPpl));
            logger.info(String.format("  Throughput:    %.0f tokens/sec", overallTokPerSec));

            // --- Phase 1 Profiler Report ---
            profiler.printSummary(epoch, epochs);

            // --- Phase 4 & Phase 9 Best Checkpoint ---
            if (valLoader != null) {
                float valLoss = evaluate(valLoader);
                if (valLoss < bestValLoss && checkpointDir != null) {
                    bestValLoss = valLoss;
                    File bestDir = new File(checkpointDir, "best_checkpoint");
                    try {
                        Checkpoint.saveCheckpoint(model, optimizer, epoch, optimizer.getStepCount(), bestDir);
                        logger.info("  [NEW BEST MODEL] Saved best checkpoint to: {}", bestDir.getAbsolutePath());
                    } catch (IOException e) {
                        logger.error("Failed to save best checkpoint: {}", e.getMessage());
                    }
                }
            }

            // --- Phase 4 & Phase 9 Epoch Checkpoint ---
            if (checkpointDir != null) {
                long tChkStart = System.nanoTime();
                File epochSaveDir = new File(checkpointDir, "epoch_" + epoch);
                try {
                    Checkpoint.saveCheckpoint(model, optimizer, epoch, optimizer.getStepCount(), epochSaveDir);
                    logger.info("  Saved Checkpoint: {}", epochSaveDir.getAbsolutePath());
                } catch (IOException e) {
                    logger.error("Failed to save epoch checkpoint: {}", e.getMessage());
                }
                profiler.addCheckpointTime(System.nanoTime() - tChkStart);
            }

            // --- Autoregressive Text Generation Samples ---
            if (prompts != null && !prompts.isEmpty() && generateTokens > 0) {
                int eosCandidate = tokenizer.tokenToId("<|endoftext|>");
                Integer eosId = (eosCandidate == tokenizer.tokenToId("[UNK]")) ? null : eosCandidate;

                logger.info("  --- Post-Epoch {} Text Generation Samples ---", epoch);
                for (String p : prompts) {
                    if (p == null || p.isBlank()) continue;
                    String sampleOutput = generator.generate(
                            model, tokenizer, p, generateTokens, 0.8f, 40, 0.9f, model.getMaxSeqLen(), eosId);
                    logger.info("  [Prompt: '{}'] -> \"{}\"", p, sampleOutput.trim());
                }
            }
            logger.info("==================================================");
        }
    }

    /**
     * Overloaded training method taking a single prompt for backwards compatibility.
     */
    public void train(DataLoader loader, DataLoader valLoader, int epochs, File checkpointDir, String singlePrompt,
            int generateTokens, int logInterval) {
        List<String> prompts = (singlePrompt != null) ? List.of(singlePrompt) : List.of();
        this.train(loader, valLoader, epochs, checkpointDir, prompts, generateTokens, logInterval);
    }

    /**
     * Legacy single-loader entry point with list of prompts.
     */
    public void train(DataLoader loader, int epochs, File checkpointDir, List<String> prompts, int generateTokens,
            int logInterval) {
        train(loader, null, epochs, checkpointDir, prompts, generateTokens, logInterval);
    }

    /**
     * Legacy single-loader entry point with single prompt for backwards compatibility.
     */
    public void train(DataLoader loader, int epochs, File checkpointDir, String prompt, int generateTokens,
            int logInterval) {
        List<String> prompts = (prompt != null) ? List.of(prompt) : List.of();
        train(loader, null, epochs, checkpointDir, prompts, generateTokens, logInterval);
    }

    private RuntimeException idleDatasetException() {
        return new IllegalStateException("DataLoader has no batches available.");
    }
}
