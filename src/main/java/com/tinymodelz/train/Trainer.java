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

/**
 * <h3>Trainer</h3>
 *
 * <p>Orchestrates training, evaluation, overfit testing, checkpointing,
 * metrics tracking, and autoregressive text generation for TinyGPT models.</p>
 *
 * <h4>Key Responsibilities:</h4>
 * <ul>
 *   <li><b>Forward Pass:</b> Computes model logits $\mathbf{z} \in \mathbb{R}^{B \times T \times |V|}$ from token input sequences.</li>
 *   <li><b>Loss Function:</b> Evaluates Cross-Entropy Loss $\mathcal{L} = -\frac{1}{N}\sum \log P(y_i | x_i)$ against target labels.</li>
 *   <li><b>Backward Pass:</b> Calculates gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ using Autograd backpropagation.</li>
 *   <li><b>Optimizer Step:</b> Updates weights $\theta_{t+1} = \theta_t - \eta \cdot m_t / (\sqrt{v_t} + \epsilon) - \eta \lambda \theta_t$ via AdamW.</li>
 *   <li><b>Metrics Computation:</b> Tracks Loss, Perplexity $\text{PPL} = e^{\mathcal{L}}$, Throughput ($\text{tokens/sec}$), and ETA.</li>
 *   <li><b>Checkpoint Persistence:</b> Persists model parameters to binary TMAT files at key checkpoints.</li>
 *   <li><b>Autoregressive Generation:</b> Evaluates model text generation capabilities periodically after epochs.</li>
 * </ul>
 */
public class Trainer {

    private static final Logger logger = LoggerFactory.getLogger(Trainer.class);

    private final TinyGPT model;
    private final Tokenizer tokenizer;
    private final AdamW optimizer;
    private final CrossEntropyLoss lossFn;
    private final Generator generator;

    /**
     * Constructs a Trainer instance for managing a TinyGPT model lifecycle.
     *
     * @param model     the TinyGPT model instance
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
    }

    /**
     * Executes an overfit test on a single batch until the loss drops below a target value
     * or maximum iterations are reached.
     *
     * @param loader     DataLoader containing token batches
     * @param maxSteps   maximum training steps to run
     * @param targetLoss loss threshold to achieve
     * @return final achieved loss value
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
            optimizer.step();

            currentLoss = loss.getData()[0];
            float perplexity = (float) Math.exp(currentLoss);

            if (step == 1 || step % 10 == 0 || currentLoss <= targetLoss || step == maxSteps) {
                logger.info(String.format("[Overfit Step %d/%d] loss = %.4f, PPL = %.2f", step, maxSteps, currentLoss, perplexity));
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
     * Executes full multi-epoch training over a dataset.
     *
     * @param loader          DataLoader configured with full dataset
     * @param epochs          number of full training epochs
     * @param checkpointDir   directory path to save model checkpoints after each epoch
     * @param prompt          text prompt for post-epoch generation sampling
     * @param generateTokens  number of new tokens to generate post-epoch
     * @param logInterval     batch frequency interval for printing detailed metrics
     */
    public void train(DataLoader loader, int epochs, File checkpointDir, String prompt, int generateTokens, int logInterval) {
        logger.info("==================================================");
        logger.info(String.format("Starting Full Training: %d Epochs, Batch Size: %d, Context Length: %d",
                epochs, loader.getBatchSize(), loader.getSeqLen()));
        logger.info("Model Config: " + model.summary());
        logger.info("==================================================");

        int totalBatchesPerEpoch = loader.getNumBatches();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            model.train();
            loader.reset();

            long epochStartTime = System.currentTimeMillis();
            float epochLossSum = 0.0f;
            int batchesProcessed = 0;
            int totalEpochTokens = 0;

            while (loader.hasNext()) {
                long batchStartTime = System.currentTimeMillis();
                Tensor[] batch = loader.nextBatch();
                Tensor x = batch[0];
                Tensor y = batch[1];

                optimizer.zeroGrad();
                Tensor logits = model.forward(x);
                Tensor loss = lossFn.forward(logits, y);
                loss.backward();
                optimizer.step();

                float lossVal = loss.getData()[0];
                epochLossSum += lossVal;
                batchesProcessed++;

                int batchTokens = x.getShape()[0] * x.getShape()[1];
                totalEpochTokens += batchTokens;

                long batchDurationMs = Math.max(1, System.currentTimeMillis() - batchStartTime);
                float tokPerSec = (batchTokens / (batchDurationMs / 1000.0f));

                if (batchesProcessed == 1 || batchesProcessed % logInterval == 0 || batchesProcessed == totalBatchesPerEpoch) {
                    float avgLossSoFar = epochLossSum / batchesProcessed;
                    float ppl = (float) Math.exp(avgLossSoFar);

                    long elapsedEpochMs = System.currentTimeMillis() - epochStartTime;
                    long remainingBatches = totalBatchesPerEpoch - batchesProcessed;
                    float avgBatchTimeMs = (float) elapsedEpochMs / batchesProcessed;
                    long etaMs = (long) (remainingBatches * avgBatchTimeMs);

                    logger.info(String.format(
                            "[Epoch %d/%d | Batch %d/%d] loss = %.4f | PPL = %.2f | lr = %.6f | tok/s = %.0f | ETA = %ds",
                            epoch, epochs, batchesProcessed, totalBatchesPerEpoch,
                            avgLossSoFar, ppl, optimizer.getLearningRate(), tokPerSec, etaMs / 1000
                    ));
                }
            }

            long totalEpochTimeMs = System.currentTimeMillis() - epochStartTime;
            float avgEpochLoss = epochLossSum / batchesProcessed;
            float epochPpl = (float) Math.exp(avgEpochLoss);
            float overallTokPerSec = (totalEpochTokens / (totalEpochTimeMs / 1000.0f));

            logger.info("==================================================");
            logger.info(String.format("=== Epoch %d/%d Complete ===", epoch, epochs));
            logger.info(String.format("  Avg Loss:      %.4f", avgEpochLoss));
            logger.info(String.format("  Perplexity:    %.2f", epochPpl));
            logger.info(String.format("  Total Time:    %.2f s", totalEpochTimeMs / 1000.0f));
            logger.info(String.format("  Throughput:    %.0f tokens/sec", overallTokPerSec));

            // --- Save Checkpoint ---
            if (checkpointDir != null) {
                File epochSaveDir = new File(checkpointDir, "epoch_" + epoch);
                try {
                    Checkpoint.saveCheckpoint(model, epochSaveDir);
                    logger.info("  Checkpoint:    {}", epochSaveDir.getAbsolutePath());
                } catch (IOException e) {
                    logger.error("Failed to save checkpoint for epoch {}: {}", epoch, e.getMessage());
                }
            }

            // --- Sample Text Generation ---
            if (prompt != null && generateTokens > 0) {
                int eosCandidate = tokenizer.tokenToId("<|endoftext|>");
                Integer eosId = (eosCandidate == tokenizer.tokenToId("[UNK]")) ? null : eosCandidate;

                String sampleOutput = generator.generate(
                        model, tokenizer, prompt, generateTokens, 0.8f, 40, 0.9f, model.getMaxSeqLen(), eosId
                );
                logger.info("  Generated Sample (prompt='{}'):\n\"{}\"", prompt, sampleOutput.trim());
            }
            logger.info("==================================================");
        }
    }

    private RuntimeException idleDatasetException() {
        return new IllegalStateException("DataLoader has no batches available.");
    }
}
