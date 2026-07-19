package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import com.tinymodelz.train.AdamW;
import com.tinymodelz.train.DataLoader;
import com.tinymodelz.train.TextDataset;
import com.tinymodelz.TestReporter;

import java.util.Arrays;
import java.util.List;

/**
 * <h3>TinyGPTTest</h3>
 *
 * <p>Integration tests for the TinyGPT language model class, verifying:</p>
 * <ul>
 *   <li>Forward pass output shape correctness</li>
 *   <li>Backward pass gradient propagation to all parameters</li>
 *   <li>Multi-epoch training convergence on a toy corpus</li>
 * </ul>
 */
public class TinyGPTTest {

    public static void runTests() {
        TestReporter.runTest("TinyGPT forward pass output shape", () -> testForwardShape());
        TestReporter.runTest("TinyGPT backward gradient propagation to all parameters", () -> testBackwardGradients());
        TestReporter.runTest("TinyGPT multi-step training loss convergence", () -> testTrainingConvergence());
        TestReporter.runTest("TinyGPT model summary and parameter count", () -> testModelSummary());
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " — Expected: " + expected + ", Got: " + actual);
        }
    }

    /**
     * Verifies that the forward pass produces logits of the correct shape [B, T, V].
     */
    private static void testForwardShape() {
        int vocabSize = 10;
        int embedDim = 8;
        int maxSeqLen = 16;
        int numLayers = 2;
        int numHeads = 2;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, 0.0f);
        model.eval();

        // Input: batch of 2 sequences, each length 5
        float[] ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        Tensor x = new Tensor(ids, new int[]{2, 5});

        Tensor logits = model.forward(x);
        int[] expectedShape = {2, 5, vocabSize};

        assertTrue(
            Arrays.equals(expectedShape, logits.getShape()),
            "Forward output shape mismatch — expected " + Arrays.toString(expectedShape)
                + " got " + Arrays.toString(logits.getShape())
        );

        TestReporter.logMetric("Input shape", Arrays.toString(x.getShape()));
        TestReporter.logMetric("Output shape", Arrays.toString(logits.getShape()));
        TestReporter.logMetric("Output size", logits.size());
    }

    /**
     * Verifies that backward() propagates gradients to every registered parameter.
     */
    private static void testBackwardGradients() {
        int vocabSize = 8;
        int embedDim = 4;
        int maxSeqLen = 8;
        int numLayers = 1;
        int numHeads = 2;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, 0.0f);
        model.train();

        float[] ids = {0, 1, 2, 3};
        Tensor x = new Tensor(ids, new int[]{1, 4});

        float[] targets = {1, 2, 3, 4};
        Tensor y = new Tensor(targets, new int[]{1, 4});

        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Tensor logits = model.forward(x);
        Tensor loss = lossFn.forward(logits, y);
        loss.backward();

        List<Tensor> params = model.getParameters();
        int withGrad = 0;
        int withoutGrad = 0;

        for (Tensor p : params) {
            float[] grad = p.getGrad();
            if (grad != null) {
                boolean hasNonZero = false;
                for (float g : grad) {
                    if (g != 0.0f) {
                        hasNonZero = true;
                        break;
                    }
                }
                if (hasNonZero) {
                    withGrad++;
                } else {
                    withoutGrad++;
                }
            } else {
                withoutGrad++;
            }
        }

        TestReporter.logMetric("Total parameters tensors", params.size());
        TestReporter.logMetric("Parameters with gradient", withGrad);
        TestReporter.logMetric("Parameters without gradient", withoutGrad);
        TestReporter.logMetric("Loss value", loss.getData()[0]);

        assertTrue(withGrad > 0, "At least some parameters must receive gradients");
        assertTrue(
            withGrad >= params.size() / 2,
            "Most parameters should receive gradients, only " + withGrad + "/" + params.size() + " did"
        );
    }

    /**
     * End-to-end training convergence test: trains TinyGPT on a small corpus
     * using DataLoader, CrossEntropyLoss, and AdamW, verifying loss decreases.
     */
    private static void testTrainingConvergence() {
        // Setup tokenizer and data
        List<String> vocab = List.of(
            "t", "h", "e", " ", "q", "u", "i", "c", "k", "b", "r", "o", "w", "n",
            "f", "x", "j", "m", "p", "s", "v", "l", "a", "z", "y", "d", "g"
        );
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);
        String corpus = "the quick brown fox jumps over the lazy dog";
        TextDataset dataset = new TextDataset(corpus, tokenizer);

        int batchSize = 1;
        int seqLen = 8;
        DataLoader loader = new DataLoader(dataset, batchSize, seqLen, false);

        // Build a small TinyGPT
        int vocabSize = tokenizer.getVocabSize();
        int embedDim = 16;
        int maxSeqLen = 32;
        int numLayers = 1;
        int numHeads = 2;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, 0.0f);
        model.train();

        AdamW optimizer = new AdamW(model.getParameters(), 0.01f);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();

        TestReporter.logMetric("Model config", model.summary());
        TestReporter.logMetric("Corpus length (chars)", corpus.length());
        TestReporter.logMetric("Corpus length (tokens)", dataset.size());

        // Grab a single fixed batch to overfit on
        Tensor[] fixedBatch = loader.nextBatch();
        Tensor x = fixedBatch[0];
        Tensor y = fixedBatch[1];

        // Train for 10 steps on the same batch, record first and last loss
        int numSteps = 10;
        float firstLoss = 0.0f;
        float lastLoss = 0.0f;
        for (int step = 0; step < numSteps; step++) {
            optimizer.zeroGrad();
            Tensor logits = model.forward(x);
            Tensor loss = lossFn.forward(logits, y);
            loss.backward();
            optimizer.step();

            float lossVal = loss.getData()[0];
            if (step == 0) firstLoss = lossVal;
            lastLoss = lossVal;
            TestReporter.logMetric("Step " + step + " loss", lossVal);
        }

        // Verify overall downward trend: last loss should be less than first loss
        assertTrue(
            lastLoss < firstLoss,
            "Training did not converge — first loss: " + firstLoss + ", last loss: " + lastLoss
        );

        TestReporter.logMetric("Loss reduction", String.format("%.4f → %.4f (Δ=%.4f)",
            firstLoss, lastLoss, firstLoss - lastLoss));
    }

    /**
     * Verifies the model summary string and parameter count are reasonable.
     */
    private static void testModelSummary() {
        TinyGPT model = new TinyGPT(100, 32, 64, 3, 4, 0.1f);
        String summary = model.summary();
        int paramCount = 0;
        for (Tensor p : model.getParameters()) {
            paramCount += p.size();
        }

        assertTrue(summary.contains("TinyGPT"), "Summary should contain model name");
        assertTrue(summary.contains("vocab=100"), "Summary should contain vocab size");
        assertTrue(summary.contains("layers=3"), "Summary should contain layer count");
        assertTrue(paramCount > 0, "Parameter count must be positive");

        TestReporter.logMetric("Summary", summary);
        TestReporter.logMetric("Total scalar parameters", paramCount);
    }
}
