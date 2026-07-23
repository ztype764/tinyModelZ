package com.tinymodelz.train;

import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.Embedding;
import com.tinymodelz.nn.Linear;
import com.tinymodelz.nn.TransformerBlock;
import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import com.tinymodelz.TestReporter;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * <h3>Training Test</h3>
 * 
 * <p>Unit and integration tests for the training loop components (Dataset, DataLoader, CrossEntropyLoss, AdamW, Checkpoints).</p>
 */
public class TrainingTest {

    public static void runTests() {
        TestReporter.runTest("Dataset and DataLoader Batching", () -> testDatasetAndDataLoader());
        TestReporter.runTest("DataLoader Bounds and Capacity Validation", () -> testDataLoaderBoundsValidation());
        TestReporter.runTest("Cross Entropy Loss Forward & Backward Pass", () -> testCrossEntropyLoss());
        TestReporter.runTest("AdamW Optimizer weight updates", () -> testAdamWOptimizer());
        TestReporter.runTest("Model Checkpoint Save & Load", () -> {
            try {
                testModelCheckpoint();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        TestReporter.runTest("End-to-End TinyGPT Language Model training step", () -> testEndToEndTraining());
        TestReporter.runTest("LRScheduler Warmup & Cosine Decay schedule", () -> testLRScheduler());
        TestReporter.runTest("Gradient Clipping L2 Norm scaling", () -> testGradientClipping());
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void testDatasetAndDataLoader() {
        List<String> vocab = List.of("a", "b", "c", "d", "e", "f", "g");
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);
        TextDataset dataset = new TextDataset("abcdefg", tokenizer);
        
        DataLoader loader = new DataLoader(dataset, 2, 3, false);
        assertEquals(2, loader.getNumBatches(), "Number of batches calculation failed");

        Tensor[] batch = loader.nextBatch();
        Tensor x = batch[0];
        Tensor y = batch[1];

        assertEquals(2, x.getShape()[0], "Batch dimension mismatch");
        assertEquals(3, x.getShape()[1], "Seq len dimension mismatch");
        assertEquals(2, y.getShape()[0], "Target batch dimension mismatch");
        assertEquals(3, y.getShape()[1], "Target seq len dimension mismatch");

        // Verify target values are shifted by 1
        float[] xVal = x.getData();
        float[] yVal = y.getData();
        for (int i = 0; i < xVal.length; i++) {
            assertEquals(xVal[i] + 1.0f, yVal[i], "Target is not shifted version of input");
        }
    }

    private static void testDataLoaderBoundsValidation() {
        List<String> vocab = List.of("a", "b", "c", "d", "e");
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);
        TextDataset dataset = new TextDataset("abcde", tokenizer); // length = 5 tokens

        // Test 1: tokenIds.length < seqLen + 1 (5 < 5 + 1) -> must throw IllegalArgumentException
        boolean threw1 = false;
        try {
            new DataLoader(dataset, 1, 5, false);
        } catch (IllegalArgumentException e) {
            threw1 = true;
        }
        if (!threw1) {
            throw new AssertionError("DataLoader failed to throw IllegalArgumentException when token count < seqLen + 1");
        }

        // Test 2: numSamples < batchSize (samples = 5 - 2 = 3 < batchSize 5) -> must throw IllegalArgumentException
        boolean threw2 = false;
        try {
            new DataLoader(dataset, 5, 2, false);
        } catch (IllegalArgumentException e) {
            threw2 = true;
        }
        if (!threw2) {
            throw new AssertionError("DataLoader failed to throw IllegalArgumentException when available samples < batchSize");
        }

        // Test 3: Shuffled and non-shuffled memory-optimized modes with strided sampling
        DataLoader loaderNoShuffle = new DataLoader(dataset, 1, 2, 2, false); // seqLen=2, stride=2 -> 2 samples
        assertEquals(2, loaderNoShuffle.getTotalSamples(), "Strided total samples calculation failed");
        assertEquals(2, loaderNoShuffle.getNumBatches(), "Strided num batches calculation failed");

        DataLoader loaderShuffle = new DataLoader(dataset, 1, 2, 2, true);
        assertEquals(2, loaderShuffle.getTotalSamples(), "Shuffled strided total samples calculation failed");
    }

    private static void testCrossEntropyLoss() {
        float[] logitsData = {
            1.0f, 2.0f, 3.0f,
            0.5f, -0.5f, 1.5f
        };
        Tensor logits = new Tensor(logitsData, new int[]{2, 3});
        logits.setRequiresGrad(true);

        float[] targetsData = {2.0f, 0.0f};
        Tensor targets = new Tensor(targetsData, new int[]{2});

        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Tensor loss = lossFn.forward(logits, targets);

        float expectedLoss = 0.9076f;
        float actualLoss = loss.getData()[0];
        
        TestReporter.logMetric("Expected Loss", expectedLoss);
        TestReporter.logMetric("Actual Loss", actualLoss);
        
        if (Math.abs(expectedLoss - actualLoss) > 1e-3f) {
            throw new AssertionError("Cross entropy loss calculation mismatch. Expected: " + expectedLoss + ", got: " + actualLoss);
        }

        loss.backward();
        float[] grads = logits.getGrad();
        if (grads == null) {
            throw new AssertionError("Gradients were not computed for logits");
        }

        TestReporter.logMetric("Logits gradient sample 0_0", grads[0]);
        TestReporter.logMetric("Logits gradient sample 0_2", grads[2]);

        if (Math.abs(grads[0] - 0.045f) > 1e-2f || Math.abs(grads[2] - (-0.167f)) > 1e-2f) {
            throw new AssertionError("Gradient computation mismatch for CrossEntropyLoss");
        }
    }

    private static void testAdamWOptimizer() {
        Tensor p = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2});
        p.setRequiresGrad(true);
        p.accumulateGrad(new float[]{0.1f, -0.2f});

        AdamW optimizer = new AdamW(List.of(p), 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);
        optimizer.step();

        float expectedVal = 0.9f;
        float actualVal = p.getData()[0];
        
        TestReporter.logMetric("Weight after step expected", expectedVal);
        TestReporter.logMetric("Weight after step actual", actualVal);

        if (Math.abs(expectedVal - actualVal) > 1e-3f) {
            throw new AssertionError("AdamW optimizer weight update mismatch");
        }
    }

    private static void testModelCheckpoint() throws IOException {
        com.tinymodelz.nn.Module testModule = new com.tinymodelz.nn.Module() {
            private final Embedding embedding = new Embedding(10, 8);
            
            {
                for (Tensor p : embedding.getParameters()) registerParameter(p);
            }

            @Override
            public Tensor forward(Tensor... inputs) {
                return embedding.forward(inputs[0]);
            }
        };

        File tempDir = new File("target/temp_checkpoint_" + System.currentTimeMillis());
        try {
            AdamW optimizer = new AdamW(testModule.getParameters(), 1e-3f);
            // Simulate a training step to populate m and v states
            for (Tensor p : testModule.getParameters()) {
                p.accumulateGrad(new float[p.size()]);
            }
            optimizer.step(); // stepCount = 1

            Checkpoint.saveCheckpoint(testModule, optimizer, 2, 50, tempDir);

            float[] origData = testModule.getParameters().get(0).getData();
            float firstVal = origData[0];
            origData[0] = firstVal + 10.0f;

            AdamW restoredOptimizer = new AdamW(testModule.getParameters(), 1e-3f);
            Checkpoint.CheckpointState state = Checkpoint.loadCheckpoint(testModule, restoredOptimizer, tempDir);

            assertEquals(firstVal, origData[0], "Checkpoint reload did not restore weight value");
            assertEquals(2, state.epoch, "Checkpoint reload did not restore epoch");
            assertEquals(50, state.globalStep, "Checkpoint reload did not restore globalStep");
            assertEquals(1, restoredOptimizer.getStepCount(), "Checkpoint reload did not restore stepCount");
        } finally {
            if (tempDir.exists()) {
                File[] files = tempDir.listFiles();
                if (files != null) {
                    for (File f : files) f.delete();
                }
                tempDir.delete();
            }
        }
    }

    private static void testEndToEndTraining() {
        List<String> vocab = List.of("h", "e", "l", "o", " ", "t", "r", "a", "i", "n");
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);
        TextDataset dataset = new TextDataset("hello train hello train", tokenizer);
        DataLoader loader = new DataLoader(dataset, 1, 4, false);

        int vocabSize = tokenizer.getVocabSize();
        int embedDim = 8;
        int numHeads = 2;

        com.tinymodelz.nn.Module model = new com.tinymodelz.nn.Module() {
            private final Embedding embedding = new Embedding(vocabSize, embedDim);
            private final TransformerBlock transformer = new TransformerBlock(embedDim, numHeads, 0.0f);
            private final Linear lmHead = new Linear(embedDim, vocabSize);

            {
                for (Tensor p : embedding.getParameters()) registerParameter(p);
                for (Tensor p : transformer.getParameters()) registerParameter(p);
                for (Tensor p : lmHead.getParameters()) registerParameter(p);
            }

            @Override
            public Tensor forward(Tensor... inputs) {
                Tensor idx = inputs[0];
                Tensor emb = embedding.forward(idx);
                Tensor trans = transformer.forward(emb);
                
                int[] shape = trans.getShape();
                int B = shape[0];
                int T = shape[1];
                int C = shape[2];
                
                Tensor trans2d = trans.reshape(B * T, C);
                Tensor logits2d = lmHead.forward(trans2d);
                return logits2d.reshape(B, T, vocabSize);
            }
        };

        AdamW optimizer = new AdamW(model.getParameters(), 0.01f);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();

        Tensor[] batch = loader.nextBatch();
        Tensor x = batch[0];
        Tensor y = batch[1];

        // 1. First step
        optimizer.zeroGrad();
        Tensor logits1 = model.forward(x);
        Tensor loss1 = lossFn.forward(logits1, y);
        loss1.backward();
        optimizer.step();

        float firstLoss = loss1.getData()[0];

        // 2. Second step
        optimizer.zeroGrad();
        Tensor logits2 = model.forward(x);
        Tensor loss2 = lossFn.forward(logits2, y);
        loss2.backward();
        optimizer.step();

        float secondLoss = loss2.getData()[0];

        TestReporter.logMetric("First loss value", firstLoss);
        TestReporter.logMetric("Second loss value", secondLoss);

        if (secondLoss >= firstLoss) {
            throw new AssertionError("Training step failed: Loss did not decrease. First: " + firstLoss + ", Second: " + secondLoss);
        }
    }

    private static void testLRScheduler() {
        Tensor weight = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2});
        weight.setRequiresGrad(true);
        AdamW optimizer = new AdamW(List.of(weight), 1e-3f);
        LRScheduler scheduler = new LRScheduler(optimizer, 1e-3f, 1e-5f, 10, 100);

        // Step 1 to 10: Warmup phase (increasing LR)
        float lr1 = scheduler.step();
        float lr5 = 0;
        for (int i = 2; i <= 5; i++) lr5 = scheduler.step();
        float lr10 = 0;
        for (int i = 6; i <= 10; i++) lr10 = scheduler.step();

        TestReporter.logMetric("Warmup Step 1 LR", lr1);
        TestReporter.logMetric("Warmup Step 5 LR", lr5);
        TestReporter.logMetric("Warmup Step 10 LR (Peak)", lr10);

        if (lr1 >= lr5 || lr5 >= lr10) {
            throw new AssertionError("LRScheduler warmup failed: LR did not monotonically increase.");
        }

        // Step 11 to 100: Cosine decay phase (decreasing LR)
        float lr50 = 0;
        for (int i = 11; i <= 50; i++) lr50 = scheduler.step();
        float lr100 = 0;
        for (int i = 51; i <= 100; i++) lr100 = scheduler.step();

        TestReporter.logMetric("Cosine Step 50 LR", lr50);
        TestReporter.logMetric("Cosine Step 100 LR (Min)", lr100);

        if (lr10 <= lr50 || lr50 <= lr100) {
            throw new AssertionError("LRScheduler cosine decay failed: LR did not monotonically decrease.");
        }
    }

    private static void testGradientClipping() {
        List<String> vocab = List.of("a", "b", "c", "d");
        CharacterTokenizer tokenizer = new CharacterTokenizer(vocab);
        com.tinymodelz.nn.TinyGPT model = new com.tinymodelz.nn.TinyGPT(vocab.size(), 8, 2, 2, 8, 16);
        AdamW optimizer = new AdamW(model.getParameters(), 1e-3f);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Trainer trainer = new Trainer(model, tokenizer, optimizer, lossFn);

        // Manually assign large gradients
        for (Tensor p : model.getParameters()) {
            if (p.requiresGrad()) {
                float[] grad = new float[p.size()];
                java.util.Arrays.fill(grad, 100.0f);
                p.accumulateGrad(grad);
            }
        }

        float normBefore = trainer.clipGradients(1.0f);
        TestReporter.logMetric("Gradient Norm Before Clipping", normBefore);

        // Compute norm after clipping
        double sumSq = 0;
        for (Tensor p : model.getParameters()) {
            if (p.requiresGrad() && p.getGrad() != null) {
                for (float g : p.getGrad()) {
                    sumSq += g * g;
                }
            }
        }
        float normAfter = (float) Math.sqrt(sumSq);
        TestReporter.logMetric("Gradient Norm After Clipping", normAfter);

        if (normAfter > 1.05f) {
            throw new AssertionError("Gradient clipping failed: norm " + normAfter + " > 1.0");
        }
    }
}
