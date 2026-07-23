package com.tinymodelz.inference;

import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.Embedding;
import com.tinymodelz.nn.Linear;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import com.tinymodelz.TestReporter;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * <h3>Generator Test</h3>
 * 
 * <p>
 * Unit tests for text generation and sampling logic (Greedy, Temperature,
 * Top-K, Top-P).
 * </p>
 */
public class GeneratorTest {

    public static void runTests() {
        TestReporter.runTest("Greedy Decoding argmax sampling", () -> testGreedySampling());
        TestReporter.runTest("Temperature Scaling behavior", () -> testTemperatureScaling());
        TestReporter.runTest("Top-K Logit filtering bounds", () -> testTopKFiltering());
        TestReporter.runTest("Top-P Nucleus filtering bounds", () -> testTopPFiltering());
        TestReporter.runTest("Autoregressive text generation step", () -> testAutoregressiveGeneration());
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void testGreedySampling() {
        Generator generator = new Generator();
        float[] logits = { 0.1f, 0.5f, 2.5f, -1.0f, 0.2f };

        // With temperature = 0.0f, it should do greedy argmax
        int sampledId = generator.sample(logits, 0.0f, 0, 0.0f);
        assertEquals(2, sampledId, "Greedy sampling did not select argmax logit");
    }

    private static void testTemperatureScaling() {
        // Run multiple times and verify that with high temperature we get multiple
        // indexes,
        // but with extremely low temperature (e.g. 0.01) we get the argmax.
        Generator generator = new Generator(42);
        float[] logits = { 1.0f, 10.0f, 1.0f };

        // Low temperature (0.01) -> should always be argmax (1)
        for (int i = 0; i < 20; i++) {
            int sampledId = generator.sample(logits, 0.01f, 0, 0.0f);
            assertEquals(1, sampledId, "Low temperature sampling did not select argmax");
        }

        // High temperature (100.0) -> should be close to uniform, sampling index 0 and
        // 2 sometimes
        Set<Integer> sampledIndices = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            sampledIndices.add(generator.sample(logits, 100.0f, 0, 0.0f));
        }
        assertTrue(sampledIndices.size() > 1, "High temperature sampling did not explore other indices");
    }

    private static void testTopKFiltering() {
        Generator generator = new Generator(42);
        // Let's set topK = 2. The top 2 largest logits are at index 1 (10.0) and 3
        // (5.0).
        // Index 0 (1.0), 2 (2.0) and 4 (0.0) should NEVER be sampled.
        float[] logits = { 1.0f, 10.0f, 2.0f, 5.0f, 0.0f };

        Set<Integer> sampledIndices = new HashSet<>();
        for (int i = 0; i < 200; i++) {
            int id = generator.sample(logits, 1.0f, 2, 0.0f);
            sampledIndices.add(id);
            assertTrue(id == 1 || id == 3, "Sampled index " + id + " which is outside top-K=2");
        }

        TestReporter.logMetric("Top-K=2 sampled indices size", sampledIndices.size());
        assertTrue(sampledIndices.contains(1), "Top-K=2 did not sample top logit");
        assertTrue(sampledIndices.contains(3), "Top-K=2 did not sample second top logit");
    }

    private static void testTopPFiltering() {
        Generator generator = new Generator(42);
        // Let's set topP = 0.9.
        // Logits: index 1 is very large (10.0), next is index 3 (2.0), then index 0
        // (0.1)
        // Softmax of logits will be dominated by index 1 (approx 0.999), meaning index
        // 1 alone exceeds topP = 0.9.
        // Therefore, only index 1 should be active and all others filtered out.
        float[] logits = { 0.1f, 10.0f, 0.0f, 2.0f, -1.0f };

        for (int i = 0; i < 100; i++) {
            int id = generator.sample(logits, 1.0f, 0, 0.9f);
            assertEquals(1, id, "Sampled index " + id + " when top-P should restrict choice to index 1 only");
        }
    }

    private static void testAutoregressiveGeneration() {
        CharacterTokenizer tokenizer = new CharacterTokenizer(List.of("a", "b", "c"));
        int vocabSize = tokenizer.getVocabSize();
        int embedDim = 4;

        // Create a simple model: Embedding + Linear projection
        com.tinymodelz.nn.Module model = new com.tinymodelz.nn.Module() {
            private final Embedding embedding = new Embedding(vocabSize, embedDim);
            private final Linear lmHead = new Linear(embedDim, vocabSize);

            {
                for (Tensor p : embedding.getParameters())
                    registerParameter(p);
                for (Tensor p : lmHead.getParameters())
                    registerParameter(p);
            }

            @Override
            public Tensor forward(Tensor... inputs) {
                Tensor idx = inputs[0];
                Tensor emb = embedding.forward(idx); // [1, contextLen, embedDim]
                int[] shape = emb.getShape();
                Tensor emb2d = emb.reshape(shape[0] * shape[1], shape[2]);
                Tensor logits2d = lmHead.forward(emb2d);
                Tensor logits = logits2d.reshape(shape[0], shape[1], vocabSize);

                // Force logits of special tokens to be extremely negative so they are never
                // sampled
                float[] data = logits.getData();
                int offset = logits.offset();
                for (int i = 0; i < logits.size(); i++) {
                    int vocabIdx = i % vocabSize;
                    if (vocabIdx >= 3) {
                        data[offset + i] = -10000.0f;
                    }
                }
                return logits;
            }
        };

        Generator generator = new Generator(1337);
        // Generate 5 new tokens from prompt "a"
        String result = generator.generate(model, tokenizer, "a", 5, 1.0f, 0, 0.0f, 10, null);

        TestReporter.logMetric("Generator Prompt", "a");
        TestReporter.logMetric("Generator Output", result);

        assertEquals(6, result.length(), "Generated sequence length mismatch (prompt + 5 tokens)");
    }
}
