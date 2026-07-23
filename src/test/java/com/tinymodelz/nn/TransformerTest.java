package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.Arrays;

/**
 * Unit tests for Transformer components (MultiHeadAttention, FeedForward,
 * TransformerBlock).
 */
public class TransformerTest {

    public static void runTests() {
        com.tinymodelz.TestReporter.runTest("Testing Multi-Head Attention forward and backward pass",
                () -> testMultiHeadAttention());
        com.tinymodelz.TestReporter.runTest("Testing Feed-Forward Network forward and backward pass",
                () -> testFeedForward());
        com.tinymodelz.TestReporter.runTest("Testing Transformer Block forward and backward pass",
                () -> testTransformerBlock());
    }

    private static void assertEquals(int[] expected, int[] actual, String message) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(
                    message + " - Expected: " + Arrays.toString(expected) + ", Got: " + Arrays.toString(actual));
        }
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void testMultiHeadAttention() {
        int B = 2; // batch size
        int T = 3; // seq length
        int C = 4; // embed dim
        int H = 2; // heads

        MultiHeadAttention mha = new MultiHeadAttention(C, H, 0.0f);
        mha.train();

        // Deterministic input
        float[] inputData = new float[B * T * C];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) Math.sin(i);
        }
        Tensor x = new Tensor(inputData, new int[] { B, T, C });
        x.setRequiresGrad(true);

        Tensor mask = MultiHeadAttention.createCausalMask(T);

        Tensor out = mha.forward(x, mask);
        assertEquals(new int[] { B, T, C }, out.getShape(), "MultiHeadAttention forward shape mismatch");

        // Backward
        Tensor loss = out.sum();
        loss.backward();

        // Verify gradients are propagated
        assertTrue(x.getGrad() != null, "Gradient of input x should not be null");
        assertEquals(x.getShape(), new int[] { B, T, C }, "Input gradient shape mismatch");

        // Verify parameter gradients are computed
        for (Tensor param : mha.getParameters()) {
            assertTrue(param.getGrad() != null, "Parameter gradient should not be null");
            // Check that parameter gradients are not all zeros
            boolean hasNonZero = false;
            for (float g : param.getGrad()) {
                if (g != 0.0f) {
                    hasNonZero = true;
                    break;
                }
            }
            assertTrue(hasNonZero, "Parameter gradient should have non-zero elements");
        }

        com.tinymodelz.TestReporter.logMetric("Batch size", B);
        com.tinymodelz.TestReporter.logMetric("Seq length", T);
        com.tinymodelz.TestReporter.logMetric("Heads", H);
        com.tinymodelz.TestReporter.logMetric("Head dim", C / H);
        com.tinymodelz.TestReporter.logMetric("Input shape", Arrays.toString(x.getShape()));
        com.tinymodelz.TestReporter.logMetric("Output shape", Arrays.toString(out.getShape()));
        com.tinymodelz.TestReporter.logMetric("Causal Mask shape", Arrays.toString(mask.getShape()));
    }

    private static void testFeedForward() {
        int B = 2;
        int T = 3;
        int C = 4;

        FeedForward mlp = new FeedForward(C, 0.0f);
        mlp.train();

        float[] inputData = new float[B * T * C];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) Math.cos(i);
        }
        Tensor x = new Tensor(inputData, new int[] { B, T, C });
        x.setRequiresGrad(true);

        Tensor out = mlp.forward(x);
        assertEquals(new int[] { B, T, C }, out.getShape(), "FeedForward forward shape mismatch");

        Tensor loss = out.sum();
        loss.backward();

        assertTrue(x.getGrad() != null, "Gradient of input x should not be null");
        for (Tensor param : mlp.getParameters()) {
            assertTrue(param.getGrad() != null, "Parameter gradient should not be null");
        }

        com.tinymodelz.TestReporter.logMetric("Input shape", Arrays.toString(x.getShape()));
        com.tinymodelz.TestReporter.logMetric("Hidden size", C * 4);
        com.tinymodelz.TestReporter.logMetric("Output shape", Arrays.toString(out.getShape()));
    }

    private static void testTransformerBlock() {
        int B = 2;
        int T = 3;
        int C = 4;
        int H = 2;

        TransformerBlock block = new TransformerBlock(C, H, 0.0f);
        block.train();

        float[] inputData = new float[B * T * C];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i] = (float) (i * 0.1);
        }
        Tensor x = new Tensor(inputData, new int[] { B, T, C });
        x.setRequiresGrad(true);

        Tensor mask = MultiHeadAttention.createCausalMask(T);

        Tensor out = block.forward(x, mask);
        assertEquals(new int[] { B, T, C }, out.getShape(), "TransformerBlock forward shape mismatch");

        Tensor loss = out.sum();
        loss.backward();

        assertTrue(x.getGrad() != null, "Gradient of input x should not be null");
        for (Tensor param : block.getParameters()) {
            assertTrue(param.getGrad() != null, "Parameter gradient should not be null");
        }

        com.tinymodelz.TestReporter.logMetric("Transformer Layer Config", "Embed=" + C + ", Heads=" + H);
        com.tinymodelz.TestReporter.logMetric("Layer Input shape", Arrays.toString(x.getShape()));
        com.tinymodelz.TestReporter.logMetric("Layer Output shape", Arrays.toString(out.getShape()));
        com.tinymodelz.TestReporter.logMetric("Parameters count", block.getParameters().size());
    }
}
