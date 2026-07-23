package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.Arrays;

/**
 * Unit tests for Neural Network layers and components (Linear, Embedding, GeLU,
 * LayerNorm, Dropout).
 */
public class NeuralNetworkTest {

    public static void runTests() {
        com.tinymodelz.TestReporter.runTest("Testing Linear Layer Forward & Backward", () -> testLinearLayer());
        com.tinymodelz.TestReporter.runTest("Testing Embedding Layer Forward & Backward", () -> testEmbeddingLayer());
        com.tinymodelz.TestReporter.runTest("Testing GeLU Activation Forward & Backward", () -> testGeLUActivation());
        com.tinymodelz.TestReporter.runTest("Testing LayerNorm Layer Forward & Backward", () -> testLayerNorm());
        com.tinymodelz.TestReporter.runTest("Testing Dropout Layer Behavior", () -> testDropout());
    }

    private static void assertEquals(float expected, float actual, float epsilon, String message) {
        if (Math.abs(expected - actual) > epsilon) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
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

    private static void testLinearLayer() {
        // Create Linear(3, 2)
        Linear linear = new Linear(3, 2, true);

        // Manually assign weights and biases for deterministic validation
        float[] wData = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        System.arraycopy(wData, 0, linear.getWeight().getData(), 0, 6);
        float[] bData = { 0.5f, -0.5f };
        System.arraycopy(bData, 0, linear.getBias().getData(), 0, 2);

        // Input shape [1, 3]
        Tensor input = new Tensor(new float[] { 1.0f, 0.0f, -1.0f }, new int[] { 1, 3 });
        input.setRequiresGrad(true);

        // Forward: Output = X * W^T + bias
        Tensor out = linear.forward(input);

        assertEquals(new int[] { 1, 2 }, out.getShape(), "Linear shape mismatch");
        assertEquals(-1.5f, out.getValByFlatIndex(0), 1e-5f, "Linear out[0] incorrect");
        assertEquals(-2.5f, out.getValByFlatIndex(1), 1e-5f, "Linear out[1] incorrect");

        // Backward
        Tensor loss = out.sum();
        loss.backward();

        // Check gradients
        assertEquals(5.0f, input.getGrad()[0], 1e-5f, "Linear input grad[0] incorrect");
        assertEquals(7.0f, input.getGrad()[1], 1e-5f, "Linear input grad[1] incorrect");
        assertEquals(9.0f, input.getGrad()[2], 1e-5f, "Linear input grad[2] incorrect");

        assertEquals(1.0f, linear.getWeight().getGrad()[0], 1e-5f, "Linear weight grad[0] incorrect");
        assertEquals(0.0f, linear.getWeight().getGrad()[1], 1e-5f, "Linear weight grad[1] incorrect");
        assertEquals(-1.0f, linear.getWeight().getGrad()[2], 1e-5f, "Linear weight grad[2] incorrect");

        assertEquals(1.0f, linear.getBias().getGrad()[0], 1e-5f, "Linear bias grad[0] incorrect");
        assertEquals(1.0f, linear.getBias().getGrad()[1], 1e-5f, "Linear bias grad[1] incorrect");

        com.tinymodelz.TestReporter.logMetric("Weight shape", "2x3");
        com.tinymodelz.TestReporter.logMetric("Input shape", Arrays.toString(input.getShape()));
        com.tinymodelz.TestReporter.logMetric("Output shape", Arrays.toString(out.getShape()));
        com.tinymodelz.TestReporter.logMetric("Output values", Arrays.toString(out.getData()));
        com.tinymodelz.TestReporter.logMetric("Input grad", Arrays.toString(input.getGrad()));
    }

    private static void testEmbeddingLayer() {
        // Embedding vocab=3, dim=4
        Embedding emb = new Embedding(3, 4);
        float[] wData = {
                0.1f, 0.2f, 0.3f, 0.4f,
                0.5f, 0.6f, 0.7f, 0.8f,
                0.9f, 1.0f, 1.1f, 1.2f
        };
        System.arraycopy(wData, 0, emb.getWeight().getData(), 0, 12);

        // Indices shape [2]
        Tensor indices = new Tensor(new float[] { 2.0f, 0.0f }, new int[] { 2 });

        Tensor out = emb.forward(indices);
        assertEquals(new int[] { 2, 4 }, out.getShape(), "Embedding out shape mismatch");

        assertEquals(0.9f, out.getValByFlatIndex(0), 1e-5f, "Embedding out row 2 incorrect");
        assertEquals(0.1f, out.getValByFlatIndex(4), 1e-5f, "Embedding out row 0 incorrect");

        // Backward
        Tensor loss = out.sum();
        loss.backward();

        // Check gradients
        assertEquals(1.0f, emb.getWeight().getGrad()[0], 1e-5f, "Embedding grad row 0 incorrect");
        assertEquals(0.0f, emb.getWeight().getGrad()[4], 1e-5f, "Embedding grad row 1 incorrect");
        assertEquals(1.0f, emb.getWeight().getGrad()[8], 1e-5f, "Embedding grad row 2 incorrect");

        com.tinymodelz.TestReporter.logMetric("Vocab size", "3");
        com.tinymodelz.TestReporter.logMetric("Embedding dim", "4");
        com.tinymodelz.TestReporter.logMetric("Indices", Arrays.toString(indices.getData()));
        com.tinymodelz.TestReporter.logMetric("Output shape", Arrays.toString(out.getShape()));
    }

    private static void testGeLUActivation() {
        Tensor x = new Tensor(new float[] { -1.0f, 0.0f, 1.0f }, new int[] { 3 });
        x.setRequiresGrad(true);

        Tensor y = x.gelu();

        assertEquals(-0.1586f, y.getValByFlatIndex(0), 1e-3f, "GeLU forward -1.0 incorrect");
        assertEquals(0.0f, y.getValByFlatIndex(1), 1e-5f, "GeLU forward 0.0 incorrect");
        assertEquals(0.8413f, y.getValByFlatIndex(2), 1e-3f, "GeLU forward 1.0 incorrect");

        Tensor loss = y.sum();
        loss.backward();

        assertEquals(-0.077f, x.getGrad()[0], 1e-2f, "GeLU grad -1.0 incorrect");
        assertEquals(0.5f, x.getGrad()[1], 1e-3f, "GeLU grad 0.0 incorrect");
        assertEquals(1.077f, x.getGrad()[2], 1e-2f, "GeLU grad 1.0 incorrect");

        com.tinymodelz.TestReporter.logMetric("Input values", Arrays.toString(x.getData()));
        com.tinymodelz.TestReporter.logMetric("GeLU output", Arrays.toString(y.getData()));
        com.tinymodelz.TestReporter.logMetric("Input grad", Arrays.toString(x.getGrad()));
    }

    private static void testLayerNorm() {
        // Normalized dim = 3
        LayerNorm ln = new LayerNorm(3, 1e-5f);
        Tensor input = new Tensor(new float[] {
                1.0f, 2.0f, 3.0f,
                10.0f, 20.0f, 30.0f
        }, new int[] { 2, 3 });
        input.setRequiresGrad(true);

        Tensor out = ln.forward(input);

        assertEquals(-1.2247f, out.getValByFlatIndex(0), 1e-3f, "LayerNorm normalized value incorrect");
        assertEquals(0.0f, out.getValByFlatIndex(1), 1e-3f, "LayerNorm normalized value incorrect");
        assertEquals(1.2247f, out.getValByFlatIndex(2), 1e-3f, "LayerNorm normalized value incorrect");

        Tensor loss = out.sum();
        loss.backward();

        assertEquals(0.0f, input.getGrad()[0], 1e-5f, "LayerNorm input grad should be zero when dy/dout is constant");
        assertEquals(0.0f, input.getGrad()[3], 1e-5f, "LayerNorm input grad should be zero when dy/dout is constant");

        com.tinymodelz.TestReporter.logMetric("Norm shape", Arrays.toString(input.getShape()));
        com.tinymodelz.TestReporter.logMetric("Normalized output", Arrays.toString(out.getData()));
    }

    private static void testDropout() {
        Tensor input = Tensor.ones(100);
        input.setRequiresGrad(true);

        Dropout dropout = new Dropout(0.2f, 42);

        // 1. Train mode
        dropout.train();
        Tensor outTrain = dropout.forward(input);

        int zeroCount = 0;
        for (int i = 0; i < 100; i++) {
            if (outTrain.getValByFlatIndex(i) == 0.0f) {
                zeroCount++;
            }
        }
        assertTrue(zeroCount > 0 && zeroCount < 100, "Dropout should drop some units in train mode");

        // Backward
        Tensor loss = outTrain.sum();
        loss.backward();

        for (int i = 0; i < 100; i++) {
            float val = outTrain.getValByFlatIndex(i);
            float grad = input.getGrad()[i];
            if (val == 0.0f) {
                assertEquals(0.0f, grad, 1e-5f, "Dropout grad should be 0 for dropped unit");
            } else {
                assertEquals(1.25f, grad, 1e-5f, "Dropout grad should be scaled for active unit");
            }
        }

        // 2. Eval mode
        input.zeroGrad();
        dropout.eval();
        Tensor outEval = dropout.forward(input);
        for (int i = 0; i < 100; i++) {
            assertEquals(1.0f, outEval.getValByFlatIndex(i), 1e-5f, "Dropout in eval mode must be identity");
        }

        com.tinymodelz.TestReporter.logMetric("Dropout rate", "0.2");
        com.tinymodelz.TestReporter.logMetric("Dropped units (Train)", zeroCount);
        com.tinymodelz.TestReporter.logMetric("Active units (Train)", 100 - zeroCount);
        com.tinymodelz.TestReporter.logMetric("Eval output check", outEval.getValByFlatIndex(0));
    }
}
