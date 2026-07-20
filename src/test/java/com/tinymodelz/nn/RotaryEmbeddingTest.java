package com.tinymodelz.nn;

import com.tinymodelz.TestReporter;
import com.tinymodelz.math.Tensor;

/**
 * <h3>RotaryEmbeddingTest</h3>
 *
 * <p>Unit tests for Rotary Position Embeddings (RoPE).</p>
 */
public class RotaryEmbeddingTest {

    public static void runTests() {
        TestReporter.runTest("Rotary Position Embedding (RoPE) rotation mathematical correctness", () -> {
            int dim = 4;
            RotaryEmbedding rope = new RotaryEmbedding(dim);

            float[] data = new float[]{1.0f, 0.0f, 0.5f, 0.5f};
            Tensor x = new Tensor(data, new int[]{1, 1, dim});

            Tensor rotated = rope.apply(x);
            float[] outData = rotated.getData();

            // Position 0 -> angle = 0, cos(0)=1, sin(0)=0 -> unchanged output
            for (int i = 0; i < dim; i++) {
                if (Math.abs(data[i] - outData[i]) > 1e-5f) {
                    throw new AssertionError("At pos 0, RoPE output should equal input. Expected " + data[i] + " got " + outData[i]);
                }
            }

            // Test pos > 0
            float[] seqData = new float[]{
                1.0f, 0.0f, 0.0f, 0.0f, // pos 0
                1.0f, 0.0f, 0.0f, 0.0f  // pos 1
            };
            Tensor seqX = new Tensor(seqData, new int[]{1, 2, dim});
            Tensor seqRotated = rope.apply(seqX);
            float[] seqOut = seqRotated.getData();

            // Pos 1 first pair rotated by theta0 = 10000^0 = 1.0 -> angle = 1.0 rad
            // x0' = 1.0*cos(1) - 0.0*sin(1) = cos(1) ~ 0.5403f
            float expectedX0 = (float) Math.cos(1.0);
            if (Math.abs(expectedX0 - seqOut[4]) > 1e-3f) {
                throw new AssertionError("Pos 1 RoPE rotation mismatch. Expected " + expectedX0 + " got " + seqOut[4]);
            }

            TestReporter.logMetric("RoPE Output Shape", java.util.Arrays.toString(rotated.getShape()));
        });
    }
}
