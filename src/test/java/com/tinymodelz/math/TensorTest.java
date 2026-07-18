package com.tinymodelz.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Unit tests for the {@link Tensor} operations, broadcasting, layout views, and autograd backprop.
 */
public class TensorTest {

    private static final Logger logger = LoggerFactory.getLogger(TensorTest.class);

    public static void runTests() {
        com.tinymodelz.TestReporter.runTest("Testing Tensor instantiation and contiguity", () -> testInstantiationAndContiguity());
        com.tinymodelz.TestReporter.runTest("Testing Tensor shape transformations (reshape, transpose, slice)", () -> testShapeTransformations());
        com.tinymodelz.TestReporter.runTest("Testing Tensor broadcasting operations", () -> testBroadcastingOperations());
        com.tinymodelz.TestReporter.runTest("Testing Tensor matrix multiplication", () -> testMatrixMultiplication());
        com.tinymodelz.TestReporter.runTest("Testing Tensor Autograd engine", () -> testAutograd());
    }

    private static void assertEquals(float expected, float actual, float epsilon, String message) {
        if (Math.abs(expected - actual) > epsilon) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " - Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void assertEquals(int[] expected, int[] actual, String message) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(message + " - Expected: " + Arrays.toString(expected) + ", Got: " + Arrays.toString(actual));
        }
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void testInstantiationAndContiguity() {
        Tensor t1 = Tensor.zeros(2, 3);
        assertEquals(6, t1.size(), "Size calculation failed");
        assertTrue(t1.isContiguous(), "Zero tensor should be contiguous");
        
        float[] rawData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        Tensor t2 = new Tensor(rawData, new int[]{2, 3});
        assertEquals(1.0f, t2.getValByFlatIndex(0), 1e-5f, "Value access failed");
        assertEquals(6.0f, t2.getValByFlatIndex(5), 1e-5f, "Value access failed");

        com.tinymodelz.TestReporter.logMetric("t1 shape", Arrays.toString(t1.getShape()));
        com.tinymodelz.TestReporter.logMetric("t2 shape", Arrays.toString(t2.getShape()));
        com.tinymodelz.TestReporter.logMetric("t2 contiguous", t2.isContiguous());
    }

    private static void testShapeTransformations() {
        float[] rawData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        Tensor t = new Tensor(rawData, new int[]{2, 3});

        // Reshape
        Tensor reshaped = t.reshape(3, 2);
        assertEquals(new int[]{3, 2}, reshaped.getShape(), "Reshape failed to update shape");
        assertTrue(reshaped.isContiguous(), "Reshaped tensor should be contiguous");
        assertEquals(3.0f, reshaped.getValByFlatIndex(2), 1e-5f, "Reshaped values misplaced");

        // Transpose
        Tensor transposed = t.transpose();
        assertEquals(new int[]{3, 2}, transposed.getShape(), "Transpose failed to swap shapes");
        assertTrue(!transposed.isContiguous(), "Transposed tensor should NOT be contiguous in memory layout");
        // Transposed elements check (original col 1, row 0 becomes row 1, col 0)
        assertEquals(4.0f, transposed.getValByFlatIndex(1), 1e-5f, "Transposed indexing mismatch"); // coords [0, 1] in transposed -> physical offset 3
        assertEquals(2.0f, transposed.getValByFlatIndex(2), 1e-5f, "Transposed indexing mismatch"); // coords [1, 0] in transposed -> physical offset 1

        // Slice
        Tensor sliced = t.slice(1, 1, 3); // Slice columns index 1 to 2
        assertEquals(new int[]{2, 2}, sliced.getShape(), "Slice failed to update shape");
        // Row 0: [2.0, 3.0], Row 1: [5.0, 6.0]
        assertEquals(2.0f, sliced.getValByFlatIndex(0), 1e-5f, "Sliced index 0 mismatch");
        assertEquals(3.0f, sliced.getValByFlatIndex(1), 1e-5f, "Sliced index 1 mismatch");
        assertEquals(5.0f, sliced.getValByFlatIndex(2), 1e-5f, "Sliced index 2 mismatch");
        assertEquals(6.0f, sliced.getValByFlatIndex(3), 1e-5f, "Sliced index 3 mismatch");

        com.tinymodelz.TestReporter.logMetric("Original shape", Arrays.toString(t.getShape()));
        com.tinymodelz.TestReporter.logMetric("Reshaped shape", Arrays.toString(reshaped.getShape()));
        com.tinymodelz.TestReporter.logMetric("Transposed shape", Arrays.toString(transposed.getShape()));
        com.tinymodelz.TestReporter.logMetric("Sliced shape", Arrays.toString(sliced.getShape()));
    }

    private static void testBroadcastingOperations() {
        // [3, 1]
        Tensor tA = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{3, 1});
        // [1, 2]
        Tensor tB = new Tensor(new float[]{10.0f, 20.0f}, new int[]{1, 2});

        // Add
        Tensor tC = tA.add(tB); // broadcast shape [3, 2]
        assertEquals(new int[]{3, 2}, tC.getShape(), "Broadcasting add shape failed");
        
        // Expected elements:
        // [ 11, 21 ]
        // [ 12, 22 ]
        // [ 13, 23 ]
        assertEquals(11.0f, tC.getValByFlatIndex(0), 1e-5f, "Broadcasting add calculation failed");
        assertEquals(21.0f, tC.getValByFlatIndex(1), 1e-5f, "Broadcasting add calculation failed");
        assertEquals(12.0f, tC.getValByFlatIndex(2), 1e-5f, "Broadcasting add calculation failed");
        assertEquals(23.0f, tC.getValByFlatIndex(5), 1e-5f, "Broadcasting add calculation failed");

        com.tinymodelz.TestReporter.logMetric("tA shape", Arrays.toString(tA.getShape()));
        com.tinymodelz.TestReporter.logMetric("tB shape", Arrays.toString(tB.getShape()));
        com.tinymodelz.TestReporter.logMetric("Broadcasted sum shape", Arrays.toString(tC.getShape()));
    }

    private static void testMatrixMultiplication() {
        Tensor tA = new Tensor(new float[]{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        }, new int[]{2, 3});

        Tensor tB = new Tensor(new float[]{
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f
        }, new int[]{3, 2});

        Tensor tC = tA.matmul(tB); // [2, 2]
        assertEquals(new int[]{2, 2}, tC.getShape(), "Matmul shape mismatch");

        // Expected output:
        // Row 0: [ 1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12 ] = [ 58, 64 ]
        // Row 1: [ 4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12 ] = [ 139, 154 ]
        assertEquals(58.0f, tC.getValByFlatIndex(0), 1e-5f, "Matmul cell [0,0] incorrect");
        assertEquals(64.0f, tC.getValByFlatIndex(1), 1e-5f, "Matmul cell [0,1] incorrect");
        assertEquals(139.0f, tC.getValByFlatIndex(2), 1e-5f, "Matmul cell [1,0] incorrect");
        assertEquals(154.0f, tC.getValByFlatIndex(3), 1e-5f, "Matmul cell [1,1] incorrect");

        com.tinymodelz.TestReporter.logMetric("tA Matmul shape", Arrays.toString(tA.getShape()));
        com.tinymodelz.TestReporter.logMetric("tB Matmul shape", Arrays.toString(tB.getShape()));
        com.tinymodelz.TestReporter.logMetric("Output Matmul shape", Arrays.toString(tC.getShape()));
        com.tinymodelz.TestReporter.logMetric("Output values", Arrays.toString(tC.getData()));
    }

    private static void testAutograd() {
        // Scalar computations: y = (x1 * x2) + x3
        Tensor x1 = new Tensor(new float[]{2.0f}, new int[]{1});
        Tensor x2 = new Tensor(new float[]{3.0f}, new int[]{1});
        Tensor x3 = new Tensor(new float[]{4.0f}, new int[]{1});

        x1.setRequiresGrad(true);
        x2.setRequiresGrad(true);
        x3.setRequiresGrad(true);

        Tensor y = x1.multiply(x2).add(x3);
        assertEquals(10.0f, y.getValByFlatIndex(0), 1e-5f, "Autograd forward output mismatch");

        y.backward();

        // Derivatives: dy/dx1 = x2 = 3, dy/dx2 = x1 = 2, dy/dx3 = 1
        assertEquals(3.0f, x1.getGrad()[0], 1e-5f, "dx1 gradient incorrect");
        assertEquals(2.0f, x2.getGrad()[0], 1e-5f, "dx2 gradient incorrect");
        assertEquals(1.0f, x3.getGrad()[0], 1e-5f, "dx3 gradient incorrect");

        // Activations & Broadcast Autograd:
        // z = relu(A * B + C)
        // A shape [2, 2], B shape [2, 1] (requires broadcasting along columns of output), C shape [2, 1]
        Tensor A = new Tensor(new float[]{
            1.0f, -2.0f,
            3.0f, 4.0f
        }, new int[]{2, 2});
        Tensor B = new Tensor(new float[]{
            5.0f,
            6.0f
        }, new int[]{2, 1});
        Tensor C = new Tensor(new float[]{
            10.0f,
            -20.0f
        }, new int[]{2, 1});

        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);

        Tensor prod = A.matmul(B); // [2, 1] -> [1*5 - 2*6, 3*5 + 4*6] = [-7, 39]
        Tensor added = prod.add(C); // [2, 1] -> [-7 + 10, 39 - 20] = [3, 19]
        Tensor z = added.relu(); // [2, 1] -> [3, 19]
        
        // Sum to get scalar output for backward
        Tensor loss = z.sum();
        loss.backward();
        
        assertEquals(1.0f, C.getGrad()[0], 1e-5f, "dC[0] incorrect");
        assertEquals(1.0f, C.getGrad()[1], 1e-5f, "dC[1] incorrect");

        assertEquals(5.0f, A.getGrad()[0], 1e-5f, "dA[0,0] incorrect");
        assertEquals(6.0f, A.getGrad()[1], 1e-5f, "dA[0,1] incorrect");
        assertEquals(5.0f, A.getGrad()[2], 1e-5f, "dA[1,0] incorrect");
        assertEquals(6.0f, A.getGrad()[3], 1e-5f, "dA[1,1] incorrect");

        assertEquals(4.0f, B.getGrad()[0], 1e-5f, "dB[0] incorrect");
        assertEquals(2.0f, B.getGrad()[1], 1e-5f, "dB[1] incorrect");

        com.tinymodelz.TestReporter.logMetric("Scalar Forward Output", y.getValByFlatIndex(0));
        com.tinymodelz.TestReporter.logMetric("dx1, dx2, dx3", Arrays.toString(new float[]{x1.getGrad()[0], x2.getGrad()[0], x3.getGrad()[0]}));
        com.tinymodelz.TestReporter.logMetric("dA grad shape", Arrays.toString(A.getGrad()));
        com.tinymodelz.TestReporter.logMetric("dB grad shape", Arrays.toString(B.getGrad()));
    }
}
