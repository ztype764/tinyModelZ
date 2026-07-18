package com.tinymodelz.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Unit tests for the Math Engine (Vector, Matrix, RandomInitializer, MathIO).
 */
public class MathEngineTest {

    private static final Logger logger = LoggerFactory.getLogger(MathEngineTest.class);

    public static void runTests() {
        com.tinymodelz.TestReporter.runTest("Testing Vector operations", () -> testVectorOperations());
        com.tinymodelz.TestReporter.runTest("Testing Matrix operations", () -> testMatrixOperations());
        com.tinymodelz.TestReporter.runTest("Testing RandomInitializer parameter fill distributions", () -> testRandomInitializers());
        com.tinymodelz.TestReporter.runTest("Testing MathIO serialization and deserialization", () -> testMathIO());
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

    private static void testVectorOperations() {
        logger.info("Testing Vector operations...");
        
        Vector v1 = new Vector(new float[]{1.0f, 2.0f, 3.0f});
        Vector v2 = new Vector(new float[]{4.0f, 5.0f, 6.0f});

        // Addition
        Vector sum = v1.add(v2);
        assertEquals(new Vector(new float[]{5.0f, 7.0f, 9.0f}), sum, "Vector addition failed");

        // Subtraction
        Vector diff = v2.subtract(v1);
        assertEquals(new Vector(new float[]{3.0f, 3.0f, 3.0f}), diff, "Vector subtraction failed");

        // Dot product
        float dot = v1.dot(v2);
        assertEquals(32.0f, dot, 1e-5f, "Vector dot product failed");

        // Scale
        Vector scaled = v1.scale(2.0f);
        assertEquals(new Vector(new float[]{2.0f, 4.0f, 6.0f}), scaled, "Vector scaling failed");

        // Norm
        float normVal = new Vector(new float[]{3.0f, 4.0f}).norm();
        assertEquals(5.0f, normVal, 1e-5f, "Vector norm calculation failed");

        // Softmax
        Vector vSoftmax = new Vector(new float[]{1.0f, 1.0f, 1.0f}).softmax();
        float expectedVal = 1.0f / 3.0f;
        assertEquals(expectedVal, vSoftmax.get(0), 1e-5f, "Softmax calculation failed");
        assertEquals(expectedVal, vSoftmax.get(1), 1e-5f, "Softmax calculation failed");
        assertEquals(expectedVal, vSoftmax.get(2), 1e-5f, "Softmax calculation failed");
    }

    private static void testMatrixOperations() {
        logger.info("Testing Matrix operations...");

        float[][] dataA = {
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        };
        float[][] dataB = {
            {5.0f, 6.0f},
            {7.0f, 8.0f}
        };

        Matrix mA = new Matrix(dataA);
        Matrix mB = new Matrix(dataB);

        // Addition
        Matrix sum = mA.add(mB);
        Matrix expectedSum = new Matrix(new float[][]{
            {6.0f, 8.0f},
            {10.0f, 12.0f}
        });
        assertEquals(expectedSum, sum, "Matrix addition failed");

        // Multiplication
        Matrix product = mA.multiply(mB);
        Matrix expectedProduct = new Matrix(new float[][]{
            {19.0f, 22.0f},
            {43.0f, 50.0f}
        });
        assertEquals(expectedProduct, product, "Matrix multiplication failed");

        // Transpose
        Matrix trans = mA.transpose();
        Matrix expectedTranspose = new Matrix(new float[][]{
            {1.0f, 3.0f},
            {2.0f, 4.0f}
        });
        assertEquals(expectedTranspose, trans, "Matrix transpose failed");

        // Matrix-Vector Product
        Vector vec = new Vector(new float[]{1.0f, 2.0f});
        Vector matVec = mA.multiply(vec);
        assertEquals(new Vector(new float[]{5.0f, 11.0f}), matVec, "Matrix-Vector multiplication failed");

        // Row-wise Bias Addition
        Vector bias = new Vector(new float[]{10.0f, 20.0f});
        Matrix biasedMatrix = mA.addBiasRowwise(bias);
        Matrix expectedBiased = new Matrix(new float[][]{
            {11.0f, 22.0f},
            {13.0f, 24.0f}
        });
        assertEquals(expectedBiased, biasedMatrix, "Matrix row-wise bias addition failed");
    }

    private static void testRandomInitializers() {
        logger.info("Testing RandomInitializer parameter fill distributions...");
        
        RandomInitializer ri = new RandomInitializer(42); // Seeded for determinism

        // Vector Uniform Fill
        Vector vec = new Vector(1000);
        ri.fillUniform(vec, -1.0f, 1.0f);
        for (int i = 0; i < vec.size(); i++) {
            float val = vec.get(i);
            if (val < -1.0f || val > 1.0f) {
                throw new AssertionError("Uniform vector initialization fell outside expected bounds");
            }
        }

        // Matrix Normal Fill Check (statistics check)
        Matrix mat = new Matrix(100, 100);
        ri.fillNormal(mat, 0.0f, 1.0f);
        
        float sum = 0.0f;
        float[][] matData = mat.getDataCopy();
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                sum += matData[i][j];
            }
        }
        float mean = sum / (mat.rows() * mat.cols());
        assertEquals(0.0f, mean, 0.05f, "Gaussian matrix initialization mean deviates from 0.0");
    }

    private static void testMathIO() {
        logger.info("Testing MathIO serialization and deserialization...");

        Vector origVec = new Vector(new float[]{1.1f, 2.2f, 3.3f});
        Matrix origMat = new Matrix(new float[][]{
            {10.5f, 20.6f},
            {30.7f, 40.8f}
        });
        Tensor origTensor = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, new int[]{2, 3});

        try {
            File tempVecFile = File.createTempFile("test_vec_", ".tma");
            File tempMatFile = File.createTempFile("test_mat_", ".tma");
            File tempTensorFile = File.createTempFile("test_tensor_", ".tma");
            tempVecFile.deleteOnExit();
            tempMatFile.deleteOnExit();
            tempTensorFile.deleteOnExit();

            // Save
            MathIO.saveVector(origVec, tempVecFile);
            MathIO.saveMatrix(origMat, tempMatFile);
            MathIO.saveTensor(origTensor, tempTensorFile);

            // Load
            Vector loadedVec = MathIO.loadVector(tempVecFile);
            Matrix loadedMat = MathIO.loadMatrix(tempMatFile);
            Tensor loadedTensor = MathIO.loadTensor(tempTensorFile);

            // Verify Vector & Matrix
            assertEquals(origVec, loadedVec, "Loaded vector does not match saved vector");
            assertEquals(origMat, loadedMat, "Loaded matrix does not match saved matrix");

            // Verify Tensor shape
            int[] origShape = origTensor.getShape();
            int[] loadedShape = loadedTensor.getShape();
            if (origShape.length != loadedShape.length) {
                throw new AssertionError("Loaded tensor rank mismatch");
            }
            for (int i = 0; i < origShape.length; i++) {
                if (origShape[i] != loadedShape[i]) {
                    throw new AssertionError("Loaded tensor shape mismatch");
                }
            }
            // Verify Tensor data
            float[] origData = origTensor.getData();
            float[] loadedData = loadedTensor.getData();
            for (int i = 0; i < origData.length; i++) {
                if (Float.compare(origData[i], loadedData[i]) != 0) {
                    throw new AssertionError("Loaded tensor value mismatch at index " + i);
                }
            }

        } catch (IOException e) {
            throw new AssertionError("MathIO save/load triggered IOException", e);
        }
    }
}
