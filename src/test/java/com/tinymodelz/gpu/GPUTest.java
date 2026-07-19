package com.tinymodelz.gpu;

import com.tinymodelz.TestReporter;
import com.tinymodelz.math.Device;
import com.tinymodelz.math.DeviceManager;
import com.tinymodelz.math.Matrix;
import com.tinymodelz.math.Tensor;

import java.util.Arrays;

/**
 * <h3>GPUTest</h3>
 *
 * <p>Unit test suite validating OpenCL GPU hardware compute engine initialization,
 * GPU matrix multiplication accuracy, and dynamic device switching.</p>
 */
public class GPUTest {

    public static void runTests() {
        TestReporter.runTest("GPU hardware compute engine probing and device info", () -> testGPUInitialization());
        TestReporter.runTest("GPU matrix multiplication numerical precision vs CPU", () -> testGPUMatMulPrecision());
        TestReporter.runTest("Batched tensor matrix multiplication on GPU", () -> testGPUTensorMatmul());
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void assertEquals(float expected, float actual, float eps, String message) {
        if (Math.abs(expected - actual) > eps) {
            throw new AssertionError(message + " — Expected: " + expected + ", Got: " + actual);
        }
    }

    private static void testGPUInitialization() {
        boolean available = GPUMathEngine.isAvailable();
        TestReporter.logMetric("GPU Operational", available);
        if (available) {
            String name = GPUMathEngine.getDeviceName();
            TestReporter.logMetric("GPU Device Name", name);
            assertTrue(name != null && name.length() > 0, "GPU device name must not be empty");
        }
    }

    private static void testGPUMatMulPrecision() {
        if (!GPUMathEngine.isAvailable()) {
            TestReporter.logMetric("GPU Test Status", "Skipped (No GPU Available)");
            return;
        }

        DeviceManager.setDevice(Device.CPU);
        Matrix a = new Matrix(new float[][]{
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}
        });
        Matrix b = new Matrix(new float[][]{
                {7.0f, 8.0f},
                {9.0f, 1.0f},
                {2.0f, 3.0f}
        });

        Matrix cpuResult = a.multiply(b);

        DeviceManager.setDevice(Device.GPU);
        Matrix gpuResult = a.multiply(b);

        float[][] cpuData = cpuResult.getData();
        float[][] gpuData = gpuResult.getData();

        for (int i = 0; i < cpuData.length; i++) {
            for (int j = 0; j < cpuData[0].length; j++) {
                assertEquals(cpuData[i][j], gpuData[i][j], 1e-4f, "GPU vs CPU MatMul mismatch at [" + i + "][" + j + "]");
            }
        }

        TestReporter.logMetric("CPU MatMul [0][0]", cpuData[0][0]);
        TestReporter.logMetric("GPU MatMul [0][0]", gpuData[0][0]);
        DeviceManager.setDevice(Device.CPU);
    }

    private static void testGPUTensorMatmul() {
        if (!GPUMathEngine.isAvailable()) {
            return;
        }

        DeviceManager.setDevice(Device.CPU);
        Tensor tA = new Tensor(new float[]{1, 2, 3, 4, 5, 6}, new int[]{1, 2, 3});
        Tensor tB = new Tensor(new float[]{7, 8, 9, 1, 2, 3}, new int[]{1, 3, 2});

        Tensor cpuOut = tA.matmul(tB);

        DeviceManager.setDevice(Device.GPU);
        Tensor gpuOut = tA.matmul(tB);

        float[] cpuData = cpuOut.getData();
        float[] gpuData = gpuOut.getData();

        assertTrue(Arrays.equals(cpuOut.getShape(), gpuOut.getShape()), "Tensor shape mismatch");
        for (int i = 0; i < cpuData.length; i++) {
            assertEquals(cpuData[i], gpuData[i], 1e-4f, "Tensor data mismatch at flat index " + i);
        }

        TestReporter.logMetric("Batched GPU Tensor MatMul shape", Arrays.toString(gpuOut.getShape()));
        DeviceManager.setDevice(Device.CPU);
    }
}
