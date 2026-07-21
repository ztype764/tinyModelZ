package com.tinymodelz.gpu;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * <h3>GPUMathEngine</h3>
 *
 * <p>Provides JNI native bindings to the OpenCL GPU compute backend.
 * Enables hardware-accelerated matrix operations on NVIDIA, AMD, and Intel GPUs.</p>
 */
public class GPUMathEngine {

    private static final Logger logger = LoggerFactory.getLogger(GPUMathEngine.class);
    private static boolean loaded = false;
    private static boolean initialized = false;
    private static String deviceName = "Unknown GPU";

    static {
        try {
            // 1. Try loading from working directory or system library path
            File localLib = new File("libtinymodelz_gpu.so");
            if (localLib.exists()) {
                System.load(localLib.getAbsolutePath());
                loaded = true;
            } else {
                // 2. Extract from classpath resources if available
                InputStream in = GPUMathEngine.class.getResourceAsStream("/native/libtinymodelz_gpu.so");
                if (in != null) {
                    File tempLib = File.createTempFile("libtinymodelz_gpu_", ".so");
                    tempLib.deleteOnExit();
                    Files.copy(in, tempLib.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    System.load(tempLib.getAbsolutePath());
                    loaded = true;
                }
            }

            if (loaded) {
                initialized = nInit();
                if (initialized) {
                    deviceName = nGetDeviceName();
                    logger.info("GPU Hardware Compute Engine initialized: {}", deviceName);
                } else {
                    logger.warn("GPU library loaded, but OpenCL context initialization failed. Defaulting to CPU.");
                }
            }
        } catch (Throwable t) {
            logger.warn("Native GPU library (libtinymodelz_gpu.so) unavailable: {}. Defaulting to CPU.", t.getMessage());
            loaded = false;
            initialized = false;
        }
    }

    // Native JNI function declarations
    private static native boolean nInit();
    private static native boolean nIsAvailable();
    private static native String nGetDeviceName();
    private static native boolean nMatMul(float[] a, float[] b, float[] c, int m, int n, int k);
    private static native boolean nBatchedMatMul(float[] a, float[] b, float[] c, int numBatches, int m, int n, int k);
    public static native long nAllocBuffer(long sizeBytes);
    public static native void nFreeBuffer(long handle);
    public static native boolean nCopyToGPU(long handle, float[] hData, long sizeBytes);
    public static native boolean nCopyFromGPU(float[] hData, long handle, long sizeBytes);
    public static native boolean nAdamWStep(long pHandle, long gHandle, long mHandle, long vHandle, int size, float lr, float beta1, float beta2, float eps, float weightDecay, float bc1, float bc2);


    /**
     * Checks whether the GPU hardware compute engine is initialized and operational.
     *
     * @return true if GPU acceleration is active
     */
    public static boolean isAvailable() {
        return loaded && initialized && nIsAvailable();
    }

    /**
     * Gets the hardware name of the detected GPU device.
     *
     * @return GPU device name string
     */
    public static String getDeviceName() {
        return deviceName;
    }

    /**
     * Performs matrix multiplication $C = A \times B$ on the GPU ($M \times K$ by $K \times N \to M \times N$).
     *
     * @param a input matrix A (flat array of size M * K)
     * @param b input matrix B (flat array of size K * N)
     * @param c output matrix C (flat array of size M * N)
     * @param m number of rows in A and C
     * @param n number of columns in B and C
     * @param k number of columns in A and rows in B
     * @return true if GPU computation succeeded, false if fallback to CPU is required
     */
    public static boolean matmul(float[] a, float[] b, float[] c, int m, int n, int k) {
        if (!isAvailable()) {
            return false;
        }
        try {
            return nMatMul(a, b, c, m, n, k);
        } catch (Throwable t) {
            logger.error("GPU MatMul execution error: {}", t.getMessage());
            return false;
        }
    }

    public static boolean batchedMatmul(float[] a, float[] b, float[] c, int numBatches, int m, int n, int k) {
        if (!isAvailable()) {
            return false;
        }
        try {
            return nBatchedMatMul(a, b, c, numBatches, m, n, k);
        } catch (Throwable t) {
            logger.error("GPU Batched MatMul execution error: {}", t.getMessage());
            return false;
        }
    }
}
