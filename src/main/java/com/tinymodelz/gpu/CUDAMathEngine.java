package com.tinymodelz.gpu;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * <h3>CUDAMathEngine</h3>
 *
 * <p>Provides JNI native bindings to the NVIDIA CUDA GPU compute backend via PTX Driver API.</p>
 */
public class CUDAMathEngine {

    private static final Logger logger = LoggerFactory.getLogger(CUDAMathEngine.class);
    private static boolean loaded = false;
    private static boolean initialized = false;
    private static String deviceName = "Unknown CUDA Device";

    static {
        try {
            File localLib = new File("libtinymodelz_cuda.so");
            if (localLib.exists()) {
                System.load(localLib.getAbsolutePath());
                loaded = true;
            } else {
                InputStream in = CUDAMathEngine.class.getResourceAsStream("/native/libtinymodelz_cuda.so");
                if (in != null) {
                    File tempLib = File.createTempFile("libtinymodelz_cuda_", ".so");
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
                    logger.info("CUDA Hardware Compute Engine initialized: {}", deviceName);
                } else {
                    logger.warn("CUDA library loaded, but CUDA context initialization failed.");
                }
            }
        } catch (Throwable t) {
            logger.warn("Native CUDA library (libtinymodelz_cuda.so) unavailable: {}.", t.getMessage());
            loaded = false;
            initialized = false;
        }
    }

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

    public static boolean isAvailable() {
        return loaded && initialized && nIsAvailable();
    }

    public static String getDeviceName() {
        return deviceName;
    }

    public static boolean matmul(float[] a, float[] b, float[] c, int m, int n, int k) {
        if (!isAvailable()) {
            return false;
        }
        try {
            return nMatMul(a, b, c, m, n, k);
        } catch (Throwable t) {
            logger.error("CUDA MatMul execution error: {}", t.getMessage());
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
            logger.error("CUDA Batched MatMul execution error: {}", t.getMessage());
            return false;
        }
    }
}
