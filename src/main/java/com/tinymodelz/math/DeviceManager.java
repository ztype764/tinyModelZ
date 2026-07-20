package com.tinymodelz.math;

import com.tinymodelz.gpu.CUDAMathEngine;
import com.tinymodelz.gpu.GPUMathEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <h3>DeviceManager</h3>
 *
 * <p>Manages execution compute target selection (CPU, GPU, OpenCL, CUDA) across the framework.</p>
 */
public class DeviceManager {

    private static final Logger logger = LoggerFactory.getLogger(DeviceManager.class);
    private static Device activeDevice = Device.CPU;

    /**
     * Sets the active compute device for matrix operations and training.
     *
     * @param device requested compute device (CPU, GPU, GPU_OPENCL, or GPU_CUDA)
     */
    public static synchronized void setDevice(Device device) {
        if (device == Device.GPU_CUDA) {
            if (CUDAMathEngine.isAvailable()) {
                activeDevice = Device.GPU_CUDA;
                logger.info("Switched compute execution target to CUDA GPU: {}", CUDAMathEngine.getDeviceName());
            } else {
                activeDevice = Device.CPU;
                logger.warn("CUDA GPU requested but hardware acceleration is unavailable. Defaulting to CPU.");
            }
        } else if (device == Device.GPU_OPENCL) {
            if (GPUMathEngine.isAvailable()) {
                activeDevice = Device.GPU_OPENCL;
                logger.info("Switched compute execution target to OpenCL GPU: {}", GPUMathEngine.getDeviceName());
            } else {
                activeDevice = Device.CPU;
                logger.warn("OpenCL GPU requested but hardware acceleration is unavailable. Defaulting to CPU.");
            }
        } else if (device == Device.GPU) {
            if (CUDAMathEngine.isAvailable()) {
                activeDevice = Device.GPU_CUDA;
                logger.info("Switched compute execution target to CUDA GPU: {}", CUDAMathEngine.getDeviceName());
            } else if (GPUMathEngine.isAvailable()) {
                activeDevice = Device.GPU_OPENCL;
                logger.info("Switched compute execution target to OpenCL GPU: {}", GPUMathEngine.getDeviceName());
            } else {
                activeDevice = Device.CPU;
                logger.warn("GPU requested but hardware acceleration is unavailable. Defaulting to CPU.");
            }
        } else {
            activeDevice = Device.CPU;
            logger.info("Switched compute execution target to CPU.");
        }
    }

    public static Device getDevice() {
        return activeDevice;
    }

    public static boolean isGpuActive() {
        return (activeDevice == Device.GPU_CUDA && CUDAMathEngine.isAvailable()) ||
               (activeDevice == Device.GPU_OPENCL && GPUMathEngine.isAvailable()) ||
               (activeDevice == Device.GPU && (CUDAMathEngine.isAvailable() || GPUMathEngine.isAvailable()));
    }

    public static String getSummary() {
        if (activeDevice == Device.GPU_CUDA && CUDAMathEngine.isAvailable()) {
            return "CUDA GPU (" + CUDAMathEngine.getDeviceName() + ")";
        } else if (activeDevice == Device.GPU_OPENCL && GPUMathEngine.isAvailable()) {
            return "OpenCL GPU (" + GPUMathEngine.getDeviceName() + ")";
        } else {
            return "CPU (Multi-threaded Java Engine)";
        }
    }
}
