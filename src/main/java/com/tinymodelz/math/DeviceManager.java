package com.tinymodelz.math;

import com.tinymodelz.gpu.GPUMathEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <h3>DeviceManager</h3>
 *
 * <p>Manages execution compute target selection (CPU vs. GPU) across the framework.</p>
 */
public class DeviceManager {

    private static final Logger logger = LoggerFactory.getLogger(DeviceManager.class);
    private static Device activeDevice = Device.CPU;

    /**
     * Sets the active compute device for matrix operations and training.
     *
     * @param device requested compute device (CPU or GPU)
     */
    public static synchronized void setDevice(Device device) {
        if (device == Device.GPU) {
            if (GPUMathEngine.isAvailable()) {
                activeDevice = Device.GPU;
                logger.info("Switched compute execution target to GPU: {}", GPUMathEngine.getDeviceName());
            } else {
                activeDevice = Device.CPU;
                logger.warn("GPU requested but hardware acceleration is unavailable. Defaulting to CPU.");
            }
        } else {
            activeDevice = Device.CPU;
            logger.info("Switched compute execution target to CPU.");
        }
    }

    /**
     * Gets the current active compute device.
     *
     * @return active device (CPU or GPU)
     */
    public static Device getDevice() {
        return activeDevice;
    }

    /**
     * Checks whether GPU acceleration is active and operational.
     *
     * @return true if active device is GPU and hardware engine is operational
     */
    public static boolean isGpuActive() {
        return activeDevice == Device.GPU && GPUMathEngine.isAvailable();
    }

    /**
     * Returns a summary description of the active device.
     *
     * @return device description string
     */
    public static String getSummary() {
        if (isGpuActive()) {
            return "GPU (" + GPUMathEngine.getDeviceName() + ")";
        } else {
            return "CPU (Multi-threaded Java Engine)";
        }
    }
}
