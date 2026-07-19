package com.tinymodelz.math;

/**
 * <h3>Device</h3>
 *
 * <p>Represents compute target devices for tensor calculations and model training.</p>
 */
public enum Device {
    /**
     * Standard multi-threaded CPU execution backend.
     */
    CPU,

    /**
     * Hardware-accelerated GPU compute backend powered by OpenCL/CUDA.
     */
    GPU
}
