package com.tinymodelz.math;

/**
 * <h3>ExecutionMode</h3>
 *
 * <p>Defines compute execution modes for model training and tensor lifecycles.</p>
 * <ul>
 *   <li><b>CPU</b>: Complete execution on host CPU.</li>
 *   <li><b>HYBRID</b>: Selective JNI GPU acceleration for compute-heavy matmuls with CPU host residency.</li>
 *   <li><b>GPU_ONLY</b>: Full GPU residency for forward, backward, activations, gradients, and AdamW optimizer updates.</li>
 * </ul>
 */
public enum ExecutionMode {
    CPU,
    HYBRID,
    GPU_ONLY
}
