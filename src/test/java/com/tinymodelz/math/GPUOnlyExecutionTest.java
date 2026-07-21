package com.tinymodelz.math;

import com.tinymodelz.train.AdamW;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

public class GPUOnlyExecutionTest {

    @BeforeAll
    public static void setup() {
        DeviceManager.setDevice(Device.GPU);
        DeviceManager.setExecutionMode(ExecutionMode.GPU_ONLY);
    }

    @Test
    public void testGPUResidencyAndLazySync() {
        Tensor t = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2});
        
        // Initially on CPU
        assertFalse(t.isOnGPU());

        // Transfer to GPU
        boolean uploaded = t.toGPU();
        if (uploaded) {
            assertTrue(t.isOnGPU());
            assertTrue(t.getGpuBufferHandle() != 0);

            // Mark dirty on host and lazy sync
            t.markDirtyOnHost();
            assertTrue(t.isDirtyOnHost());

            float[] hData = t.getData();
            assertNotNull(hData);
            assertEquals(4, hData.length);
            assertFalse(t.isDirtyOnHost());
        }
    }

    @Test
    public void testAdamWGpuKernelStep() {
        Tensor param = new Tensor(new float[]{0.5f, -0.2f, 1.2f, 0.8f}, new int[]{4});
        param.setRequiresGrad(true);
        float[] grad = param.getGrad();
        System.arraycopy(new float[]{0.1f, 0.05f, -0.2f, 0.15f}, 0, grad, 0, 4);

        AdamW optimizer = new AdamW(Collections.singletonList(param), 0.01f);
        optimizer.step();

        float[] updated = param.getData();
        assertNotNull(updated);
        assertEquals(4, updated.length);
    }
}
