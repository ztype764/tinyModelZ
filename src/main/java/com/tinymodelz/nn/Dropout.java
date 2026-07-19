package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.List;
import java.util.Random;

/**
 * <h3>Dropout</h3>
 * 
 * <p>During training, randomly zeroes some of the elements of the input tensor with probability $p$.
 * Uses inverted dropout scaling: $y = \frac{x \odot \text{mask}}{1 - p}$ to keep expectations consistent.</p>
 */
public class Dropout extends Module {

    private final float p;
    private final Random random;

    /**
     * Constructs a Dropout layer.
     * 
     * @param p dropout probability (0.0 <= p < 1.0)
     */
    public Dropout(float p) {
        this.p = p;
        this.random = new Random();
    }

    /**
     * Constructs a Dropout layer with a specific seed for reproducibility.
     * 
     * @param p dropout probability
     * @param seed random seed value
     */
    public Dropout(float p, long seed) {
        this.p = p;
        this.random = new Random(seed);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Dropout forward pass requires at least 1 input tensor.");
        }
        Tensor x = inputs[0];
        
        // Identity mapping in evaluation mode or when p = 0
        if (!training || p == 0.0f) {
            return x;
        }

        int size = x.size();
        float[] outData = new float[size];
        float[] mask = new float[size];
        float scale = 1.0f / (1.0f - p);

        Tensor contiguousX = x.toContiguous();
        float[] xData = contiguousX.getData();
        int xOffset = contiguousX.offset();

        for (int i = 0; i < size; i++) {
            if (random.nextFloat() >= p) {
                mask[i] = 1.0f;
                outData[i] = xData[xOffset + i] * scale;
            } else {
                mask[i] = 0.0f;
                outData[i] = 0.0f;
            }
        }

        Tensor result = new Tensor(outData, x.getShape());
        if (x.requiresGrad()) {
            result.setRequiresGrad(true);
            result.setAutogradMetadata(
                List.of(x),
                "dropout",
                (gradOutput) -> {
                    if (x.getGrad() == null) {
                        x.accumulateGrad(new float[x.getData().length]);
                    }
                    float[] xGrad = x.getGrad();
                    int xGradOffset = x.offset();
                    for (int i = 0; i < size; i++) {
                        xGrad[xGradOffset + i] += gradOutput[i] * mask[i] * scale;
                    }
                }
            );
        }
        return result;
    }

    public float getProbability() {
        return p;
    }
}
