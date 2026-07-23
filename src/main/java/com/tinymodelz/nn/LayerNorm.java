package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.List;

/**
 * <h3>LayerNorm</h3>
 * 
 * <p>Applies Layer Normalization over a mini-batch of inputs.</p>
 * <p>Formulation: $y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \gamma + \beta$</p>
 */
public class LayerNorm extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final int normalizedDim;
    private final float eps;

    /**
     * Constructs a LayerNorm layer with epsilon defaulting to 1e-5.
     * 
     * @param normalizedDim the size of the dimension to normalize over (typically embeddingDim)
     */
    public LayerNorm(int normalizedDim) {
        this(normalizedDim, 1e-5f);
    }

    /**
     * Constructs a LayerNorm layer.
     * 
     * @param normalizedDim the size of the dimension to normalize over
     * @param eps a value added to the denominator for numerical stability
     */
    public LayerNorm(int normalizedDim, float eps) {
        this.normalizedDim = normalizedDim;
        this.eps = eps;

        // Learnable scaling parameter gamma (initialized to 1.0)
        this.weight = Tensor.ones(1, normalizedDim);
        this.weight.setRequiresGrad(true);
        registerParameter(this.weight);

        // Learnable shift parameter beta (initialized to 0.0)
        this.bias = Tensor.zeros(1, normalizedDim);
        this.bias.setRequiresGrad(true);
        registerParameter(this.bias);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("LayerNorm forward pass requires at least 1 input tensor.");
        }
        Tensor x = inputs[0];
        int[] originalShape = x.getShape();
        int lastDim = originalShape[originalShape.length - 1];
        if (lastDim != normalizedDim) {
            throw new IllegalArgumentException("Input last dimension " + lastDim + " does not match LayerNorm normalizedDim " + normalizedDim);
        }

        int D = normalizedDim;
        int N = x.size() / D;

        if (com.tinymodelz.math.DeviceManager.isGpuActive() && com.tinymodelz.math.DeviceManager.getDevice() == com.tinymodelz.math.Device.GPU_CUDA && com.tinymodelz.gpu.CUDAMathEngine.isAvailable()) {
            x.toGPU();
            weight.toGPU();
            bias.toGPU();
            Tensor result = new Tensor(originalShape);
            result.toGPU();

            long meanHandle = com.tinymodelz.gpu.CUDAMathEngine.nAllocBuffer((long) N * Float.BYTES);
            long rstdHandle = com.tinymodelz.gpu.CUDAMathEngine.nAllocBuffer((long) N * Float.BYTES);

            if (meanHandle != 0 && rstdHandle != 0) {
                boolean ok = com.tinymodelz.gpu.CUDAMathEngine.nLayerNormForward(x.getGpuBufferHandle(), weight.getGpuBufferHandle(), bias.getGpuBufferHandle(), result.getGpuBufferHandle(), meanHandle, rstdHandle, N, D, eps);
                com.tinymodelz.gpu.CUDAMathEngine.nFreeBuffer(meanHandle);
                com.tinymodelz.gpu.CUDAMathEngine.nFreeBuffer(rstdHandle);
                if (ok) {
                    result.markDirtyOnHost();
                    if (x.requiresGrad() || weight.requiresGrad() || bias.requiresGrad()) {
                        result.setRequiresGrad(true);
                        result.setAutogradMetadata(
                            List.of(x, weight, bias),
                            "layernorm",
                            (gradOutput) -> {
                                Tensor gradOutTensor = new Tensor(gradOutput, originalShape);
                                if (x.requiresGrad()) {
                                    x.accumulateGrad(gradOutTensor.getData());
                                }
                            }
                        );
                    }
                    return result;
                }
            }
        }

        // Ensure we operate on contiguous data
        Tensor contiguousX = x.toContiguous();
        float[] xData = contiguousX.getData();
        int xOffset = contiguousX.offset();

        float[] invStd = new float[N];
        float[] xHat = new float[N * D];
        float[] outData = new float[N * D];

        float[] wData = weight.getData();
        float[] bData = bias.getData();

        java.util.stream.IntStream.range(0, N).parallel().forEach(i -> {
            int rowOffset = i * D;
            
            // Calculate Mean (mu)
            float sum = 0.0f;
            for (int j = 0; j < D; j++) {
                sum += xData[xOffset + rowOffset + j];
            }
            float mu = sum / D;

            // Calculate Variance
            float varSum = 0.0f;
            for (int j = 0; j < D; j++) {
                float diff = xData[xOffset + rowOffset + j] - mu;
                varSum += diff * diff;
            }
            float var = varSum / D;
            float stdInv = (float) (1.0 / Math.sqrt(var + eps));
            invStd[i] = stdInv;

            // Apply normalization
            for (int j = 0; j < D; j++) {
                float normalizedVal = (xData[xOffset + rowOffset + j] - mu) * stdInv;
                xHat[rowOffset + j] = normalizedVal;
                outData[rowOffset + j] = normalizedVal * wData[weight.offset() + j] + bData[bias.offset() + j];
            }
        });

        Tensor result = new Tensor(outData, originalShape);
        if (x.requiresGrad() || weight.requiresGrad() || bias.requiresGrad()) {
            result.setRequiresGrad(true);
            result.setAutogradMetadata(
                List.of(x, weight, bias),
                "layernorm",
                (gradOutput) -> {
                    int wOffset = weight.offset();
                    int bOffset = bias.offset();

                    // Accumulate parameter gradients
                    if (weight.requiresGrad() || bias.requiresGrad()) {
                        float[] wGrad = weight.getGrad();
                        float[] bGrad = bias.getGrad();
                        if (weight.requiresGrad() && wGrad == null) {
                            weight.accumulateGrad(new float[weight.getData().length]);
                            wGrad = weight.getGrad();
                        }
                        if (bias.requiresGrad() && bGrad == null) {
                            bias.accumulateGrad(new float[bias.getData().length]);
                            bGrad = bias.getGrad();
                        }
                        for (int i = 0; i < N; i++) {
                            int rowOffset = i * D;
                            for (int j = 0; j < D; j++) {
                                float dy = gradOutput[rowOffset + j];
                                float xh = xHat[rowOffset + j];
                                
                                if (weight.requiresGrad()) {
                                    wGrad[wOffset + j] += dy * xh;
                                }
                                if (bias.requiresGrad()) {
                                    bGrad[bOffset + j] += dy;
                                }
                            }
                        }
                    }

                    // Compute input gradient dX in parallel
                    if (x.requiresGrad()) {
                        if (x.getGrad() == null) {
                            x.accumulateGrad(new float[x.getData().length]);
                        }
                        float[] xGrad = x.getGrad();
                        int xGradOffset = x.offset();

                        java.util.stream.IntStream.range(0, N).parallel().forEach(i -> {
                            int rowOffset = i * D;
                            float stdInv = invStd[i];
                            
                            float dBetaPrime = 0.0f;
                            float dGammaPrime = 0.0f;
                            for (int k = 0; k < D; k++) {
                                float dy = gradOutput[rowOffset + k];
                                float wk = wData[wOffset + k];
                                float xhk = xHat[rowOffset + k];
                                dBetaPrime += dy * wk;
                                dGammaPrime += dy * wk * xhk;
                            }

                            for (int j = 0; j < D; j++) {
                                float dy = gradOutput[rowOffset + j];
                                float wj = wData[wOffset + j];
                                float xhj = xHat[rowOffset + j];
                                
                                float dxVal = (stdInv / D) * (D * dy * wj - dBetaPrime - xhj * dGammaPrime);
                                xGrad[xGradOffset + rowOffset + j] += dxVal;
                            }
                        });
                    }
                }
            );
        }
        return result;
    }

    public Tensor getWeight() {
        return weight;
    }

    public Tensor getBias() {
        return bias;
    }

    public int getNormalizedDim() {
        return normalizedDim;
    }
}
