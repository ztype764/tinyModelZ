package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;

/**
 * <h3>RotaryEmbedding (RoPE)</h3>
 *
 * <p>Implements Rotary Position Embeddings (RoPE) for transformers.</p>
 * <p>RoPE encodes absolute position with a rotation matrix and naturally incorporates relative
 * positional information into self-attention query ($Q$) and key ($K$) projections.</p>
 *
 * <h4>Mathematical Formulation:</h4>
 * $$\theta_i = 10000^{-2i/d}$$
 * $$R_{\Theta, m}^d x_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)$$
 * $$R_{\Theta, m}^d x_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)$$
 */
public class RotaryEmbedding {

    private final int dim;
    private final float base;

    /**
     * Constructs a Rotary Position Embedding helper.
     *
     * @param dim feature dimension (must be even)
     * @param base frequency theta base (default 10000.0)
     */
    public RotaryEmbedding(int dim, float base) {
        if (dim % 2 != 0) {
            throw new IllegalArgumentException("RoPE feature dimension must be even. Got: " + dim);
        }
        this.dim = dim;
        this.base = base;
    }

    public RotaryEmbedding(int dim) {
        this(dim, 10000.0f);
    }

    /**
     * Applies rotary embedding to a 3D or 4D tensor $x$ of shape [..., T, dim].
     *
     * @param x input tensor (e.g. Query or Key tensor)
     * @return rotated output tensor of identical shape
     */
    public Tensor apply(Tensor x) {
        int[] shape = x.getShape();
        int T = shape[shape.length - 2];
        int D = shape[shape.length - 1];

        if (D != dim) {
            throw new IllegalArgumentException("Tensor dimension " + D + " does not match RoPE dim " + dim);
        }

        Tensor xCont = x.toContiguous();
        float[] inData = xCont.getData();
        float[] outData = new float[inData.length];
        System.arraycopy(inData, 0, outData, 0, inData.length);

        int outerSize = inData.length / (T * D);

        for (int b = 0; b < outerSize; b++) {
            for (int t = 0; t < T; t++) {
                int baseOffset = (b * T + t) * D;
                for (int i = 0; i < D / 2; i++) {
                    float theta = (float) Math.pow(base, -2.0 * i / D);
                    float angle = t * theta;
                    float cos = (float) Math.cos(angle);
                    float sin = (float) Math.sin(angle);

                    int idx0 = baseOffset + 2 * i;
                    int idx1 = baseOffset + 2 * i + 1;

                    float x0 = inData[idx0];
                    float x1 = inData[idx1];

                    outData[idx0] = x0 * cos - x1 * sin;
                    outData[idx1] = x0 * sin + x1 * cos;
                }
            }
        }

        return new Tensor(outData, shape);
    }
}
