package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.Arrays;

/**
 * <h3>Feed-Forward Network (MLP)</h3>
 * 
 * <p>Implements the two-layer feed-forward network used in Transformer layers.</p>
 * <p>Formulation: $\mathrm{FFN}(x) = \mathrm{GeLU}(x W_1 + b_1) W_2 + b_2$</p>
 */
public class FeedForward extends Module {

    private final Linear cFc;
    private final Linear cProj;
    private final Dropout dropout;

    /**
     * Constructs a Feed-Forward network.
     * 
     * @param embedDim the embedding dimension size (C)
     * @param dropoutProb dropout rate
     */
    public FeedForward(int embedDim, float dropoutProb) {
        this.cFc = new Linear(embedDim, 4 * embedDim);
        this.cProj = new Linear(4 * embedDim, embedDim);
        this.dropout = new Dropout(dropoutProb);

        // Register parameters
        for (Tensor p : cFc.getParameters()) registerParameter(p);
        for (Tensor p : cProj.getParameters()) registerParameter(p);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("FeedForward requires at least 1 input tensor.");
        }
        Tensor x = inputs[0];

        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("Input tensor must be 3D [B, T, C]. Got: " + Arrays.toString(xShape));
        }

        int B = xShape[0];
        int T = xShape[1];
        int C = xShape[2];

        Tensor x2d = x.reshape(B * T, C);
        Tensor h = cFc.forward(x2d).gelu();
        Tensor projected = cProj.forward(h);
        Tensor result2d = dropout.forward(projected);

        return result2d.reshape(B, T, C);
    }

    @Override
    public void train() {
        super.train();
        cFc.train();
        cProj.train();
        dropout.train();
    }

    @Override
    public void eval() {
        super.eval();
        cFc.eval();
        cProj.eval();
        dropout.eval();
    }
}
