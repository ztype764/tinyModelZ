package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.Arrays;

/**
 * <h3>Multi-Head Attention</h3>
 * 
 * <p>Implements the Multi-Head Attention mechanism from "Attention Is All You Need".</p>
 * <p>Formulation: $\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W^O$</p>
 */
public class MultiHeadAttention extends Module {

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    
    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear outProj;
    
    private final Dropout attnDropout;
    private final Dropout residDropout;

    /**
     * Constructs a Multi-Head Attention layer.
     * 
     * @param embedDim total embedding dimension size (C)
     * @param numHeads number of attention heads (H)
     * @param dropoutProb dropout rate for attention weights and residual projections
     */
    public MultiHeadAttention(int embedDim, int numHeads, float dropoutProb) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        this.qProj = new Linear(embedDim, embedDim);
        this.kProj = new Linear(embedDim, embedDim);
        this.vProj = new Linear(embedDim, embedDim);
        this.outProj = new Linear(embedDim, embedDim);

        this.attnDropout = new Dropout(dropoutProb);
        this.residDropout = new Dropout(dropoutProb);

        // Register all parameters for optimizer visibility
        for (Tensor p : qProj.getParameters()) registerParameter(p);
        for (Tensor p : kProj.getParameters()) registerParameter(p);
        for (Tensor p : vProj.getParameters()) registerParameter(p);
        for (Tensor p : outProj.getParameters()) registerParameter(p);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("MultiHeadAttention requires at least 1 input tensor.");
        }
        Tensor x = inputs[0];
        Tensor mask = inputs.length > 1 ? inputs[1] : null;
        return forwardWithCache(x, mask, null);
    }

    public Tensor forwardWithCache(Tensor x, Tensor mask, KVCache.LayerCache layerCache) {
        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("Input tensor must be 3D [B, T, C]. Got: " + Arrays.toString(xShape));
        }

        int B = xShape[0];
        int T = xShape[1];
        int C = xShape[2];

        if (C != embedDim) {
            throw new IllegalArgumentException("Input feature dimension " + C + " does not match embedDim " + embedDim);
        }

        // Project Q, K, V
        Tensor x2d = x.reshape(B * T, C);
        Tensor q2d = qProj.forward(x2d);
        Tensor k2d = kProj.forward(x2d);
        Tensor v2d = vProj.forward(x2d);

        Tensor q = q2d.reshape(B, T, C);
        Tensor k = k2d.reshape(B, T, C);
        Tensor v = v2d.reshape(B, T, C);

        // Split into [B, H, T, D]
        Tensor qSplit = q.reshape(B, T, numHeads, headDim).transpose(1, 2);
        Tensor kSplit = k.reshape(B, T, numHeads, headDim).transpose(1, 2);
        Tensor vSplit = v.reshape(B, T, numHeads, headDim).transpose(1, 2);

        // Cache accumulation along sequence length dimension (dim 2)
        if (layerCache != null) {
            if (layerCache.keyCache != null && layerCache.valueCache != null) {
                kSplit = Tensor.cat(java.util.List.of(layerCache.keyCache, kSplit), 2);
                vSplit = Tensor.cat(java.util.List.of(layerCache.valueCache, vSplit), 2);
            }
            layerCache.keyCache = kSplit;
            layerCache.valueCache = vSplit;
        }

        // Compute scores = Q * K^T / sqrt(D)
        Tensor kTransposed = kSplit.transpose(2, 3); // [B, H, D, total_cached_T]
        Tensor scores = qSplit.matmul(kTransposed); // [B, H, T, total_cached_T]

        float scale = 1.0f / (float) Math.sqrt(headDim);
        scores = scores.multiply(scale);

        // Apply masking if present (only when processing multiple tokens with unpopulated cache)
        if (mask != null && T > 1 && (layerCache == null || layerCache.keyCache == null)) {
            scores = scores.maskedFill(mask, -1e9f);
        }

        // Attention probabilities
        Tensor attnProbs = scores.softmax();
        attnProbs = attnDropout.forward(attnProbs);

        // Attention output = probs * V
        Tensor out = attnProbs.matmul(vSplit); // [B, H, T, D]

        // Concatenate heads back to [B, T, C]
        Tensor outT = out.transpose(1, 2); // [B, T, H, D]
        Tensor outConcat = outT.reshape(B, T, C);

        // Project output and apply residual dropout
        Tensor out2d = outConcat.reshape(B * T, C);
        Tensor projected2d = outProj.forward(out2d);
        Tensor result2d = residDropout.forward(projected2d);

        return result2d.reshape(B, T, C);
    }

    @Override
    public void train() {
        super.train();
        qProj.train();
        kProj.train();
        vProj.train();
        outProj.train();
        attnDropout.train();
        residDropout.train();
    }

    @Override
    public void eval() {
        super.eval();
        qProj.eval();
        kProj.eval();
        vProj.eval();
        outProj.eval();
        attnDropout.eval();
        residDropout.eval();
    }

    public static Tensor createCausalMask(int seqLen) {
        float[] data = new float[seqLen * seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                if (j > i) {
                    data[i * seqLen + j] = 1.0f; // masked
                } else {
                    data[i * seqLen + j] = 0.0f; // unmasked
                }
            }
        }
        return new Tensor(data, new int[]{seqLen, seqLen});
    }
}
