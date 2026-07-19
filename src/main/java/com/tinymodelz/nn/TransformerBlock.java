package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;

/**
 * <h3>Transformer Block</h3>
 * 
 * <p>Implements a single Transformer layer with Pre-LayerNormalization architecture (GPT-2 style).</p>
 * <p>Formulation:
 * $$x_{attn} = x + \mathrm{Attention}(\mathrm{LN}_1(x))$$
 * $$x_{out} = x_{attn} + \mathrm{MLP}(\mathrm{LN}_2(x_{attn}))$$
 * </p>
 */
public class TransformerBlock extends Module {

    private final LayerNorm ln1;
    private final MultiHeadAttention attn;
    private final LayerNorm ln2;
    private final FeedForward mlp;

    /**
     * Constructs a Transformer block.
     * 
     * @param embedDim embedding size (C)
     * @param numHeads number of attention heads (H)
     * @param dropoutProb dropout probability rate
     */
    public TransformerBlock(int embedDim, int numHeads, float dropoutProb) {
        this(embedDim, numHeads, 4 * embedDim, dropoutProb);
    }

    /**
     * Constructs a Transformer block with custom feed-forward dimension.
     * 
     * @param embedDim embedding size (C)
     * @param numHeads number of attention heads (H)
     * @param feedForwardDim feed-forward hidden dimension (d_ff)
     * @param dropoutProb dropout probability rate
     */
    public TransformerBlock(int embedDim, int numHeads, int feedForwardDim, float dropoutProb) {
        this.ln1 = new LayerNorm(embedDim);
        this.attn = new MultiHeadAttention(embedDim, numHeads, dropoutProb);
        this.ln2 = new LayerNorm(embedDim);
        this.mlp = new FeedForward(embedDim, feedForwardDim, dropoutProb);

        // Register all parameters
        for (Tensor p : ln1.getParameters()) registerParameter(p);
        for (Tensor p : attn.getParameters()) registerParameter(p);
        for (Tensor p : ln2.getParameters()) registerParameter(p);
        for (Tensor p : mlp.getParameters()) registerParameter(p);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("TransformerBlock requires at least 1 input tensor.");
        }
        Tensor x = inputs[0];
        Tensor mask = inputs.length > 1 ? inputs[1] : null;

        // Pre-LN Attention Branch
        Tensor normX = ln1.forward(x);
        Tensor attnOut = attn.forward(normX, mask);
        Tensor xAttn = x.add(attnOut);

        // Pre-LN Feed-Forward Branch
        Tensor normXAttn = ln2.forward(xAttn);
        Tensor mlpOut = mlp.forward(normXAttn);
        Tensor result = xAttn.add(mlpOut);

        return result;
    }

    @Override
    public void train() {
        super.train();
        ln1.train();
        attn.train();
        ln2.train();
        mlp.train();
    }

    @Override
    public void eval() {
        super.eval();
        ln1.eval();
        attn.eval();
        ln2.eval();
        mlp.eval();
    }
}
