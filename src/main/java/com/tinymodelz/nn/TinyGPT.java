package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;

import java.util.Arrays;

/**
 * <h3>TinyGPT</h3>
 *
 * <p>
 * A minimal GPT-style decoder-only language model assembled from the TinyModelZ
 * component library. Follows the GPT-2 Pre-LayerNorm architecture.
 * </p>
 *
 * <h4>Architecture:</h4>
 * 
 * <pre>
 * tokens [B, T]
 *    ↓
 * Token Embedding  (vocabSize → embedDim)
 *    +
 * Positional Embedding (maxSeqLen → embedDim)
 *    ↓
 * TransformerBlock × numLayers
 *    ↓
 * Final LayerNorm
 *    ↓
 * Linear Head (embedDim → vocabSize)
 *    ↓
 * Logits [B, T, vocabSize]
 * </pre>
 *
 * <h4>Mathematical Formulation:</h4>
 * <p>
 * Given a token sequence $x = (x_1, x_2, \dots, x_T)$:
 * </p>
 * <ol>
 * <li>$h^0 = W_e[x] + W_p[\mathrm{pos}]$ where $W_e \in \mathbb{R}^{|V| \times
 * d}$
 * and $W_p \in \mathbb{R}^{T_{max} \times d}$</li>
 * <li>$h^l = \text{TransformerBlock}^l(h^{l-1})$ for $l = 1, \dots, L$</li>
 * <li>$h^{final} = \text{LayerNorm}(h^L)$</li>
 * <li>$\text{logits} = h^{final} W_{lm}^T + b_{lm}$ where $W_{lm} \in
 * \mathbb{R}^{|V| \times d}$</li>
 * </ol>
 */
public class TinyGPT extends Module {

    private final int vocabSize;
    private final int embedDim;
    private final int maxSeqLen;
    private final int numLayers;
    private final int numHeads;

    private final Embedding tokenEmbedding;
    private final Embedding positionEmbedding;
    private final TransformerBlock[] blocks;
    private final LayerNorm finalNorm;
    private final Linear lmHead;

    /**
     * Constructs a TinyGPT language model.
     *
     * @param vocabSize   vocabulary size (|V|)
     * @param embedDim    embedding/hidden dimension (d)
     * @param maxSeqLen   maximum sequence length (T_max)
     * @param numLayers   number of stacked transformer blocks (L)
     * @param numHeads    number of attention heads per block (H)
     * @param dropoutProb dropout probability applied inside attention and
     *                    feed-forward layers
     */
    public TinyGPT(int vocabSize, int embedDim, int maxSeqLen, int numLayers, int numHeads, float dropoutProb) {
        this(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, 4 * embedDim, dropoutProb);
    }

    /**
     * Constructs a TinyGPT language model with custom feed-forward layer dimension.
     *
     * @param vocabSize      vocabulary size (|V|)
     * @param embedDim       embedding/hidden dimension (d)
     * @param maxSeqLen      maximum sequence length (T_max)
     * @param numLayers      number of stacked transformer blocks (L)
     * @param numHeads       number of attention heads per block (H)
     * @param feedForwardDim feed-forward hidden dimension (d_ff)
     * @param dropoutProb    dropout probability applied inside attention and
     *                       feed-forward layers
     */
    public TinyGPT(int vocabSize, int embedDim, int maxSeqLen, int numLayers, int numHeads, int feedForwardDim,
            float dropoutProb) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                    "embedDim (" + embedDim + ") must be divisible by numHeads (" + numHeads + ")");
        }

        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        this.numHeads = numHeads;

        // --- Token Embedding ---
        this.tokenEmbedding = new Embedding(vocabSize, embedDim);
        for (Tensor p : tokenEmbedding.getParameters())
            registerParameter(p);

        // --- Positional Embedding ---
        this.positionEmbedding = new Embedding(maxSeqLen, embedDim);
        for (Tensor p : positionEmbedding.getParameters())
            registerParameter(p);

        // --- Transformer Blocks ---
        this.blocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            blocks[i] = new TransformerBlock(embedDim, numHeads, feedForwardDim, dropoutProb);
            for (Tensor p : blocks[i].getParameters())
                registerParameter(p);
        }

        // --- Final LayerNorm ---
        this.finalNorm = new LayerNorm(embedDim);
        for (Tensor p : finalNorm.getParameters())
            registerParameter(p);

        // --- LM Head projection ---
        this.lmHead = new Linear(embedDim, vocabSize);
        for (Tensor p : lmHead.getParameters())
            registerParameter(p);
    }

    /**
     * Forward pass of TinyGPT.
     *
     * @param inputs inputs[0] = token index tensor of shape [B, T]
     * @return logits tensor of shape [B, T, vocabSize]
     */
    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("TinyGPT forward requires a token index tensor.");
        }
        Tensor tokenIds = inputs[0]; // [B, T]
        int[] idShape = tokenIds.getShape();

        if (idShape.length != 2) {
            throw new IllegalArgumentException(
                    "Expected token tensor of shape [B, T], got: " + Arrays.toString(idShape));
        }

        int B = idShape[0];
        int T = idShape[1];

        if (T > maxSeqLen) {
            throw new IllegalArgumentException(
                    "Sequence length " + T + " exceeds maxSeqLen " + maxSeqLen);
        }

        // --- 1. Token Embedding: [B, T] → [B, T, embedDim] ---
        Tensor tokEmb = tokenEmbedding.forward(tokenIds);

        // --- 2. Positional Embedding: build position indices [1, T] → [1, T, embedDim]
        // ---
        float[] posData = new float[T];
        for (int i = 0; i < T; i++) {
            posData[i] = i;
        }
        Tensor posIds = new Tensor(posData, new int[] { 1, T });
        Tensor posEmb = positionEmbedding.forward(posIds); // [1, T, embedDim] — broadcasts over B

        // --- 3. Sum token + position embeddings ---
        Tensor hidden = tokEmb.add(posEmb); // [B, T, embedDim]

        // --- 4. Causal mask ---
        Tensor mask = MultiHeadAttention.createCausalMask(T);

        // --- 5. Transformer Blocks ---
        for (int i = 0; i < numLayers; i++) {
            hidden = blocks[i].forward(hidden, mask);
        }

        // --- 6. Final LayerNorm ---
        hidden = finalNorm.forward(hidden);

        // --- 7. LM Head projection: [B, T, embedDim] → [B, T, vocabSize] ---
        // Linear expects 2D input, so reshape, project, then reshape back.
        Tensor hidden2d = hidden.reshape(B * T, embedDim);
        Tensor logits2d = lmHead.forward(hidden2d);
        Tensor logits = logits2d.reshape(B, T, vocabSize);

        return logits;
    }

    @Override
    public void train() {
        super.train();
        tokenEmbedding.train();
        positionEmbedding.train();
        for (TransformerBlock block : blocks)
            block.train();
        finalNorm.train();
        lmHead.train();
    }

    @Override
    public void eval() {
        super.eval();
        tokenEmbedding.eval();
        positionEmbedding.eval();
        for (TransformerBlock block : blocks)
            block.eval();
        finalNorm.eval();
        lmHead.eval();
    }

    // --- Accessors ---

    public int getVocabSize() {
        return vocabSize;
    }

    public int getEmbedDim() {
        return embedDim;
    }

    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public int getNumHeads() {
        return numHeads;
    }

    /**
     * Returns a formatted summary string describing the model architecture and
     * parameter count.
     */
    public String summary() {
        int totalParams = 0;
        for (Tensor p : getParameters()) {
            totalParams += p.size();
        }
        return String.format(
                "TinyGPT(vocab=%d, embed=%d, maxSeq=%d, layers=%d, heads=%d, params=%,d)",
                vocabSize, embedDim, maxSeqLen, numLayers, numHeads, totalParams);
    }
}
