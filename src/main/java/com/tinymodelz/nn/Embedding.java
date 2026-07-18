package com.tinymodelz.nn;

import com.tinymodelz.math.Matrix;
import com.tinymodelz.math.RandomInitializer;
import com.tinymodelz.math.Tensor;

import java.util.List;

/**
 * <h3>Embedding</h3>
 * 
 * <p>A lookup table that stores embeddings of a fixed dictionary and size.</p>
 * <p>Maps integer/discrete token indices to dense representation vectors.</p>
 */
public class Embedding extends Module {

    private final Tensor weight;
    private final int numEmbeddings;
    private final int embeddingDim;

    /**
     * Constructs an Embedding layer.
     * 
     * @param numEmbeddings size of the dictionary of embeddings (vocab size)
     * @param embeddingDim size of each embedding vector
     */
    public Embedding(int numEmbeddings, int embeddingDim) {
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        // Initialize weights using normal distribution: shape [numEmbeddings, embeddingDim]
        RandomInitializer initializer = new RandomInitializer();
        Matrix weightMatrix = new Matrix(numEmbeddings, embeddingDim);
        initializer.fillNormal(weightMatrix, 0.0f, 1.0f);
        
        this.weight = Tensor.fromMatrix(weightMatrix);
        this.weight.setRequiresGrad(true);
        registerParameter(this.weight);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Embedding forward pass requires an index tensor.");
        }
        Tensor indices = inputs[0];
        int[] indexShape = indices.getShape();
        
        // Output shape: indexShape + [embeddingDim]
        int[] outShape = new int[indexShape.length + 1];
        System.arraycopy(indexShape, 0, outShape, 0, indexShape.length);
        outShape[indexShape.length] = embeddingDim;
        
        int numElements = indices.size();
        float[] outData = new float[numElements * embeddingDim];
        
        // Retrieve embeddings row by row
        for (int i = 0; i < numElements; i++) {
            int tokenIdx = (int) indices.getValByFlatIndex(i);
            if (tokenIdx < 0 || tokenIdx >= numEmbeddings) {
                throw new IndexOutOfBoundsException("Token ID " + tokenIdx + " is out of bounds for vocab size " + numEmbeddings);
            }
            int outOffset = i * embeddingDim;
            for (int d = 0; d < embeddingDim; d++) {
                outData[outOffset + d] = weight.getValByFlatIndex(tokenIdx * embeddingDim + d);
            }
        }
        
        Tensor result = new Tensor(outData, outShape);
        if (weight.requiresGrad()) {
            result.setRequiresGrad(true);
            result.setAutogradMetadata(
                List.of(weight),
                "embedding",
                (gradOutput) -> {
                    float[] weightGrad = weight.getGrad();
                    int weightOffset = weight.offset();
                    for (int i = 0; i < numElements; i++) {
                        int tokenIdx = (int) indices.getValByFlatIndex(i);
                        int outOffset = i * embeddingDim;
                        for (int d = 0; d < embeddingDim; d++) {
                            weightGrad[weightOffset + tokenIdx * embeddingDim + d] += gradOutput[outOffset + d];
                        }
                    }
                }
            );
        }
        return result;
    }

    public Tensor getWeight() {
        return weight;
    }

    public int getNumEmbeddings() {
        return numEmbeddings;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }
}
