package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class KVCacheTest {

    @Test
    public void testKVCacheEquivalence() {
        int vocabSize = 64;
        int embedDim = 32;
        int maxSeqLen = 64;
        int numLayers = 2;
        int numHeads = 4;
        float dropout = 0.0f;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, dropout);
        model.eval();

        // Prompt token indices: [1, 5]
        float[] promptTokens = new float[]{1, 10, 15, 20, 25};
        Tensor promptTensor = new Tensor(promptTokens, new int[]{1, 5});

        // Standard forward pass on full sequence [1, 5]
        Tensor fullLogits = model.forward(promptTensor);
        float[] fullData = fullLogits.toContiguous().getData();

        // KV Cache forward pass: step 1 (prompt [1, 4]) + step 2 (single token [1, 1])
        KVCache cache = new KVCache(numLayers);
        float[] p1Data = new float[]{1, 10, 15, 20};
        Tensor p1Tensor = new Tensor(p1Data, new int[]{1, 4});
        model.forwardWithCache(p1Tensor, cache, 0);

        Tensor step2Tensor = new Tensor(new float[]{25}, new int[]{1, 1});
        Tensor cachedStep2Logits = model.forwardWithCache(step2Tensor, cache, 4);
        float[] cachedData = cachedStep2Logits.toContiguous().getData();

        // Check that last token logits match between standard forward and KV cache forward
        int offsetFull = (5 - 1) * vocabSize;
        for (int i = 0; i < vocabSize; i++) {
            assertEquals(fullData[offsetFull + i], cachedData[i], 1e-4f, "Logit mismatch at idx " + i);
        }
    }
}
