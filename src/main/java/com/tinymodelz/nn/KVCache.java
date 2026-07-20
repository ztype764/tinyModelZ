package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;

import java.util.Arrays;

/**
 * <h3>KVCache</h3>
 *
 * <p>Manages Key and Value tensor caching across Transformer layers for fast autoregressive decoding.</p>
 */
public class KVCache {

    public static class LayerCache {
        public Tensor keyCache;
        public Tensor valueCache;

        public void reset() {
            keyCache = null;
            valueCache = null;
        }
    }

    private final LayerCache[] layers;

    public KVCache(int numLayers) {
        this.layers = new LayerCache[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new LayerCache();
        }
    }

    public LayerCache getLayer(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.length) {
            throw new IndexOutOfBoundsException("Invalid layer index " + layerIndex + " for KVCache with " + layers.length + " layers.");
        }
        return layers[layerIndex];
    }

    public int getNumLayers() {
        return layers.length;
    }

    public void clear() {
        for (LayerCache layer : layers) {
            layer.reset();
        }
    }
}
