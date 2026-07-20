package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.List;

/**
 * <h3>Cross Entropy Loss</h3>
 * 
 * <p>Computes the cross-entropy loss between model logits and target label IDs.</p>
 * <p>Supports input logits of shape [N, V] or 3D sequence predictions [B, T, V],
 * automatically flattening dimensions to compute loss and backpropagate gradients.</p>
 */
public class CrossEntropyLoss extends Module {

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("CrossEntropyLoss requires 2 inputs: logits and targets.");
        }
        Tensor logits = inputs[0];  // Shape [N, V] or [B, T, V]
        Tensor targets = inputs[1]; // Shape [N] or [B, T]

        int[] logitsShape = logits.getShape();
        int N, V;
        Tensor flatLogits = logits;

        if (logitsShape.length == 3) {
            N = logitsShape[0] * logitsShape[1];
            V = logitsShape[2];
            flatLogits = logits.reshape(N, V);
        } else if (logitsShape.length == 2) {
            N = logitsShape[0];
            V = logitsShape[1];
        } else {
            throw new IllegalArgumentException("Logits must be 2D [N, V] or 3D [B, T, V], got shape: " 
                + java.util.Arrays.toString(logitsShape));
        }

        int targetSize = targets.size();
        if (targetSize != N) {
            throw new IllegalArgumentException("Target size (" + targetSize + ") must match logits batch size N (" + N + ")");
        }

        // Ensure contiguity for safe array indexing
        Tensor contLogits = flatLogits.toContiguous();
        Tensor contTargets = targets.toContiguous();

        float[] logitsData = contLogits.getData();
        float[] targetsData = contTargets.getData();

        float[] probs = new float[N * V];
        float totalLoss = 0.0f;

        for (int i = 0; i < N; i++) {
            int targetIdx = (int) targetsData[i];
            if (targetIdx < 0 || targetIdx >= V) {
                throw new IndexOutOfBoundsException("Target label " + targetIdx + " is out of bounds for vocab size " + V);
            }
            
            // Find max logit for numerical stability (prevent overflow/underflow)
            float maxLogit = -Float.MAX_VALUE;
            for (int j = 0; j < V; j++) {
                float val = logitsData[i * V + j];
                if (val > maxLogit) {
                    maxLogit = val;
                }
            }

            // Compute sum(exp(x - max)) and store exp values
            float sumExp = 0.0f;
            for (int j = 0; j < V; j++) {
                float expVal = (float) Math.exp(logitsData[i * V + j] - maxLogit);
                probs[i * V + j] = expVal;
                sumExp += expVal;
            }

            // Probabilities and negative log-likelihood
            float invSumExp = 1.0f / sumExp;
            for (int j = 0; j < V; j++) {
                probs[i * V + j] *= invSumExp;
            }

            float sampleLoss = - (logitsData[i * V + targetIdx] - maxLogit) + (float) Math.log(sumExp);
            totalLoss += sampleLoss;
        }

        float avgLoss = totalLoss / N;
        Tensor lossTensor = new Tensor(new float[]{avgLoss}, new int[]{1});

        if (logits.requiresGrad()) {
            lossTensor.setRequiresGrad(true);
            
            final Tensor finalFlatLogits = flatLogits;
            final int finalN = N;
            final int finalV = V;
            
            lossTensor.setAutogradMetadata(
                List.of(finalFlatLogits),
                "cross_entropy",
                (gradOutput) -> {
                    if (finalFlatLogits.requiresGrad()) {
                        float[] incomingGrad = new float[finalN * finalV];
                        for (int i = 0; i < finalN; i++) {
                            int targetIdx = (int) targetsData[i];
                            for (int j = 0; j < finalV; j++) {
                                float p = probs[i * finalV + j];
                                float delta = (j == targetIdx) ? 1.0f : 0.0f;
                                // Scale by the incoming loss gradient (usually 1.0f) and average over N samples
                                incomingGrad[i * finalV + j] = gradOutput[0] * (p - delta) / finalN;
                            }
                        }
                        finalFlatLogits.accumulateGrad(incomingGrad);
                    }
                }
            );
        }

        return lossTensor;
    }
}
