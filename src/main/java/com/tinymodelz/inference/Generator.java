package com.tinymodelz.inference;

import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.Module;
import com.tinymodelz.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * <h3>Generator</h3>
 * 
 * <p>
 * Handles autoregressive text generation using a trained language model.
 * </p>
 * <p>
 * Implements decoding strategies:
 * <ul>
 * <li>Greedy Decoding (argmax)</li>
 * <li>Temperature Scaling</li>
 * <li>Top-K Filtering</li>
 * <li>Top-P (Nucleus) Filtering</li>
 * </ul>
 * </p>
 */
public class Generator {

    private final Random random;

    /**
     * Constructs a Generator with a default Random source.
     */
    public Generator() {
        this.random = new Random();
    }

    /**
     * Constructs a Generator with a seeded Random source for deterministic results.
     * 
     * @param seed the random seed
     */
    public Generator(long seed) {
        this.random = new Random(seed);
    }

    /**
     * Autoregressively generates text starting from a prompt.
     * 
     * @param model        the language model Module
     * @param tokenizer    the tokenizer
     * @param prompt       the starting prompt text
     * @param maxNewTokens number of new tokens to generate
     * @param temperature  temperature scaling factor (<= 0.0f for greedy argmax)
     * @param topK         keep only top-K logits (<= 0 for disabled)
     * @param topP         nucleus top-P threshold (<= 0.0f or >= 1.0f for disabled)
     * @param seqLen       context length window of the model (max tokens allowed in
     *                     model input)
     * @param eosId        optional End-Of-Sequence token ID to stop early
     * @return the generated text (including the prompt)
     */
    public String generate(
            Module model,
            Tokenizer tokenizer,
            String prompt,
            int maxNewTokens,
            float temperature,
            int topK,
            float topP,
            int seqLen,
            Integer eosId) {

        model.eval(); // Set model to evaluation mode

        List<Integer> tokenIds = new ArrayList<>(tokenizer.encode(prompt));

        if (model instanceof com.tinymodelz.nn.TinyGPT) {
            com.tinymodelz.nn.TinyGPT gpt = (com.tinymodelz.nn.TinyGPT) model;
            com.tinymodelz.nn.KVCache kvCache = new com.tinymodelz.nn.KVCache(gpt.getNumLayers());

            // Step 1: Forward pass prompt tokens all at once
            float[] promptData = new float[tokenIds.size()];
            for (int i = 0; i < tokenIds.size(); i++) {
                promptData[i] = tokenIds.get(i);
            }
            Tensor promptTensor = new Tensor(promptData, new int[]{1, tokenIds.size()});
            Tensor logits = gpt.forwardWithCache(promptTensor, kvCache, 0);

            for (int step = 0; step < maxNewTokens; step++) {
                int[] logitsShape = logits.getShape();
                int vocabSize = logitsShape[2];
                int lastTokenOffset = (logitsShape[1] - 1) * vocabSize;

                float[] lastLogits = new float[vocabSize];
                Tensor contLogits = logits.toContiguous();
                System.arraycopy(contLogits.getData(), contLogits.offset() + lastTokenOffset, lastLogits, 0, vocabSize);

                applyRepetitionPenalty(lastLogits, tokenIds, 1.15f);

                int nextTokenId = sample(lastLogits, temperature, topK, topP);
                tokenIds.add(nextTokenId);

                if (eosId != null && nextTokenId == eosId) {
                    break;
                }

                // Step 2: Forward pass ONLY the single new token with cached K/V history!
                int currentPos = tokenIds.size() - 1;
                if (currentPos >= seqLen) {
                    break; // Max sequence length reached
                }
                Tensor nextTokenTensor = new Tensor(new float[]{nextTokenId}, new int[]{1, 1});
                logits = gpt.forwardWithCache(nextTokenTensor, kvCache, currentPos);
            }

            return tokenizer.decode(tokenIds);
        }

        // Fallback without KV cache
        for (int step = 0; step < maxNewTokens; step++) {
            int startIdx = Math.max(0, tokenIds.size() - seqLen);
            List<Integer> context = tokenIds.subList(startIdx, tokenIds.size());

            float[] inputData = new float[context.size()];
            for (int i = 0; i < context.size(); i++) {
                inputData[i] = context.get(i);
            }
            Tensor x = new Tensor(inputData, new int[] { 1, context.size() });

            Tensor logits = model.forward(x);
            int[] logitsShape = logits.getShape();
            int vocabSize = logitsShape[2];

            float[] lastLogits = new float[vocabSize];
            int lastTokenOffset = (context.size() - 1) * vocabSize;

            Tensor contLogits = logits.toContiguous();
            System.arraycopy(contLogits.getData(), contLogits.offset() + lastTokenOffset, lastLogits, 0, vocabSize);

            applyRepetitionPenalty(lastLogits, tokenIds, 1.15f);

            int nextTokenId = sample(lastLogits, temperature, topK, topP);
            tokenIds.add(nextTokenId);

            if (eosId != null && nextTokenId == eosId) {
                break;
            }
        }

        return tokenizer.decode(tokenIds);
    }

    private void applyRepetitionPenalty(float[] lastLogits, List<Integer> tokenIds, float repetitionPenalty) {
        if (repetitionPenalty > 1.0f) {
            int vocabSize = lastLogits.length;
            for (int previousId : tokenIds) {
                if (previousId >= 0 && previousId < vocabSize) {
                    if (lastLogits[previousId] < 0) {
                        lastLogits[previousId] *= repetitionPenalty;
                    } else {
                        lastLogits[previousId] /= repetitionPenalty;
                    }
                }
            }
        }
    }

    /**
     * Samples a token ID from a logits distribution using temperature, top-K, and
     * top-P filters.
     */
    public int sample(float[] logits, float temperature, int topK, float topP) {
        int vocabSize = logits.length;

        // 1. Greedy decoding (if temperature <= 0)
        if (temperature <= 0.0f) {
            int maxIdx = 0;
            float maxVal = -Float.MAX_VALUE;
            for (int i = 0; i < vocabSize; i++) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        // Apply Temperature
        float[] scaled = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] = logits[i] / temperature;
        }

        // Create logit entries for sorting
        List<LogitEntry> entries = new ArrayList<>(vocabSize);
        for (int i = 0; i < vocabSize; i++) {
            entries.add(new LogitEntry(i, scaled[i]));
        }

        // Sort logits in descending order
        Collections.sort(entries);

        // 2. Top-K filtering
        if (topK > 0 && topK < vocabSize) {
            for (int i = topK; i < vocabSize; i++) {
                entries.get(i).val = -Float.MAX_VALUE;
            }
        }

        // 3. Top-P (Nucleus) filtering
        if (topP > 0.0f && topP < 1.0f) {
            // Find max value in active entries for stable softmax
            float maxActiveLogit = -Float.MAX_VALUE;
            for (LogitEntry entry : entries) {
                if (entry.val > maxActiveLogit) {
                    maxActiveLogit = entry.val;
                }
            }

            // Compute sum of exponentials of valid entries
            float sumExp = 0.0f;
            for (LogitEntry entry : entries) {
                if (entry.val > -Float.MAX_VALUE) {
                    sumExp += (float) Math.exp(entry.val - maxActiveLogit);
                }
            }

            // Calculate probabilities and scan cumulative sum
            float cumSum = 0.0f;
            boolean cutoffReached = false;
            for (LogitEntry entry : entries) {
                if (entry.val == -Float.MAX_VALUE)
                    continue;

                if (cutoffReached) {
                    entry.val = -Float.MAX_VALUE;
                } else {
                    float p = (float) Math.exp(entry.val - maxActiveLogit) / sumExp;
                    cumSum += p;
                    if (cumSum >= topP) {
                        cutoffReached = true; // Cut off all remaining items
                    }
                }
            }
        }

        // 4. Softmax and sampling from the remaining distribution
        float maxFilteredLogit = -Float.MAX_VALUE;
        for (LogitEntry entry : entries) {
            if (entry.val > maxFilteredLogit) {
                maxFilteredLogit = entry.val;
            }
        }

        float sumExp = 0.0f;
        for (LogitEntry entry : entries) {
            if (entry.val > -Float.MAX_VALUE) {
                sumExp += (float) Math.exp(entry.val - maxFilteredLogit);
            }
        }

        // If all logits were filtered out, fallback to argmax of original logits
        if (sumExp == 0.0f) {
            return entries.get(0).idx;
        }

        float[] probs = new float[vocabSize];
        for (LogitEntry entry : entries) {
            if (entry.val > -Float.MAX_VALUE) {
                probs[entry.idx] = (float) Math.exp(entry.val - maxFilteredLogit) / sumExp;
            }
        }

        // Draw a random sample
        float r = random.nextFloat();
        float accumulator = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            accumulator += probs[i];
            if (r <= accumulator) {
                return i;
            }
        }

        // Fallback
        return entries.get(0).idx;
    }

    private static class LogitEntry implements Comparable<LogitEntry> {
        final int idx;
        float val;

        LogitEntry(int idx, float val) {
            this.idx = idx;
            this.val = val;
        }

        @Override
        public int compareTo(LogitEntry o) {
            // Sort in descending order
            return Float.compare(o.val, this.val);
        }
    }
}
