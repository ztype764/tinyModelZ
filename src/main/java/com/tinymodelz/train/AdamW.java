package com.tinymodelz.train;

import com.tinymodelz.math.Tensor;

import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * <h3>AdamW Optimizer</h3>
 * 
 * <p>
 * Implements the AdamW optimization algorithm (Loshchilov & Hutter, 2017),
 * which decouples L2 regularization/weight decay from the gradient update step.
 * </p>
 */
public class AdamW {

    private final Collection<Tensor> parameters;
    private float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    private int stepCount = 0;
    private final Map<Tensor, float[]> m = new IdentityHashMap<>();
    private final Map<Tensor, float[]> v = new IdentityHashMap<>();

    /**
     * Constructs an AdamW optimizer.
     * 
     * @param parameters  the collection of parameter tensors to optimize
     * @param lr          the learning rate (eta)
     * @param beta1       first moment decay rate (beta_1)
     * @param beta2       second moment decay rate (beta_2)
     * @param eps         division stability factor (epsilon)
     * @param weightDecay weight decay coefficient (lambda)
     */
    public AdamW(Collection<Tensor> parameters, float lr, float beta1, float beta2, float eps, float weightDecay) {
        if (parameters == null) {
            throw new IllegalArgumentException("Parameters collection cannot be null");
        }
        this.parameters = parameters;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    /**
     * Constructs an AdamW optimizer with default hyperparameters (beta1=0.9,
     * beta2=0.999, eps=1e-8, weightDecay=0.01).
     * 
     * @param parameters the collection of parameter tensors to optimize
     * @param lr         the learning rate
     */
    public AdamW(Collection<Tensor> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    /**
     * Performs a single optimization step, updating all parameters using their
     * accumulated gradients.
     */
    public void step() {
        stepCount++;
        float biasCorrection1 = (float) (1.0 - Math.pow(beta1, stepCount));
        float biasCorrection2 = (float) (1.0 - Math.pow(beta2, stepCount));

        for (Tensor p : parameters) {
            if (!p.requiresGrad())
                continue;
            float[] grad = p.getGrad();
            if (grad == null)
                continue;

            int size = p.size();

            // Initialize moment state buffers using computing map
            float[] mState = m.computeIfAbsent(p, k -> new float[size]);
            float[] vState = v.computeIfAbsent(p, k -> new float[size]);

            float[] data = p.getData();

            // Apply updates in parallel for contiguous parameter tensors
            if (p.isContiguous() && p.offset() == 0) {
                java.util.stream.IntStream.range(0, size).parallel().forEach(i -> {
                    float g = grad[i];
                    float w = data[i];

                    float mVal = beta1 * mState[i] + (1.0f - beta1) * g;
                    mState[i] = mVal;

                    float vVal = beta2 * vState[i] + (1.0f - beta2) * g * g;
                    vState[i] = vVal;

                    float mHat = mVal / biasCorrection1;
                    float vHat = vVal / biasCorrection2;

                    float adamStep = (lr / ((float) Math.sqrt(vHat) + eps)) * mHat;
                    float decayStep = (weightDecay != 0.0f) ? lr * weightDecay * w : 0.0f;

                    data[i] = w - adamStep - decayStep;
                });
            } else {
                for (int i = 0; i < size; i++) {
                    int idx = p.getContiguousToPhysicalOffset(i);
                    float g = grad[idx];
                    float w = data[idx];

                    float mVal = beta1 * mState[i] + (1.0f - beta1) * g;
                    mState[i] = mVal;

                    float vVal = beta2 * vState[i] + (1.0f - beta2) * g * g;
                    vState[i] = vVal;

                    float mHat = mVal / biasCorrection1;
                    float vHat = vVal / biasCorrection2;

                    float adamStep = (lr / ((float) Math.sqrt(vHat) + eps)) * mHat;
                    float decayStep = (weightDecay != 0.0f) ? lr * weightDecay * w : 0.0f;

                    data[idx] = w - adamStep - decayStep;
                }
            }
        }
    }

    /**
     * Resets the accumulated gradients of all parameters to zero.
     */
    public void zeroGrad() {
        parameters.parallelStream().forEach(p -> {
            float[] grad = p.getGrad();
            if (grad != null) {
                java.util.Arrays.fill(grad, 0.0f);
            }
        });
    }

    /**
     * Gets the current training step count.
     * 
     * @return the number of steps performed
     */
    public int getStepCount() {
        return stepCount;
    }

    /**
     * Gets the configured learning rate (eta).
     * 
     * @return the learning rate
     */
    public float getLearningRate() {
        return lr;
    }

    public void setLearningRate(float newLr) {
        this.lr = newLr;
    }

    public void setStepCount(int stepCount) {
        this.stepCount = stepCount;
    }

    public Map<Tensor, float[]> getMState() {
        return m;
    }

    public Map<Tensor, float[]> getVState() {
        return v;
    }
}
