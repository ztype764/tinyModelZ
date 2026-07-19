package com.tinymodelz.train;

import com.tinymodelz.math.Tensor;

import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * <h3>AdamW Optimizer</h3>
 * 
 * <p>Implements the AdamW optimization algorithm (Loshchilov & Hutter, 2017),
 * which decouples L2 regularization/weight decay from the gradient update step.</p>
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
     * @param parameters the collection of parameter tensors to optimize
     * @param lr the learning rate (eta)
     * @param beta1 first moment decay rate (beta_1)
     * @param beta2 second moment decay rate (beta_2)
     * @param eps division stability factor (epsilon)
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
     * Constructs an AdamW optimizer with default hyperparameters (beta1=0.9, beta2=0.999, eps=1e-8, weightDecay=0.01).
     * 
     * @param parameters the collection of parameter tensors to optimize
     * @param lr the learning rate
     */
    public AdamW(Collection<Tensor> parameters, float lr) {
        this(parameters, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    /**
     * Performs a single optimization step, updating all parameters using their accumulated gradients.
     */
    public void step() {
        stepCount++;
        float biasCorrection1 = (float) (1.0 - Math.pow(beta1, stepCount));
        float biasCorrection2 = (float) (1.0 - Math.pow(beta2, stepCount));

        for (Tensor p : parameters) {
            if (!p.requiresGrad()) continue;
            float[] grad = p.getGrad();
            if (grad == null) continue;

            float[] data = p.getData();
            int size = p.size();

            // Initialize moment state buffers using computing map
            float[] mState = m.computeIfAbsent(p, k -> new float[size]);
            float[] vState = v.computeIfAbsent(p, k -> new float[size]);

            // Apply updates
            for (int i = 0; i < size; i++) {
                int idx = p.getContiguousToPhysicalOffset(i);
                float g = grad[idx];
                float w = data[idx];

                // 1. Decoupled weight decay update
                if (weightDecay != 0.0f) {
                    w -= lr * weightDecay * w;
                }

                // 2. Momentum (first moment)
                mState[i] = beta1 * mState[i] + (1.0f - beta1) * g;

                // 3. RMSprop (second raw moment)
                vState[i] = beta2 * vState[i] + (1.0f - beta2) * g * g;

                // 4. Bias correction
                float mHat = mState[i] / biasCorrection1;
                float vHat = vState[i] / biasCorrection2;

                // 5. Param update step
                data[idx] = w - (lr / ((float) Math.sqrt(vHat) + eps)) * mHat;
            }
        }
    }

    /**
     * Resets the accumulated gradients of all parameters to zero.
     */
    public void zeroGrad() {
        for (Tensor p : parameters) {
            float[] grad = p.getGrad();
            if (grad != null) {
                java.util.Arrays.fill(grad, 0.0f);
            }
        }
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

    /**
     * Sets the active learning rate for dynamic scheduling.
     * 
     * @param newLr updated learning rate
     */
    public void setLearningRate(float newLr) {
        this.lr = newLr;
    }
}
