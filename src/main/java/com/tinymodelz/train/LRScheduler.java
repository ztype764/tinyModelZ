package com.tinymodelz.train;

/**
 * <h3>LRScheduler</h3>
 *
 * <p>Implements dynamic learning rate scheduling with Linear Warmup and Cosine Decay.</p>
 *
 * <h4>Mathematical Formulation:</h4>
 * <ul>
 *   <li><b>Warmup Phase ($t < T_{warmup}$):</b> $\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}$</li>
 *   <li><b>Cosine Phase ($t \ge T_{warmup}$):</b> $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{max} - T_{warmup}} \pi\right)\right)$</li>
 * </ul>
 */
public class LRScheduler {

    private final AdamW optimizer;
    private final float maxLr;
    private final float minLr;
    private final int warmupSteps;
    private final int totalSteps;
    private int currentStep = 0;

    /**
     * Constructs a Learning Rate Scheduler with Warmup and Cosine Decay.
     *
     * @param optimizer   target AdamW optimizer
     * @param maxLr       peak learning rate
     * @param minLr       minimum (floor) learning rate
     * @param warmupSteps number of linear warmup steps
     * @param totalSteps  total training steps across all epochs
     */
    public LRScheduler(AdamW optimizer, float maxLr, float minLr, int warmupSteps, int totalSteps) {
        if (optimizer == null) {
            throw new IllegalArgumentException("Optimizer cannot be null.");
        }
        this.optimizer = optimizer;
        this.maxLr = maxLr;
        this.minLr = minLr;
        this.warmupSteps = Math.max(0, warmupSteps);
        this.totalSteps = Math.max(1, totalSteps);
    }

    /**
     * Advances the scheduler by one step, calculates the current learning rate, and updates the optimizer.
     *
     * @return the newly calculated learning rate
     */
    public float step() {
        currentStep++;
        float lr;

        if (warmupSteps > 0 && currentStep <= warmupSteps) {
            // Linear Warmup
            lr = maxLr * ((float) currentStep / warmupSteps);
        } else {
            // Cosine Decay
            int decaySteps = Math.max(1, totalSteps - warmupSteps);
            int currentDecayStep = Math.min(decaySteps, currentStep - warmupSteps);
            float progress = (float) currentDecayStep / decaySteps;
            lr = minLr + 0.5f * (maxLr - minLr) * (1.0f + (float) Math.cos(progress * Math.PI));
        }

        optimizer.setLearningRate(lr);
        return lr;
    }

    public float getLearningRate() {
        return optimizer.getLearningRate();
    }

    public int getCurrentStep() {
        return currentStep;
    }

    public void setCurrentStep(int step) {
        this.currentStep = Math.max(0, step);
    }
}
