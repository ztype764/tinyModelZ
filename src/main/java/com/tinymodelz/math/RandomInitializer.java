package com.tinymodelz.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * <h3>RandomInitializer</h3>
 * 
 * <p>Handles weight and parameter initialization for vectors and matrices using standard distributions.
 * This class implements standard initializers used to prevent vanishing/exploding gradients during training.</p>
 * 
 * <h4>Mathematical Formulations:</h4>
 * <ul>
 *   <li><b>Uniform Distribution:</b> Let $X \sim U(a, b)$.
 *       $$P(x) = \frac{1}{b - a} \quad \text{for } x \in [a, b]$$</li>
 *   <li><b>Normal (Gaussian) Distribution:</b> Let $X \sim N(\mu, \sigma^2)$.
 *       $$P(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$</li>
 *   <li><b>Xavier (Glorot) Uniform Initialization:</b> Let $d_{in}$ be input dimension, $d_{out}$ be output dimension.
 *       $$W_{ij} \sim U(-r, r) \quad \text{where} \quad r = \sqrt{\frac{6}{d_{in} + d_{out}}}$$</li>
 *   <li><b>Xavier (Glorot) Normal Initialization:</b>
 *       $$W_{ij} \sim N\left(0, \sigma^2\right) \quad \text{where} \quad \sigma^2 = \frac{2}{d_{in} + d_{out}}$$</li>
 *   <li><b>He (Kaiming) Normal Initialization:</b>
 *       $$W_{ij} \sim N\left(0, \sigma^2\right) \quad \text{where} \quad \sigma^2 = \frac{2}{d_{in}}$$</li>
 * </ul>
 */
public class RandomInitializer {

    private static final Logger logger = LoggerFactory.getLogger(RandomInitializer.class);

    private final Random random;

    /**
     * Constructs a RandomInitializer with a default seed.
     */
    public RandomInitializer() {
        this.random = new Random();
    }

    /**
     * Constructs a RandomInitializer with a specific seed for reproducibility.
     * 
     * @param seed the seed value
     */
    public RandomInitializer(long seed) {
        this.random = new Random(seed);
        logger.info("RandomInitializer seeded with: {}", seed);
    }

    /**
     * Fills a vector with values from a uniform distribution $U(a, b)$.
     * 
     * @param vector the vector to fill
     * @param min the lower bound $a$
     * @param max the upper bound $b$
     */
    public void fillUniform(Vector vector, float min, float max) {
        if (min >= max) {
            throw new IllegalArgumentException("Min boundary must be less than max boundary");
        }
        for (int i = 0; i < vector.size(); i++) {
            float val = min + random.nextFloat() * (max - min);
            vector.set(i, val);
        }
    }

    /**
     * Fills a matrix with values from a uniform distribution $U(a, b)$.
     * 
     * @param matrix the matrix to fill
     * @param min the lower bound $a$
     * @param max the upper bound $b$
     */
    public void fillUniform(Matrix matrix, float min, float max) {
        if (min >= max) {
            throw new IllegalArgumentException("Min boundary must be less than max boundary");
        }
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                float val = min + random.nextFloat() * (max - min);
                matrix.set(i, j, val);
            }
        }
    }

    /**
     * Fills a vector with values from a normal distribution $N(\mu, \sigma^2)$.
     * 
     * @param vector the vector to fill
     * @param mean the mean $\mu$
     * @param stdDev the standard deviation $\sigma$
     */
    public void fillNormal(Vector vector, float mean, float stdDev) {
        for (int i = 0; i < vector.size(); i++) {
            float val = mean + (float) random.nextGaussian() * stdDev;
            vector.set(i, val);
        }
    }

    /**
     * Fills a matrix with values from a normal distribution $N(\mu, \sigma^2)$.
     * 
     * @param matrix the matrix to fill
     * @param mean the mean $\mu$
     * @param stdDev the standard deviation $\sigma$
     */
    public void fillNormal(Matrix matrix, float mean, float stdDev) {
        for (int i = 0; i < matrix.rows(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                float val = mean + (float) random.nextGaussian() * stdDev;
                matrix.set(i, j, val);
            }
        }
    }

    /**
     * Fills a weight matrix using Xavier (Glorot) Uniform initialization.
     * 
     * @param matrix the matrix to fill
     * @param dIn the input dimension size
     * @param dOut the output dimension size
     */
    public void fillXavierUniform(Matrix matrix, int dIn, int dOut) {
        float r = (float) Math.sqrt(6.0 / (dIn + dOut));
        fillUniform(matrix, -r, r);
    }

    /**
     * Fills a weight matrix using Xavier (Glorot) Normal initialization.
     * 
     * @param matrix the matrix to fill
     * @param dIn the input dimension size
     * @param dOut the output dimension size
     */
    public void fillXavierNormal(Matrix matrix, int dIn, int dOut) {
        float stdDev = (float) Math.sqrt(2.0 / (dIn + dOut));
        fillNormal(matrix, 0.0f, stdDev);
    }

    /**
     * Fills a weight matrix using He (Kaiming) Normal initialization.
     * 
     * @param matrix the matrix to fill
     * @param dIn the input dimension size
     */
    public void fillHeNormal(Matrix matrix, int dIn) {
        float stdDev = (float) Math.sqrt(2.0 / dIn);
        fillNormal(matrix, 0.0f, stdDev);
    }
}
