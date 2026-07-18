package com.tinymodelz.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * <h3>Vector</h3>
 * 
 * <p>Represents a 1D mathematical vector $\mathbf{x} \in \mathbb{R}^D$ of float values.
 * This class implements fundamental vector operations required for building neural networks and transformer components.</p>
 * 
 * <h4>Mathematical Formulations:</h4>
 * <ul>
 *   <li><b>Vector Addition:</b> Let $\mathbf{a}, \mathbf{b} \in \mathbb{R}^D$.
 *       $$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_D + b_D \end{bmatrix}$$</li>
 *   <li><b>Scalar Multiplication:</b> Let $\mathbf{a} \in \mathbb{R}^D, c \in \mathbb{R}$.
 *       $$c\mathbf{a} = \begin{bmatrix} c a_1 \\ c a_2 \\ \vdots \\ c a_D \end{bmatrix}$$</li>
 *   <li><b>Dot Product:</b> Let $\mathbf{a}, \mathbf{b} \in \mathbb{R}^D$.
 *       $$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^D a_i b_i$$</li>
 *   <li><b>Euclidean ($L_2$) Norm:</b> Let $\mathbf{a} \in \mathbb{R}^D$.
 *       $$\|\mathbf{a}\|_2 = \sqrt{\sum_{i=1}^D a_i^2}$$</li>
 *   <li><b>Softmax Function:</b> Let $\mathbf{a} \in \mathbb{R}^D$.
 *       $$\text{softmax}(\mathbf{a})_i = \frac{\exp(a_i)}{\sum_{j=1}^D \exp(a_j)}$$</li>
 * </ul>
 * 
 * <h4>Complexity Analysis:</h4>
 * <ul>
 *   <li><b>Vector Addition / Subtraction:</b> $O(D)$ time complexity.</li>
 *   <li><b>Dot Product / Norm:</b> $O(D)$ time complexity.</li>
 *   <li><b>Softmax:</b> $O(D)$ time complexity (performed in a numerically stable way by subtracting the max value).</li>
 *   <li><b>Space Complexity:</b> $O(D)$ to allocate result vectors.</li>
 * </ul>
 */
public class Vector {

    private static final Logger logger = LoggerFactory.getLogger(Vector.class);

    private final float[] data;
    private final int size;

    /**
     * Constructs a Vector of the specified size initialized to zero.
     * 
     * @param size the size of the vector
     * @throws IllegalArgumentException if size is less than or equal to 0
     */
    public Vector(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }
        this.size = size;
        this.data = new float[size];
    }

    /**
     * Constructs a Vector by copying the provided data array.
     * 
     * @param data the initial float data array
     * @throws IllegalArgumentException if data is null or empty
     */
    public Vector(float[] data) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        this.size = data.length;
        this.data = new float[size];
        System.arraycopy(data, 0, this.data, 0, size);
    }

    /**
     * Gets the element at the specified index.
     * 
     * @param index the index of the element (0-indexed)
     * @return the value at the index
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    public float get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + size);
        }
        return data[index];
    }

    /**
     * Sets the element at the specified index.
     * 
     * @param index the index of the element (0-indexed)
     * @param value the new value
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    public void set(int index, float value) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + size);
        }
        data[index] = value;
    }

    /**
     * Gets the dimension size of the vector.
     * 
     * @return the vector size
     */
    public int size() {
        return size;
    }

    /**
     * Gets a copy of the underlying float array.
     * 
     * @return a copy of the float array data
     */
    public float[] getDataCopy() {
        float[] copy = new float[size];
        System.arraycopy(data, 0, copy, 0, size);
        return copy;
    }

    public float[] getData() {
        return data;
    }

    /**
     * Adds another vector to this vector.
     * 
     * @param other the other vector to add
     * @return a new Vector representing the sum
     * @throws IllegalArgumentException if the sizes do not match
     */
    public Vector add(Vector other) {
        if (other == null || other.size != this.size) {
            throw new IllegalArgumentException("Vector sizes must match for addition");
        }
        float[] resultData = new float[size];
        for (int i = 0; i < size; i++) {
            resultData[i] = this.data[i] + other.data[i];
        }
        return new Vector(resultData);
    }

    /**
     * Subtracts another vector from this vector.
     * 
     * @param other the other vector to subtract
     * @return a new Vector representing the difference
     * @throws IllegalArgumentException if the sizes do not match
     */
    public Vector subtract(Vector other) {
        if (other == null || other.size != this.size) {
            throw new IllegalArgumentException("Vector sizes must match for subtraction");
        }
        float[] resultData = new float[size];
        for (int i = 0; i < size; i++) {
            resultData[i] = this.data[i] - other.data[i];
        }
        return new Vector(resultData);
    }

    /**
     * Performs element-wise multiplication (Hadamard product) with another vector.
     * 
     * @param other the other vector
     * @return a new Vector representing the element-wise product
     * @throws IllegalArgumentException if the sizes do not match
     */
    public Vector elementwiseMultiply(Vector other) {
        if (other == null || other.size != this.size) {
            throw new IllegalArgumentException("Vector sizes must match for element-wise multiplication");
        }
        float[] resultData = new float[size];
        for (int i = 0; i < size; i++) {
            resultData[i] = this.data[i] * other.data[i];
        }
        return new Vector(resultData);
    }

    /**
     * Scales this vector by a scalar value.
     * 
     * @param scalar the scaling factor
     * @return a new Vector representing the scaled vector
     */
    public Vector scale(float scalar) {
        float[] resultData = new float[size];
        for (int i = 0; i < size; i++) {
            resultData[i] = this.data[i] * scalar;
        }
        return new Vector(resultData);
    }

    /**
     * Computes the dot product of this vector with another vector.
     * 
     * @param other the other vector
     * @return the float scalar value representing the dot product
     * @throws IllegalArgumentException if the sizes do not match
     */
    public float dot(Vector other) {
        if (other == null || other.size != this.size) {
            throw new IllegalArgumentException("Vector sizes must match for dot product");
        }
        float dotProduct = 0.0f;
        for (int i = 0; i < size; i++) {
            dotProduct += this.data[i] * other.data[i];
        }
        return dotProduct;
    }

    /**
     * Computes the Euclidean ($L_2$) norm/magnitude of the vector.
     * 
     * @return the L2 norm
     */
    public float norm() {
        float sumSq = 0.0f;
        for (float val : data) {
            sumSq += val * val;
        }
        return (float) Math.sqrt(sumSq);
    }

    /**
     * Normalizes the vector into a unit vector.
     * 
     * @return a normalized unit Vector
     * @throws ArithmeticException if the vector norm is zero
     */
    public Vector normalize() {
        float n = norm();
        if (n == 0.0f) {
            throw new ArithmeticException("Cannot normalize a vector with zero norm");
        }
        return scale(1.0f / n);
    }

    /**
     * Applies the numerically stable Softmax operation on this vector.
     * 
     * @return a new Vector representing the probability distribution
     */
    public Vector softmax() {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : data) {
            if (val > max) {
                max = val;
            }
        }

        float sumExp = 0.0f;
        float[] expData = new float[size];
        for (int i = 0; i < size; i++) {
            expData[i] = (float) Math.exp(data[i] - max);
            sumExp += expData[i];
        }

        if (sumExp == 0.0f) {
            logger.warn("Softmax denominator is zero. Defaulting to uniform distribution.");
            float uniformVal = 1.0f / size;
            float[] uniformData = new float[size];
            Arrays.fill(uniformData, uniformVal);
            return new Vector(uniformData);
        }

        for (int i = 0; i < size; i++) {
            expData[i] /= sumExp;
        }

        return new Vector(expData);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Vector)) return false;
        Vector other = (Vector) obj;
        if (this.size != other.size) return false;
        for (int i = 0; i < size; i++) {
            if (Float.compare(this.data[i], other.data[i]) != 0) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }
}
