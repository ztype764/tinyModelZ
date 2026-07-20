package com.tinymodelz.math;

import java.util.Arrays;

/**
 * <h3>Matrix</h3>
 * 
 * <p>Represents a 2D mathematical matrix $\mathbf{A} \in \mathbb{R}^{R \times C}$ of float values.
 * This class implements fundamental matrix operations required for layer transformations and self-attention operations.</p>
 * 
 * <h4>Mathematical Formulations:</h4>
 * <ul>
 *   <li><b>Matrix Addition:</b> Let $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{R \times C}$.
 *       $$\mathbf{A} + \mathbf{B} = \left[ A_{ij} + B_{ij} \right]$$</li>
 *   <li><b>Matrix Transposition:</b> Let $\mathbf{A} \in \mathbb{R}^{R \times C}$.
 *       $$\mathbf{A}^T \in \mathbb{R}^{C \times R} \quad \text{where} \quad (A^T)_{ji} = A_{ij}$$</li>
 *   <li><b>Matrix Multiplication (Dot Product):</b> Let $\mathbf{A} \in \mathbb{R}^{M \times K}, \mathbf{B} \in \mathbb{R}^{K \times N}$.
 *       $$\mathbf{C} = \mathbf{A} \mathbf{B} \in \mathbb{R}^{M \times N} \quad \text{where} \quad C_{ij} = \sum_{k=1}^K A_{ik} B_{kj}$$</li>
 *   <li><b>Matrix-Vector Product:</b> Let $\mathbf{A} \in \mathbb{R}^{R \times C}, \mathbf{x} \in \mathbb{R}^C$.
 *       $$\mathbf{y} = \mathbf{A} \mathbf{x} \in \mathbb{R}^R \quad \text{where} \quad y_i = \sum_{j=1}^C A_{ij} x_j$$</li>
 *   <li><b>Row-wise Bias Vector Addition (Broadcasting):</b> Let $\mathbf{A} \in \mathbb{R}^{R \times C}, \mathbf{v} \in \mathbb{R}^C$.
 *       $$\mathbf{C} \in \mathbb{R}^{R \times C} \quad \text{where} \quad C_{ij} = A_{ij} + v_j$$</li>
 * </ul>
 * 
 * <h4>Complexity Analysis:</h4>
 * <ul>
 *   <li><b>Matrix Addition / Subtraction:</b> $O(R \cdot C)$ time complexity.</li>
 *   <li><b>Matrix Transposition:</b> $O(R \cdot C)$ time complexity.</li>
 *   <li><b>Matrix Multiplication:</b> $O(M \cdot N \cdot K)$ time complexity.</li>
 *   <li><b>Matrix-Vector Product:</b> $O(R \cdot C)$ time complexity.</li>
 *   <li><b>Space Complexity:</b> $O(R \cdot C)$ to allocate result matrices.</li>
 * </ul>
 */
public class Matrix {

    private final float[][] data;
    private final int rows;
    private final int cols;

    /**
     * Constructs a Matrix of specified dimensions initialized to zero.
     * 
     * @param rows the number of rows
     * @param cols the number of columns
     * @throws IllegalArgumentException if rows or cols are less than or equal to 0
     */
    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Matrix dimensions must be positive");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];
    }

    /**
     * Constructs a Matrix by copying the provided 2D data array.
     * 
     * @param data the 2D float data array
     * @throws IllegalArgumentException if data is null, empty, or non-rectangular
     */
    public Matrix(float[][] data) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) {
                throw new IllegalArgumentException("Input matrix data must be rectangular");
            }
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    /**
     * Gets the element at row $i$ and column $j$.
     * 
     * @param row the row index (0-indexed)
     * @param col the column index (0-indexed)
     * @return the value at $(row, col)$
     * @throws IndexOutOfBoundsException if indices are out of bounds
     */
    public float get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Matrix index out of bounds: (" + row + ", " + col + ")");
        }
        return data[row][col];
    }

    /**
     * Sets the element at row $i$ and column $j$.
     * 
     * @param row the row index (0-indexed)
     * @param col the column index (0-indexed)
     * @param value the new value
     * @throws IndexOutOfBoundsException if indices are out of bounds
     */
    public void set(int row, int col, float value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Matrix index out of bounds: (" + row + ", " + col + ")");
        }
        data[row][col] = value;
    }

    /**
     * Gets the number of rows.
     * 
     * @return the rows count
     */
    public int rows() {
        return rows;
    }

    /**
     * Gets the number of columns.
     * 
     * @return the columns count
     */
    public int cols() {
        return cols;
    }

    /**
     * Gets a copy of the underlying 2D array.
     * 
     * @return copy of the matrix values
     */
    public float[][] getDataCopy() {
        float[][] copy = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, cols);
        }
        return copy;
    }

    public float[][] getData() {
        return data;
    }

    /**
     * Extracts a single row as a {@link Vector}.
     * 
     * @param rowIndex the index of the row to retrieve
     * @return the row as a Vector
     * @throws IndexOutOfBoundsException if rowIndex is out of bounds
     */
    public Vector getRow(int rowIndex) {
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IndexOutOfBoundsException("Row index out of bounds: " + rowIndex);
        }
        return new Vector(data[rowIndex]);
    }

    /**
     * Sets a row using values from a {@link Vector}.
     * 
     * @param rowIndex the index of the row to set
     * @param vector the vector values
     * @throws IndexOutOfBoundsException if rowIndex is out of bounds
     * @throws IllegalArgumentException if vector size doesn't match columns count
     */
    public void setRow(int rowIndex, Vector vector) {
        if (rowIndex < 0 || rowIndex >= rows) {
            throw new IndexOutOfBoundsException("Row index out of bounds: " + rowIndex);
        }
        if (vector == null || vector.size() != cols) {
            throw new IllegalArgumentException("Vector size must match column count: " + cols);
        }
        System.arraycopy(vector.getDataCopy(), 0, data[rowIndex], 0, cols);
    }

    /**
     * Adds another matrix to this matrix.
     * 
     * @param other the other matrix
     * @return a new Matrix representing the sum
     * @throws IllegalArgumentException if matrix shapes do not match
     */
    public Matrix add(Matrix other) {
        if (other == null || other.rows != this.rows || other.cols != this.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition");
        }
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return new Matrix(result);
    }

    /**
     * Subtracts another matrix from this matrix.
     * 
     * @param other the other matrix
     * @return a new Matrix representing the difference
     * @throws IllegalArgumentException if matrix shapes do not match
     */
    public Matrix subtract(Matrix other) {
        if (other == null || other.rows != this.rows || other.cols != this.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for subtraction");
        }
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return new Matrix(result);
    }

    /**
     * Multiplies this matrix with another matrix (matrix dot product).
     * 
     * @param other the other matrix
     * @return a new Matrix representing the product
     * @throws IllegalArgumentException if inner dimensions do not match (i.e. this.cols != other.rows)
     */
    public Matrix multiply(Matrix other) {
        if (other == null || this.cols != other.rows) {
            throw new IllegalArgumentException("Inner matrix dimensions must match for multiplication: " +
                    "this.cols=" + this.cols + ", other.rows=" + (other != null ? other.rows : "null"));
        }

        // --- GPU Hardware Compute Dispatch ---
        long totalFlops = (long) this.rows * this.cols * other.cols;
        if (DeviceManager.isGpuActive() && totalFlops >= 200_000L) {
            int m = this.rows;
            int k = this.cols;
            int n = other.cols;

            float[] flatA = new float[m * k];
            for (int i = 0; i < m; i++) {
                System.arraycopy(this.data[i], 0, flatA, i * k, k);
            }

            float[] flatB = new float[k * n];
            for (int i = 0; i < k; i++) {
                System.arraycopy(other.data[i], 0, flatB, i * n, n);
            }

            float[] flatC = new float[m * n];
            boolean success = false;
            if (DeviceManager.getDevice() == Device.GPU_CUDA && com.tinymodelz.gpu.CUDAMathEngine.isAvailable()) {
                success = com.tinymodelz.gpu.CUDAMathEngine.matmul(flatA, flatB, flatC, m, n, k);
            } else {
                success = com.tinymodelz.gpu.GPUMathEngine.matmul(flatA, flatB, flatC, m, n, k);
            }
            if (success) {
                float[][] result = new float[m][n];
                for (int i = 0; i < m; i++) {
                    System.arraycopy(flatC, i * n, result[i], 0, n);
                }
                return new Matrix(result);
            }
        }
        
        // --- CPU Fallback Execution ---
        float[][] result = new float[this.rows][other.cols];
        float[][] otherT = other.transpose().data; // Transposing other speeds up cache locality
        
        for (int i = 0; i < this.rows; i++) {
            float[] thisRow = this.data[i];
            for (int j = 0; j < other.cols; j++) {
                float[] otherCol = otherT[j];
                float sum = 0.0f;
                for (int k = 0; k < this.cols; k++) {
                    sum += thisRow[k] * otherCol[k];
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result);
    }

    /**
     * Multiplies this matrix with a vector (Matrix-Vector multiplication).
     * 
     * @param vector the vector
     * @return a new Vector representing the product
     * @throws IllegalArgumentException if the vector size does not match columns count
     */
    public Vector multiply(Vector vector) {
        if (vector == null || vector.size() != this.cols) {
            throw new IllegalArgumentException("Vector size (" + (vector != null ? vector.size() : "null") +
                    ") must match matrix columns (" + this.cols + ")");
        }
        float[] result = new float[rows];
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            float[] rowData = data[i];
            for (int j = 0; j < cols; j++) {
                sum += rowData[j] * vector.get(j);
            }
            result[i] = sum;
        }
        return new Vector(result);
    }

    /**
     * Transposes the matrix.
     * 
     * @return a new Matrix representing the transpose
     */
    public Matrix transpose() {
        float[][] result = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = this.data[i][j];
            }
        }
        return new Matrix(result);
    }

    /**
     * Scales the matrix by a scalar factor.
     * 
     * @param scalar the scaling factor
     * @return a new Matrix representing the scaled matrix
     */
    public Matrix scale(float scalar) {
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = this.data[i][j] * scalar;
            }
        }
        return new Matrix(result);
    }

    /**
     * Performs element-wise multiplication (Hadamard product) with another matrix.
     * 
     * @param other the other matrix
     * @return a new Matrix representing the element-wise product
     * @throws IllegalArgumentException if matrix shapes do not match
     */
    public Matrix elementwiseMultiply(Matrix other) {
        if (other == null || other.rows != this.rows || other.cols != this.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for element-wise multiplication");
        }
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return new Matrix(result);
    }

    /**
     * Adds a bias vector to each row of the matrix (broadcasting).
     * 
     * @param bias the bias vector to add
     * @return a new Matrix containing the sum
     * @throws IllegalArgumentException if bias size does not match column count
     */
    public Matrix addBiasRowwise(Vector bias) {
        if (bias == null || bias.size() != this.cols) {
            throw new IllegalArgumentException("Bias size must match matrix columns: " + this.cols);
        }
        float[][] result = new float[rows][cols];
        float[] biasData = bias.getDataCopy();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = this.data[i][j] + biasData[j];
            }
        }
        return new Matrix(result);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Matrix)) return false;
        Matrix other = (Matrix) obj;
        if (this.rows != other.rows || this.cols != other.cols) return false;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Float.compare(this.data[i][j], other.data[i][j]) != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + rows;
        result = 31 * result + cols;
        for (int i = 0; i < rows; i++) {
            result = 31 * result + Arrays.hashCode(data[i]);
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < rows; i++) {
            sb.append(Arrays.toString(data[i]));
            if (i < rows - 1) {
                sb.append(",\n ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
