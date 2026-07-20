package com.tinymodelz.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * <h3>Tensor</h3>
 * 
 * <p>A multidimensional tensor implementation with support for arbitrary dimensions,
 * strides-based memory layout, automatic broadcasting, shape transformations,
 * and a dynamic backpropagation-based automatic differentiation engine (autograd).</p>
 * 
 * <h4>Mathematical Foundations:</h4>
 * <ul>
 *   <li><b>Strided Memory Indexing:</b> For a tensor of shape $\mathbf{s} = [s_0, s_1, \dots, s_{N-1}]$ and 
 *       strides $\mathbf{w} = [w_0, w_1, \dots, w_{N-1}]$, the absolute physical offset of a coordinate 
 *       $\mathbf{i} = [i_0, i_1, \dots, i_{N-1}]$ relative to the base offset $\text{offset}_{base}$ is:
 *       $$\text{offset}(\mathbf{i}) = \text{offset}_{base} + \sum_{d=0}^{N-1} i_d \times w_d$$</li>
 *   <li><b>Contiguous Strides:</b> Computed in row-major order:
 *       $$w_d = \prod_{j=d+1}^{N-1} s_j \quad \text{with} \quad w_{N-1} = 1$$</li>
 *   <li><b>Shape Broadcasting:</b> Shapes $\mathbf{a}$ and $\mathbf{b}$ are compatible if for each trailing dimension:
 *       $$a_d == b_d \quad \text{or} \quad a_d == 1 \quad \text{or} \quad b_d == 1$$</li>
 *   <li><b>Autograd Jacobian Backpropagation:</b>
 *       Given $Z = f(X, Y)$, backpropagation computes the gradient w.r.t inputs:
 *       $$\frac{\partial L}{\partial X} = \sum_{\text{broadcast}} \frac{\partial L}{\partial Z} \odot \frac{\partial Z}{\partial X}$$</li>
 * </ul>
 */
public class Tensor {

    private final float[] data;
    private final int[] shape;
    private final int[] strides;
    private final int size;
    private final int offset;

    // Autograd fields
    private float[] grad;
    private boolean requiresGrad;
    private List<Tensor> creators;
    private String opName;
    private BackwardFunction backwardFn;

    @FunctionalInterface
    public interface BackwardFunction {
        void apply(float[] gradOutput);
    }

    /**
     * Constructs a contiguous tensor of a specific shape initialized to zero.
     * 
     * @param shape the dimensions of the tensor
     */
    public Tensor(int... shape) {
        this.shape = shape.clone();
        this.size = computeSize(shape);
        this.data = new float[size];
        this.strides = computeStrides(shape);
        this.offset = 0;
        this.requiresGrad = false;
    }

    /**
     * Constructs a contiguous tensor from a flat array and shape.
     * 
     * @param data the float data payload
     * @param shape the dimensions of the tensor
     */
    public Tensor(float[] data, int[] shape) {
        this.shape = shape.clone();
        this.size = computeSize(shape);
        if (data.length < size) {
            throw new IllegalArgumentException("Data array length is smaller than the calculated shape size.");
        }
        this.data = data;
        this.strides = computeStrides(shape);
        this.offset = 0;
        this.requiresGrad = false;
    }

    /**
     * Constructs a tensor view with custom strides and offset.
     * 
     * @param data the shared float data payload
     * @param shape the dimensions of the tensor view
     * @param strides the custom strides mapping dimensions to physical data indices
     * @param offset the physical index offset in the data array
     */
    public Tensor(float[] data, int[] shape, int[] strides, int offset) {
        this.data = data;
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.size = computeSize(shape);
        this.offset = offset;
        this.requiresGrad = false;
    }

    // Helper constructor for contiguous views sharing data
    public Tensor(float[] data, int[] shape, int[] strides) {
        this(data, shape, strides, 0);
    }

    // Static Factories
    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, 1.0f);
        return t;
    }

    public static Tensor scalar(float val) {
        return new Tensor(new float[]{val}, new int[]{1});
    }

    // Getters and Configurations
    public float[] getData() {
        return data;
    }

    public int[] getShape() {
        return shape.clone();
    }

    public int[] getStrides() {
        return strides.clone();
    }

    public int size() {
        return size;
    }

    public int offset() {
        return offset;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public String getOpName() {
        return opName;
    }

    public List<Tensor> getCreators() {
        return creators;
    }

    public void setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        if (requiresGrad && this.grad == null) {
            this.grad = new float[this.data.length];
        }
    }

    public float[] getGrad() {
        return grad;
    }

    public void zeroGrad() {
        if (grad != null) {
            Arrays.fill(grad, 0.0f);
        }
    }

    public void accumulateGrad(float[] incomingGrad) {
        if (this.grad == null) {
            this.grad = new float[this.data.length];
        }
        if (isContiguous() && offset == 0) {
            java.util.stream.IntStream.range(0, incomingGrad.length).parallel().forEach(i -> {
                this.grad[i] += incomingGrad[i];
            });
        } else {
            for (int i = 0; i < incomingGrad.length; i++) {
                this.grad[getContiguousToPhysicalOffset(i)] += incomingGrad[i];
            }
        }
    }

    public void setAutogradMetadata(List<Tensor> creators, String opName, BackwardFunction backwardFn) {
        this.creators = creators;
        this.opName = opName;
        this.backwardFn = backwardFn;
    }

    // Stride Calculations
    public static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        if (shape.length == 0) return strides;
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    private static int computeSize(int[] shape) {
        if (shape.length == 0) return 0;
        int size = 1;
        for (int s : shape) size *= s;
        return size;
    }

    public boolean isContiguous() {
        int[] standard = computeStrides(shape);
        for (int i = 0; i < shape.length; i++) {
            if (strides[i] != standard[i]) return false;
        }
        return true;
    }

    public Tensor toContiguous() {
        if (isContiguous() && offset == 0) {
            return this;
        }
        float[] contiguousData = new float[size];
        for (int i = 0; i < size; i++) {
            contiguousData[i] = getValByFlatIndex(i);
        }
        return new Tensor(contiguousData, shape);
    }

    public int getContiguousToPhysicalOffset(int i) {
        int physicalOffset = this.offset;
        int remaining = i;
        int[] contiguousStrides = computeStrides(shape);
        for (int d = 0; d < shape.length; d++) {
            int coord = remaining / contiguousStrides[d];
            remaining %= contiguousStrides[d];
            physicalOffset += coord * strides[d];
        }
        return physicalOffset;
    }

    public float getValByFlatIndex(int i) {
        return data[getContiguousToPhysicalOffset(i)];
    }

    // Shape Transformations
    public Tensor reshape(int... newShape) {
        int newSize = computeSize(newShape);
        if (newSize != this.size) {
            throw new IllegalArgumentException("Cannot reshape shape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
        }
        
        Tensor contiguousTensor = this.toContiguous();
        Tensor result = new Tensor(contiguousTensor.data, newShape);
        
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "reshape";
            result.backwardFn = (gradOutput) -> {
                if (this.requiresGrad) {
                    if (this.isContiguous() && this.offset == 0) {
                        this.accumulateGrad(gradOutput);
                    } else {
                        if (this.grad == null) {
                            this.grad = new float[this.data.length];
                        }
                        for (int i = 0; i < size; i++) {
                            int offsetIdx = this.getContiguousToPhysicalOffset(i);
                            this.grad[offsetIdx] += gradOutput[i];
                        }
                    }
                }
            };
        }
        return result;
    }

    public Tensor transpose() {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Default transpose is only supported for 2D matrices.");
        }
        return transpose(0, 1);
    }

    public Tensor transpose(int dim1, int dim2) {
        if (dim1 < 0 || dim1 >= shape.length || dim2 < 0 || dim2 >= shape.length) {
            throw new IllegalArgumentException("Invalid transpose dimensions index.");
        }
        int[] newShape = shape.clone();
        int[] newStrides = strides.clone();
        
        // Swap shapes and strides
        int tempShape = newShape[dim1];
        newShape[dim1] = newShape[dim2];
        newShape[dim2] = tempShape;
        
        int tempStride = newStrides[dim1];
        newStrides[dim1] = newStrides[dim2];
        newStrides[dim2] = tempStride;
        
        Tensor result = new Tensor(this.data, newShape, newStrides, this.offset);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "transpose";
            result.backwardFn = (gradOutput) -> {
                Tensor gradOutView = new Tensor(gradOutput, newShape, newStrides, 0);
                Tensor backGrad = gradOutView.transpose(dim1, dim2).toContiguous();
                this.accumulateGrad(backGrad.data);
            };
        }
        return result;
    }

    public Tensor slice(int dim, int start, int end) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Invalid slice dimension: " + dim);
        }
        if (start < 0 || end > shape[dim] || start > end) {
            throw new IllegalArgumentException("Invalid slice range: [" + start + ", " + end + "]");
        }
        
        int[] newShape = shape.clone();
        newShape[dim] = end - start;
        int[] newStrides = strides.clone();
        int newOffset = this.offset + start * strides[dim];
        
        Tensor result = new Tensor(this.data, newShape, newStrides, newOffset);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "slice";
            result.backwardFn = (gradOutput) -> {
                if (this.grad == null) {
                    this.grad = new float[this.data.length];
                }
                int[] outStrides = computeStrides(newShape);
                for (int i = 0; i < gradOutput.length; i++) {
                    int remaining = i;
                    int sliceOffset = 0;
                    for (int d = 0; d < newShape.length; d++) {
                        int coord = remaining / outStrides[d];
                        remaining %= outStrides[d];
                        sliceOffset += coord * newStrides[d];
                    }
                    this.grad[newOffset + sliceOffset] += gradOutput[i];
                }
            };
        }
        return result;
    }

    // Broadcasting index projection helper
    public static int getBroadcastedFlatIndex(int flatIndex, int[] outShape, int[] srcShape, int[] srcStrides) {
        int remaining = flatIndex;
        int srcFlatIndex = 0;
        for (int d = outShape.length - 1; d >= 0; d--) {
            int outDimSize = outShape[d];
            int coord = remaining % outDimSize;
            remaining /= outDimSize;

            int srcDimIdx = d - (outShape.length - srcShape.length);
            int srcDimSize = (srcDimIdx < 0) ? 1 : srcShape[srcDimIdx];
            
            if (srcDimSize > 1) {
                int srcStride = srcStrides[srcDimIdx];
                srcFlatIndex += coord * srcStride;
            }
        }
        return srcFlatIndex;
    }

    public static int[] broadcastShapes(int[] shapeA, int[] shapeB) {
        int lenA = shapeA.length;
        int lenB = shapeB.length;
        int outLen = Math.max(lenA, lenB);
        int[] outShape = new int[outLen];
        
        for (int i = 0; i < outLen; i++) {
            int idxA = i - (outLen - lenA);
            int dimA = (idxA >= 0) ? shapeA[idxA] : 1;
            
            int idxB = i - (outLen - lenB);
            int dimB = (idxB >= 0) ? shapeB[idxB] : 1;
            
            if (dimA == dimB) {
                outShape[i] = dimA;
            } else if (dimA == 1) {
                outShape[i] = dimB;
            } else if (dimB == 1) {
                outShape[i] = dimA;
            } else {
                throw new IllegalArgumentException("Incompatible shapes for broadcasting: " 
                        + Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB));
            }
        }
        return outShape;
    }

    private void accumulateBroadcastedGrad(Tensor tensor, float[] gradOutput, int[] outShape) {
        if (tensor.grad == null) {
            tensor.grad = new float[tensor.data.length];
        }
        for (int i = 0; i < gradOutput.length; i++) {
            int idx = getBroadcastedFlatIndex(i, outShape, tensor.shape, tensor.strides);
            tensor.grad[tensor.offset + idx] += gradOutput[i];
        }
    }

    // Element-Wise Mathematical Operations with Broadcasting
    public Tensor add(Tensor other) {
        int[] outShape = broadcastShapes(this.shape, other.shape);
        int outSize = computeSize(outShape);
        float[] outData = new float[outSize];
        
        if (Arrays.equals(this.shape, other.shape) && isContiguous() && other.isContiguous()) {
            int offA = this.offset;
            int offB = other.offset;
            java.util.stream.IntStream.range(0, outSize).parallel().forEach(i -> {
                outData[i] = this.data[offA + i] + other.data[offB + i];
            });
        } else {
            for (int i = 0; i < outSize; i++) {
                int idxA = getBroadcastedFlatIndex(i, outShape, this.shape, this.strides);
                int idxB = getBroadcastedFlatIndex(i, outShape, other.shape, other.strides);
                outData[i] = this.data[this.offset + idxA] + other.data[other.offset + idxB];
            }
        }
        
        Tensor result = new Tensor(outData, outShape);
        if (this.requiresGrad || other.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this, other);
            result.opName = "add";
            result.backwardFn = (gradOutput) -> {
                if (this.requiresGrad) {
                    accumulateBroadcastedGrad(this, gradOutput, outShape);
                }
                if (other.requiresGrad) {
                    accumulateBroadcastedGrad(other, gradOutput, outShape);
                }
            };
        }
        return result;
    }

    public Tensor subtract(Tensor other) {
        int[] outShape = broadcastShapes(this.shape, other.shape);
        int outSize = computeSize(outShape);
        float[] outData = new float[outSize];
        
        if (Arrays.equals(this.shape, other.shape) && isContiguous() && other.isContiguous()) {
            int offA = this.offset;
            int offB = other.offset;
            java.util.stream.IntStream.range(0, outSize).parallel().forEach(i -> {
                outData[i] = this.data[offA + i] - other.data[offB + i];
            });
        } else {
            for (int i = 0; i < outSize; i++) {
                int idxA = getBroadcastedFlatIndex(i, outShape, this.shape, this.strides);
                int idxB = getBroadcastedFlatIndex(i, outShape, other.shape, other.strides);
                outData[i] = this.data[this.offset + idxA] - other.data[other.offset + idxB];
            }
        }
        
        Tensor result = new Tensor(outData, outShape);
        if (this.requiresGrad || other.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this, other);
            result.opName = "subtract";
            result.backwardFn = (gradOutput) -> {
                if (this.requiresGrad) {
                    accumulateBroadcastedGrad(this, gradOutput, outShape);
                }
                if (other.requiresGrad) {
                    float[] negGrad = new float[gradOutput.length];
                    for (int i = 0; i < gradOutput.length; i++) {
                        negGrad[i] = -gradOutput[i];
                    }
                    accumulateBroadcastedGrad(other, negGrad, outShape);
                }
            };
        }
        return result;
    }

    public Tensor multiply(Tensor other) {
        int[] outShape = broadcastShapes(this.shape, other.shape);
        int outSize = computeSize(outShape);
        float[] outData = new float[outSize];
        
        if (Arrays.equals(this.shape, other.shape) && isContiguous() && other.isContiguous()) {
            int offA = this.offset;
            int offB = other.offset;
            java.util.stream.IntStream.range(0, outSize).parallel().forEach(i -> {
                outData[i] = this.data[offA + i] * other.data[offB + i];
            });
        } else {
            for (int i = 0; i < outSize; i++) {
                int idxA = getBroadcastedFlatIndex(i, outShape, this.shape, this.strides);
                int idxB = getBroadcastedFlatIndex(i, outShape, other.shape, other.strides);
                outData[i] = this.data[this.offset + idxA] * other.data[other.offset + idxB];
            }
        }
        
        Tensor result = new Tensor(outData, outShape);
        if (this.requiresGrad || other.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this, other);
            result.opName = "multiply";
            result.backwardFn = (gradOutput) -> {
                if (this.requiresGrad) {
                    float[] gradA = new float[gradOutput.length];
                    for (int i = 0; i < gradOutput.length; i++) {
                        int idxB = getBroadcastedFlatIndex(i, outShape, other.shape, other.strides);
                        gradA[i] = gradOutput[i] * other.data[other.offset + idxB];
                    }
                    accumulateBroadcastedGrad(this, gradA, outShape);
                }
                if (other.requiresGrad) {
                    float[] gradB = new float[gradOutput.length];
                    for (int i = 0; i < gradOutput.length; i++) {
                        int idxA = getBroadcastedFlatIndex(i, outShape, this.shape, this.strides);
                        gradB[i] = gradOutput[i] * this.data[this.offset + idxA];
                    }
                    accumulateBroadcastedGrad(other, gradB, outShape);
                }
            };
        }
        return result;
    }

    // Matrix Multiplication (supports 2D and N-dimensional batch matrix multiplication)
    public Tensor matmul(Tensor other) {
        if (this.shape.length < 2 || other.shape.length < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires tensors of rank >= 2.");
        }
        int rank = this.shape.length;
        if (rank != other.shape.length) {
            throw new IllegalArgumentException("Rank mismatch for matrix multiplication: " + rank + " vs " + other.shape.length);
        }
        
        int M = this.shape[rank - 2];
        int K = this.shape[rank - 1];
        int K2 = other.shape[rank - 2];
        int N = other.shape[rank - 1];
        
        if (K != K2) {
            throw new IllegalArgumentException("Incompatible shapes for matmul: " + K + " and " + K2);
        }
        
        // Batch dimensions check
        int[] batchShape = new int[rank - 2];
        for (int i = 0; i < rank - 2; i++) {
            if (this.shape[i] != other.shape[i]) {
                throw new IllegalArgumentException("Batch dimensions must match: " + Arrays.toString(this.shape) + " vs " + Arrays.toString(other.shape));
            }
            batchShape[i] = this.shape[i];
        }
        
        int[] outShape = new int[rank];
        System.arraycopy(batchShape, 0, outShape, 0, rank - 2);
        outShape[rank - 2] = M;
        outShape[rank - 1] = N;
        
        int outSize = computeSize(outShape);
        float[] outData = new float[outSize];
        
        int numBatches = 1;
        for (int s : batchShape) {
            numBatches *= s;
        }
        
        Tensor ACont = this.toContiguous();
        Tensor BCont = other.toContiguous();
        
        // --- GPU Acceleration Dispatch ---
        long totalFlops = (long) numBatches * M * K * N;
        if (DeviceManager.isGpuActive() && totalFlops >= 200_000L) {
            boolean success = false;
            if (DeviceManager.getDevice() == Device.GPU_CUDA && com.tinymodelz.gpu.CUDAMathEngine.isAvailable()) {
                success = com.tinymodelz.gpu.CUDAMathEngine.batchedMatmul(ACont.data, BCont.data, outData, numBatches, M, N, K);
            } else {
                success = com.tinymodelz.gpu.GPUMathEngine.batchedMatmul(ACont.data, BCont.data, outData, numBatches, M, N, K);
            }
            if (success) {
                Tensor result = new Tensor(outData, outShape);
                if (this.requiresGrad || other.requiresGrad) {
                    result.requiresGrad = true;
                    result.creators = List.of(this, other);
                    result.opName = "matmul";
                    result.backwardFn = (gradOutput) -> {
                        Tensor gradOutTensor = new Tensor(gradOutput, outShape);
                        if (this.requiresGrad) {
                            Tensor dX = gradOutTensor.matmul(other.transpose(rank - 2, rank - 1));
                            this.accumulateGrad(dX.data);
                        }
                        if (other.requiresGrad) {
                            Tensor dY = this.transpose(rank - 2, rank - 1).matmul(gradOutTensor);
                            other.accumulateGrad(dY.data);
                        }
                    };
                }
                return result;
            }
        }

        // --- CPU Execution Path ---
        int totalRows = numBatches * M;
        java.util.stream.IntStream.range(0, totalRows).parallel().forEach(bi -> {
            int b = bi / M;
            int i = bi % M;

            int offsetA = b * M * K + i * K;
            int offsetB = b * K * N;
            int offsetOut = b * M * N + i * N;

            for (int k = 0; k < K; k++) {
                float aVal = ACont.data[offsetA + k];
                if (aVal == 0.0f) continue;
                int bRowOffset = offsetB + k * N;
                for (int j = 0; j < N; j++) {
                    outData[offsetOut + j] += aVal * BCont.data[bRowOffset + j];
                }
            }
        });
        
        Tensor result = new Tensor(outData, outShape);
        if (this.requiresGrad || other.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this, other);
            result.opName = "matmul";
            result.backwardFn = (gradOutput) -> {
                Tensor gradOutTensor = new Tensor(gradOutput, outShape);
                if (this.requiresGrad) {
                    Tensor dX = gradOutTensor.matmul(other.transpose(rank - 2, rank - 1));
                    this.accumulateGrad(dX.data);
                }
                if (other.requiresGrad) {
                    Tensor dY = this.transpose(rank - 2, rank - 1).matmul(gradOutTensor);
                    other.accumulateGrad(dY.data);
                }
            };
        }
        return result;
    }

    public Tensor softmax() {
        return softmax(shape.length - 1);
    }

    public Tensor softmax(int dim) {
        if (dim != shape.length - 1) {
            throw new UnsupportedOperationException("Softmax is currently optimized and supported only for the last dimension.");
        }
        Tensor contiguous = this.toContiguous();
        float[] outData = new float[size];
        int D = shape[dim];
        int numRows = size / D;
        
        for (int r = 0; r < numRows; r++) {
            int offset = r * D;
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < D; c++) {
                maxVal = Math.max(maxVal, contiguous.data[offset + c]);
            }
            float sumExps = 0.0f;
            for (int c = 0; c < D; c++) {
                float expVal = (float) Math.exp(contiguous.data[offset + c] - maxVal);
                outData[offset + c] = expVal;
                sumExps += expVal;
            }
            for (int c = 0; c < D; c++) {
                outData[offset + c] /= sumExps;
            }
        }
        
        Tensor result = new Tensor(outData, shape);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "softmax";
            result.backwardFn = (gradOutput) -> {
                if (this.grad == null) {
                    this.grad = new float[this.data.length];
                }
                for (int r = 0; r < numRows; r++) {
                    int offset = r * D;
                    float sumDYtimesY = 0.0f;
                    for (int c = 0; c < D; c++) {
                        sumDYtimesY += gradOutput[offset + c] * result.data[offset + c];
                    }
                    for (int c = 0; c < D; c++) {
                        float dy = gradOutput[offset + c];
                        float y = result.data[offset + c];
                        float dxVal = y * (dy - sumDYtimesY);
                        int physIdx = this.getContiguousToPhysicalOffset(offset + c);
                        this.grad[physIdx] += dxVal;
                    }
                }
            };
        }
        return result;
    }

    public Tensor maskedFill(Tensor mask, float value) {
        int[] outShape = broadcastShapes(this.shape, mask.shape);
        int outSize = computeSize(outShape);
        float[] outData = new float[outSize];
        
        for (int i = 0; i < outSize; i++) {
            int idxA = getBroadcastedFlatIndex(i, outShape, this.shape, this.strides);
            int idxM = getBroadcastedFlatIndex(i, outShape, mask.shape, mask.strides);
            if (mask.data[mask.offset + idxM] != 0.0f) {
                outData[i] = value;
            } else {
                outData[i] = this.data[this.offset + idxA];
            }
        }
        
        Tensor result = new Tensor(outData, outShape);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "maskedFill";
            result.backwardFn = (gradOutput) -> {
                if (this.grad == null) {
                    this.grad = new float[this.data.length];
                }
                for (int i = 0; i < gradOutput.length; i++) {
                    int idxM = getBroadcastedFlatIndex(i, outShape, mask.shape, mask.strides);
                    if (mask.data[mask.offset + idxM] == 0.0f) {
                        int idxA = getBroadcastedFlatIndex(i, outShape, this.shape, this.strides);
                        this.grad[this.offset + idxA] += gradOutput[i];
                    }
                }
            };
        }
        return result;
    }

    public Tensor add(float val) {
        return this.add(Tensor.scalar(val));
    }

    public Tensor multiply(float val) {
        return this.multiply(Tensor.scalar(val));
    }


    // Reduction Operations
    public Tensor sum() {
        float sumVal = 0.0f;
        for (int i = 0; i < size; i++) {
            sumVal += getValByFlatIndex(i);
        }
        
        Tensor result = new Tensor(new float[]{sumVal}, new int[]{1});
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "sum";
            result.backwardFn = (gradOutput) -> {
                if (this.requiresGrad) {
                    if (this.grad == null) {
                        this.grad = new float[this.data.length];
                    }
                    for (int i = 0; i < size; i++) {
                        int offsetIdx = getContiguousToPhysicalOffset(i);
                        this.grad[offsetIdx] += gradOutput[0];
                    }
                }
            };
        }
        return result;
    }

    // Activations
    public Tensor relu() {
        float[] outData = new float[size];
        for (int i = 0; i < size; i++) {
            float val = getValByFlatIndex(i);
            outData[i] = Math.max(0.0f, val);
        }
        Tensor result = new Tensor(outData, shape);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "relu";
            result.backwardFn = (gradOutput) -> {
                if (this.grad == null) {
                    this.grad = new float[this.data.length];
                }
                for (int i = 0; i < size; i++) {
                    float val = getValByFlatIndex(i);
                    if (val > 0.0f) {
                        int offsetIdx = getContiguousToPhysicalOffset(i);
                        this.grad[offsetIdx] += gradOutput[i];
                    }
                }
            };
        }
        return result;
    }

    public Tensor sigmoid() {
        float[] outData = new float[size];
        for (int i = 0; i < size; i++) {
            float val = getValByFlatIndex(i);
            outData[i] = (float) (1.0 / (1.0 + Math.exp(-val)));
        }
        Tensor result = new Tensor(outData, shape);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "sigmoid";
            result.backwardFn = (gradOutput) -> {
                if (this.grad == null) {
                    this.grad = new float[this.data.length];
                }
                for (int i = 0; i < size; i++) {
                    float sig = result.data[i];
                    int offsetIdx = getContiguousToPhysicalOffset(i);
                    this.grad[offsetIdx] += gradOutput[i] * sig * (1.0f - sig);
                }
            };
        }
        return result;
    }

    public Tensor gelu() {
        float[] outData = new float[size];
        if (isContiguous() && offset == 0) {
            java.util.stream.IntStream.range(0, size).parallel().forEach(i -> {
                float val = data[i];
                double v = val + 0.044715 * val * val * val;
                double u = 0.7978845608 * v;
                outData[i] = (float) (0.5 * val * (1.0 + Math.tanh(u)));
            });
        } else {
            for (int i = 0; i < size; i++) {
                float val = getValByFlatIndex(i);
                double v = val + 0.044715 * val * val * val;
                double u = 0.7978845608 * v;
                outData[i] = (float) (0.5 * val * (1.0 + Math.tanh(u)));
            }
        }
        Tensor result = new Tensor(outData, shape);
        if (this.requiresGrad) {
            result.requiresGrad = true;
            result.creators = List.of(this);
            result.opName = "gelu";
            result.backwardFn = (gradOutput) -> {
                if (this.grad == null) {
                    this.grad = new float[this.data.length];
                }
                if (isContiguous() && offset == 0) {
                    java.util.stream.IntStream.range(0, size).parallel().forEach(i -> {
                        float val = data[i];
                        double v = val + 0.044715 * val * val * val;
                        double u = 0.7978845608 * v;
                        double tanhVal = Math.tanh(u);
                        double sech2 = 1.0 - tanhVal * tanhVal;
                        double dGelu = 0.5 * (1.0 + tanhVal) + 0.5 * val * sech2 * 0.7978845608 * (1.0 + 0.134145 * val * val);
                        this.grad[i] += gradOutput[i] * (float) dGelu;
                    });
                } else {
                    for (int i = 0; i < size; i++) {
                        float val = getValByFlatIndex(i);
                        double v = val + 0.044715 * val * val * val;
                        double u = 0.7978845608 * v;
                        double tanhVal = Math.tanh(u);
                        double sech2 = 1.0 - tanhVal * tanhVal;
                        double dGelu = 0.5 * (1.0 + tanhVal) + 0.5 * val * sech2 * 0.7978845608 * (1.0 + 0.134145 * val * val);
                        int offsetIdx = getContiguousToPhysicalOffset(i);
                        this.grad[offsetIdx] += gradOutput[i] * (float) dGelu;
                    }
                }
            };
        }
        return result;
    }

    // Convert to standard formats
    public float[][] to2DArray() {
        if (shape.length != 2) {
            throw new IllegalStateException("Tensor must be 2D to convert to a 2D float array.");
        }
        int rows = shape[0];
        int cols = shape[1];
        float[][] res = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                res[r][c] = data[offset + r * strides[0] + c * strides[1]];
            }
        }
        return res;
    }

    public Matrix toMatrix() {
        if (shape.length != 2) {
            throw new IllegalStateException("Tensor must be 2D to convert to Matrix.");
        }
        Tensor cont = this.toContiguous();
        float[][] matData = new float[shape[0]][shape[1]];
        for (int r = 0; r < shape[0]; r++) {
            System.arraycopy(cont.data, r * shape[1], matData[r], 0, shape[1]);
        }
        return new Matrix(matData);
    }

    public Vector toVector() {
        if (shape.length != 1) {
            throw new IllegalStateException("Tensor must be 1D to convert to Vector.");
        }
        Tensor cont = this.toContiguous();
        return new Vector(cont.data.clone());
    }

    public static Tensor fromMatrix(Matrix matrix) {
        float[][] mData = matrix.getDataCopy();
        int rows = mData.length;
        int cols = mData[0].length;
        float[] data = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(mData[r], 0, data, r * cols, cols);
        }
        return new Tensor(data, new int[]{rows, cols});
    }

    public static Tensor fromVector(Vector vector) {
        float[] data = vector.getDataCopy();
        return new Tensor(data, new int[]{data.length});
    }

    // Autograd Topological Execution Engine
    public void backward() {
        if (!requiresGrad) {
            throw new IllegalStateException("Cannot call backward on a tensor that does not require gradients.");
        }
        if (grad == null) {
            grad = new float[data.length];
        }
        // Initialize output gradient to 1.0 (for scalar loss output)
        Arrays.fill(grad, 1.0f);
        
        List<Tensor> order = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopologicalOrder(this, order, visited);
        
        for (int i = order.size() - 1; i >= 0; i--) {
            Tensor node = order.get(i);
            if (node.backwardFn != null && node.grad != null) {
                float[] nodeGradSlice = new float[node.size];
                System.arraycopy(node.grad, node.offset, nodeGradSlice, 0, node.size);
                node.backwardFn.apply(nodeGradSlice);
            }
        }
    }

    private void buildTopologicalOrder(Tensor node, List<Tensor> order, Set<Tensor> visited) {
        if (visited.contains(node)) return;
        visited.add(node);
        if (node.creators != null) {
            for (Tensor parent : node.creators) {
                buildTopologicalOrder(parent, order, visited);
            }
        }
        order.add(node);
    }



    @Override
    public String toString() {
        return "Tensor{shape=" + Arrays.toString(shape) + ", size=" + size + ", contiguous=" + isContiguous() + "}";
    }
}
