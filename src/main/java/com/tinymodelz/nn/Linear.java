package com.tinymodelz.nn;

import com.tinymodelz.math.Matrix;
import com.tinymodelz.math.RandomInitializer;
import com.tinymodelz.math.Tensor;

/**
 * <h3>Linear</h3>
 * 
 * <p>A fully connected (dense) neural network layer.</p>
 * <p>Computes the linear transformation: $\mathbf{Y} = \mathbf{X}\mathbf{W}^T + \mathbf{b}$</p>
 */
public class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;
    private final boolean useBias;

    /**
     * Constructs a Linear layer with bias enabled.
     * 
     * @param inFeatures number of input features
     * @param outFeatures number of output features
     */
    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    /**
     * Constructs a Linear layer.
     * 
     * @param inFeatures number of input features
     * @param outFeatures number of output features
     * @param useBias whether to learn an additive bias parameter
     */
    public Linear(int inFeatures, int outFeatures, boolean useBias) {
        this.useBias = useBias;

        // Initialize weight using Xavier Uniform: shape [outFeatures, inFeatures]
        RandomInitializer initializer = new RandomInitializer();
        Matrix weightMatrix = new Matrix(outFeatures, inFeatures);
        initializer.fillXavierUniform(weightMatrix, inFeatures, outFeatures);
        
        this.weight = Tensor.fromMatrix(weightMatrix);
        this.weight.setRequiresGrad(true);
        registerParameter(this.weight);

        if (useBias) {
            // Bias shape: [1, outFeatures] for easy row-broadcasting
            this.bias = Tensor.zeros(1, outFeatures);
            this.bias.setRequiresGrad(true);
            registerParameter(this.bias);
        } else {
            this.bias = null;
        }
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Linear layer forward pass requires at least 1 input tensor.");
        }
        Tensor x = inputs[0];
        
        // y = x * W^T
        Tensor y = x.matmul(weight.transpose());
        if (useBias) {
            y = y.add(bias);
        }
        return y;
    }

    public Tensor getWeight() {
        return weight;
    }

    public Tensor getBias() {
        return bias;
    }
}
