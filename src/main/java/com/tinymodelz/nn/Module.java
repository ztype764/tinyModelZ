package com.tinymodelz.nn;

import com.tinymodelz.math.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * <h3>Module</h3>
 * 
 * <p>Base abstract class for all neural network components (layers, networks).</p>
 * <p>Manages training/evaluation mode state and encapsulates learnable parameters.</p>
 */
public abstract class Module {

    protected boolean training = true;
    protected final List<Tensor> parameters = new ArrayList<>();

    /**
     * Executes the forward pass computation of the module.
     * 
     * @param inputs input tensors to the layer
     * @return the result tensor
     */
    public abstract Tensor forward(Tensor... inputs);

    /**
     * Sets the module (and recursively any submodules) to training mode.
     */
    public void train() {
        this.training = true;
    }

    /**
     * Sets the module (and recursively any submodules) to evaluation/inference mode.
     */
    public void eval() {
        this.training = false;
    }

    /**
     * Registers a learnable tensor parameter in this module.
     * 
     * @param param the parameter tensor
     */
    protected void registerParameter(Tensor param) {
        if (param != null) {
            parameters.add(param);
        }
    }

    /**
     * Returns a list of all parameters registered in this module.
     * 
     * @return a list of parameter tensors
     */
    public List<Tensor> getParameters() {
        return new ArrayList<>(parameters);
    }
}
