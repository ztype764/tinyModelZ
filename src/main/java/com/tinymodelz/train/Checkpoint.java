package com.tinymodelz.train;

import com.tinymodelz.math.MathIO;
import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.Module;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * <h3>Checkpoint</h3>
 * 
 * <p>Handles weight persistence for training models.
 * Saves and loads neural network parameters using binary TMAT format serialization.</p>
 */
public class Checkpoint {

    /**
     * Saves all model parameters as checkpoint TMAT files in the target directory.
     * 
     * @param model the model to checkpoint
     * @param directory the destination directory
     * @throws IOException if saving fails
     */
    public static void saveCheckpoint(Module model, File directory) throws IOException {
        if (model == null || directory == null) {
            throw new IllegalArgumentException("Model and directory parameters cannot be null");
        }
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                throw new IOException("Failed to create checkpoint directory: " + directory.getAbsolutePath());
            }
        }
        
        List<Tensor> params = model.getParameters();
        for (int i = 0; i < params.size(); i++) {
            File paramFile = new File(directory, "param_" + i + ".tmat");
            MathIO.saveTensor(params.get(i), paramFile);
        }
    }

    /**
     * Loads parameter weights from target directory checkpoint files into the model.
     * Validates dimensions before performing raw updates.
     * 
     * @param model the target model to update
     * @param directory the checkpoint directory to load weights from
     * @throws IOException if files are missing or shape mismatches occur
     */
    public static void loadCheckpoint(Module model, File directory) throws IOException {
        if (model == null || directory == null) {
            throw new IllegalArgumentException("Model and directory parameters cannot be null");
        }
        if (!directory.exists() || !directory.isDirectory()) {
            throw new IOException("Checkpoint directory does not exist: " + directory.getAbsolutePath());
        }

        List<Tensor> params = model.getParameters();
        for (int i = 0; i < params.size(); i++) {
            File paramFile = new File(directory, "param_" + i + ".tmat");
            if (!paramFile.exists()) {
                throw new IOException("Checkpoint parameter file missing: " + paramFile.getAbsolutePath());
            }

            Tensor loaded = MathIO.loadTensor(paramFile);
            Tensor current = params.get(i);

            if (!java.util.Arrays.equals(loaded.getShape(), current.getShape())) {
                throw new IOException("Dimension mismatch at index " + i + ": expected shape " 
                    + java.util.Arrays.toString(current.getShape()) + ", got shape " 
                    + java.util.Arrays.toString(loaded.getShape()));
            }

            // Copy contiguous values over to memory representation
            float[] src = loaded.toContiguous().getData();
            float[] dest = current.getData();
            System.arraycopy(src, 0, dest, current.offset(), src.length);
        }
    }
}
