package com.tinymodelz.train;

import com.tinymodelz.math.MathIO;
import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.Module;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Properties;

/**
 * <h3>Checkpoint</h3>
 *
 * <p>Handles complete training checkpoint state persistence and restoration (Phase 9).</p>
 * <p>Persists model parameter tensors, AdamW optimizer momentum/variance state vectors ($m, v$),
 * step counts, epoch metrics, and learning rates to binary TMAT files.</p>
 */
public class Checkpoint {

    private static final Logger logger = LoggerFactory.getLogger(Checkpoint.class);

    /**
     * Data holder representing restored checkpoint metadata.
     */
    public static class CheckpointState {
        public final int epoch;
        public final int globalStep;
        public final float learningRate;

        public CheckpointState(int epoch, int globalStep, float learningRate) {
            this.epoch = epoch;
            this.globalStep = globalStep;
            this.learningRate = learningRate;
        }
    }

    /**
     * Saves model weights, optimizer state, and training metadata to the target directory.
     *
     * @param model      the model instance
     * @param optimizer  the AdamW optimizer instance
     * @param epoch      current completed epoch number
     * @param globalStep current global step count
     * @param directory  target checkpoint directory
     * @throws IOException if file creation or writing fails
     */
    public static void saveCheckpoint(Module model, AdamW optimizer, int epoch, int globalStep, File directory) throws IOException {
        if (model == null || directory == null) {
            throw new IllegalArgumentException("Model and directory parameters cannot be null.");
        }
        if (!directory.exists()) {
            if (!directory.mkdirs()) {
                throw new IOException("Failed to create checkpoint directory: " + directory.getAbsolutePath());
            }
        }

        // 1. Save model parameters
        List<Tensor> params = model.getParameters();
        for (int i = 0; i < params.size(); i++) {
            File paramFile = new File(directory, "param_" + i + ".tmat");
            MathIO.saveTensor(params.get(i), paramFile);
        }

        // 2. Save optimizer state (if optimizer is provided)
        if (optimizer != null) {
            for (int i = 0; i < params.size(); i++) {
                Tensor p = params.get(i);
                float[] m = optimizer.getMState().get(p);
                float[] v = optimizer.getVState().get(p);

                if (m != null) {
                    File mFile = new File(directory, "opt_m_" + i + ".tmat");
                    MathIO.saveTensor(new Tensor(m, p.getShape()), mFile);
                }
                if (v != null) {
                    File vFile = new File(directory, "opt_v_" + i + ".tmat");
                    MathIO.saveTensor(new Tensor(v, p.getShape()), vFile);
                }
            }
        }

        // 3. Save training metadata properties
        Properties props = new Properties();
        props.setProperty("epoch", String.valueOf(epoch));
        props.setProperty("globalStep", String.valueOf(globalStep));
        props.setProperty("learningRate", String.valueOf(optimizer != null ? optimizer.getLearningRate() : 0.001f));
        props.setProperty("stepCount", String.valueOf(optimizer != null ? optimizer.getStepCount() : globalStep));

        File metaFile = new File(directory, "checkpoint_meta.properties");
        try (FileOutputStream out = new FileOutputStream(metaFile)) {
            props.store(out, "TinyModelZ Checkpoint Metadata");
        }
    }

    /**
     * Backwards-compatible save for model weights only.
     */
    public static void saveCheckpoint(Module model, File directory) throws IOException {
        saveCheckpoint(model, null, 0, 0, directory);
    }

    /**
     * Loads model weights, optimizer state, and training metadata from a checkpoint directory.
     *
     * @param model     the target model to populate
     * @param optimizer optional AdamW optimizer to restore state into
     * @param directory checkpoint source directory
     * @return restored CheckpointState metadata
     * @throws IOException if files are missing or dimension mismatch occurs
     */
    public static CheckpointState loadCheckpoint(Module model, AdamW optimizer, File directory) throws IOException {
        if (model == null || directory == null) {
            throw new IllegalArgumentException("Model and directory parameters cannot be null.");
        }
        if (!directory.exists() || !directory.isDirectory()) {
            throw new IOException("Checkpoint directory does not exist: " + directory.getAbsolutePath());
        }

        // 1. Load model parameters
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

            float[] src = loaded.toContiguous().getData();
            float[] dest = current.getData();
            System.arraycopy(src, 0, dest, current.offset(), src.length);

            // 2. Load optimizer state if available
            if (optimizer != null) {
                File mFile = new File(directory, "opt_m_" + i + ".tmat");
                if (mFile.exists()) {
                    Tensor mTensor = MathIO.loadTensor(mFile);
                    optimizer.getMState().put(current, mTensor.toContiguous().getData());
                }
                File vFile = new File(directory, "opt_v_" + i + ".tmat");
                if (vFile.exists()) {
                    Tensor vTensor = MathIO.loadTensor(vFile);
                    optimizer.getVState().put(current, vTensor.toContiguous().getData());
                }
            }
        }

        // 3. Read metadata properties if present
        int epoch = 0;
        int globalStep = 0;
        float lr = 0.001f;

        File metaFile = new File(directory, "checkpoint_meta.properties");
        if (metaFile.exists()) {
            Properties props = new Properties();
            try (FileInputStream in = new FileInputStream(metaFile)) {
                props.load(in);
                epoch = Integer.parseInt(props.getProperty("epoch", "0"));
                globalStep = Integer.parseInt(props.getProperty("globalStep", "0"));
                lr = Float.parseFloat(props.getProperty("learningRate", "0.001"));
                int stepCount = Integer.parseInt(props.getProperty("stepCount", String.valueOf(globalStep)));
                if (optimizer != null) {
                    optimizer.setStepCount(stepCount);
                    optimizer.setLearningRate(lr);
                }
            } catch (Exception e) {
                logger.warn("Failed to parse metadata file in checkpoint: {}", e.getMessage());
            }
        } else {
            String dirName = directory.getName();
            if (dirName.startsWith("epoch_")) {
                try {
                    epoch = Integer.parseInt(dirName.substring(6));
                } catch (Exception ignored) {}
            }
        }

        logger.info("Successfully loaded complete training checkpoint from: {} (Epoch: {}, Step: {}, LR: {})",
                directory.getAbsolutePath(), epoch, globalStep, lr);
        return new CheckpointState(epoch, globalStep, lr);
    }

    /**
     * Backwards-compatible load for model weights only.
     */
    public static void loadCheckpoint(Module model, File directory) throws IOException {
        loadCheckpoint(model, null, directory);
    }
}
