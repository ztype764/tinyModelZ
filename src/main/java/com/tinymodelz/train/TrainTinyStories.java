package com.tinymodelz.train;

import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.nn.TinyGPT;
import com.tinymodelz.tokenizer.Tokenizer;
import com.tinymodelz.tokenizer.TokenizerFactory;
import com.tinymodelz.tokenizer.TokenizerFactory.TokenizerType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Properties;

/**
 * <h3>TrainTinyStories</h3>
 *
 * <p>Main executable pipeline for training TinyGPT models with configurable dataset,
 * tokenizer (character, BPE, Trie), training resumption, and structured checkpointing.</p>
 */
public class TrainTinyStories {

    private static final Logger logger = LoggerFactory.getLogger(TrainTinyStories.class);

    public static void main(String[] args) {
        String datasetPath = "TinyStories-valid-reduced.txt";
        if (!Files.exists(Path.of(datasetPath)) && Files.exists(Path.of("TinyStories-valid.txt"))) {
            datasetPath = "TinyStories-valid.txt";
        }
        int epochs = 5;
        int batchSize = 16;
        int seqLen = 64;
        String deviceArg = "gpu";
        int embedDim = 64;
        int numLayers = 2;
        int numHeads = 2;
        int feedForwardDim = 4 * embedDim;
        String tokenizerTypeArg = "character";
        String resumePath = null;
        int requestedStartEpoch = -1;

        // Parse CLI arguments
        for (int i = 0; i < args.length; i++) {
            if ("--dataset".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                datasetPath = args[++i];
            } else if ("--epochs".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                epochs = Integer.parseInt(args[++i]);
            } else if ("--batch-size".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                batchSize = Integer.parseInt(args[++i]);
            } else if ("--seq-len".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                seqLen = Integer.parseInt(args[++i]);
            } else if ("--device".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                deviceArg = args[++i];
            } else if ("--tokenizer".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                tokenizerTypeArg = args[++i];
            } else if (("--resume".equalsIgnoreCase(args[i]) || "--checkpoint".equalsIgnoreCase(args[i])) && i + 1 < args.length && !args[i + 1].startsWith("--")) {
                resumePath = args[++i];
            } else if ("--resume".equalsIgnoreCase(args[i])) {
                resumePath = "auto";
            } else if ("--start-epoch".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                requestedStartEpoch = Integer.parseInt(args[++i]);
            } else if (i == 0 && !args[i].startsWith("--")) {
                datasetPath = args[0];
            } else if (i == 1 && !args[i].startsWith("--")) {
                epochs = Integer.parseInt(args[1]);
            } else if (i == 2 && !args[i].startsWith("--")) {
                batchSize = Integer.parseInt(args[2]);
            } else if (i == 3 && !args[i].startsWith("--")) {
                seqLen = Integer.parseInt(args[3]);
            } else if (i == 4 && !args[i].startsWith("--")) {
                deviceArg = args[4];
            } else if (i == 5 && !args[i].startsWith("--")) {
                embedDim = Integer.parseInt(args[5]);
            } else if (i == 6 && !args[i].startsWith("--")) {
                numLayers = Integer.parseInt(args[6]);
            } else if (i == 7 && !args[i].startsWith("--")) {
                numHeads = Integer.parseInt(args[7]);
            } else if (i == 8 && !args[i].startsWith("--")) {
                feedForwardDim = Integer.parseInt(args[8]);
            } else if (i == 9 && !args[i].startsWith("--")) {
                tokenizerTypeArg = args[9];
            } else if (i == 10 && !args[i].startsWith("--")) {
                resumePath = args[10];
            }
        }

        if ("cuda".equalsIgnoreCase(deviceArg)) {
            com.tinymodelz.math.DeviceManager.setDevice(com.tinymodelz.math.Device.GPU_CUDA);
        } else if ("opencl".equalsIgnoreCase(deviceArg)) {
            com.tinymodelz.math.DeviceManager.setDevice(com.tinymodelz.math.Device.GPU_OPENCL);
        } else if ("gpu".equalsIgnoreCase(deviceArg)) {
            com.tinymodelz.math.DeviceManager.setDevice(com.tinymodelz.math.Device.GPU);
        } else {
            com.tinymodelz.math.DeviceManager.setDevice(com.tinymodelz.math.Device.CPU);
        }

        TokenizerType tokenizerType = TokenizerType.fromString(tokenizerTypeArg);

        logger.info("==================================================");
        logger.info("TinyGPT Language Model Training Pipeline");
        logger.info("Compute Device: {}", com.tinymodelz.math.DeviceManager.getSummary());
        logger.info("Dataset Path:   {}", datasetPath);
        logger.info("Tokenizer Type: {}", tokenizerType.name());
        logger.info("Epochs:         {}", epochs);
        logger.info("Batch Size:     {}", batchSize);
        logger.info("Context Length: {}", seqLen);
        if (resumePath != null) {
            logger.info("Resume Mode:    Active ({})", resumePath);
        }
        logger.info("==================================================");

        Path datasetFile = Path.of(datasetPath);
        if (!Files.exists(datasetFile)) {
            logger.error("Dataset file not found at path: {}", datasetFile.toAbsolutePath());
            System.exit(1);
        }

        // --- Step 1: Read Dataset ---
        logger.info("1. Reading raw dataset file...");
        String text;
        try {
            text = Files.readString(datasetFile, StandardCharsets.UTF_8);
            logger.info("Dataset loaded successfully: {} characters, {} bytes", text.length(), Files.size(datasetFile));
        } catch (IOException e) {
            logger.error("Failed to read dataset file: {}", e.getMessage());
            System.exit(1);
            return;
        }

        // Determine or find Checkpoint Run Directory
        File checkpointRunDir = null;
        File resumeCheckpointDir = null;

        if (resumePath != null) {
            if (!"auto".equalsIgnoreCase(resumePath)) {
                File candidate = new File(resumePath);
                if (candidate.exists()) {
                    if (candidate.getName().startsWith("epoch_") || candidate.getName().equals("best_checkpoint")) {
                        resumeCheckpointDir = candidate;
                        checkpointRunDir = candidate.getParentFile();
                    } else {
                        checkpointRunDir = candidate;
                        resumeCheckpointDir = findLatestEpochCheckpoint(candidate);
                    }
                }
            }
            if (checkpointRunDir == null || !checkpointRunDir.exists()) {
                // Auto-discover latest run matching dataset/tokenizer
                File baseDir = new File("checkpoints");
                if (baseDir.exists() && baseDir.isDirectory()) {
                    File[] subdirs = baseDir.listFiles(File::isDirectory);
                    if (subdirs != null) {
                        File latestRun = null;
                        long lastModified = 0;
                        for (File dir : subdirs) {
                            if (dir.lastModified() > lastModified && findLatestEpochCheckpoint(dir) != null) {
                                lastModified = dir.lastModified();
                                latestRun = dir;
                            }
                        }
                        if (latestRun != null) {
                            checkpointRunDir = latestRun;
                            resumeCheckpointDir = findLatestEpochCheckpoint(latestRun);
                        }
                    }
                }
            }
        }

        // --- Step 2: Vocabulary & Tokenizer ---
        logger.info("2. Constructing vocabulary and Tokenizer ({}) ...", tokenizerType.name());
        Tokenizer tokenizer = null;
        if (checkpointRunDir != null) {
            tokenizer = TokenizerFactory.loadTokenizer(checkpointRunDir);
        }
        if (tokenizer == null) {
            List<String> customTokens = List.of("<|endoftext|>");
            tokenizer = TokenizerFactory.createTokenizer(tokenizerType, text, customTokens);
        }

        int vocabSize = tokenizer.getVocabSize();
        logger.info("Vocabulary derived: {} tokens (Tokenizer: {})", vocabSize, tokenizer.getClass().getSimpleName());

        // --- Step 3 & 4: TextDataset & DataLoader ---
        logger.info("3. Tokenizing dataset into continuous token stream...");
        TextDataset dataset = new TextDataset(text, tokenizer);
        logger.info("Total tokens in dataset: {}", dataset.size());

        DataLoader loader = new DataLoader(dataset, batchSize, seqLen, true);
        logger.info("DataLoader ready: {} batches per epoch", loader.getNumBatches());

        // --- Step 5: TinyGPT Model Construction & Resuming ---
        logger.info("4. Instantiating TinyGPT model architecture...");
        float dropoutProb = 0.1f;
        float learningRate = 3e-4f;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, seqLen, numLayers, numHeads, feedForwardDim, dropoutProb);
        logger.info("Model Summary: {}", model.summary());

        AdamW optimizer = new AdamW(model.getParameters(), learningRate);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Trainer trainer = new Trainer(model, tokenizer, optimizer, lossFn);

        int startEpoch = 1;

        if (resumeCheckpointDir != null && resumeCheckpointDir.exists()) {
            try {
                Checkpoint.CheckpointState state = Checkpoint.loadCheckpoint(model, optimizer, resumeCheckpointDir);
                if (state.epoch > 0) {
                    startEpoch = state.epoch + 1;
                } else if (requestedStartEpoch > 0) {
                    startEpoch = requestedStartEpoch;
                } else {
                    startEpoch = 2; // Default to restarting at epoch 2
                }
                logger.info("✅ Resumed model weights & optimizer state from: {} (Next Start Epoch: {})",
                        resumeCheckpointDir.getAbsolutePath(), startEpoch);
            } catch (IOException e) {
                logger.error("Failed to load resume checkpoint from {}: {}", resumeCheckpointDir.getAbsolutePath(), e.getMessage());
            }
        } else if (requestedStartEpoch > 1) {
            startEpoch = requestedStartEpoch;
        }

        // --- Step 6: Create Checkpoint Run Subfolder if not resuming existing ---
        if (checkpointRunDir == null) {
            String cleanDatasetName = datasetFile.getFileName().toString().replace(".txt", "").replaceAll("[^a-zA-Z0-9_]", "_");
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String runFolderName = cleanDatasetName + "_" + tokenizerType.toSubfolderName() + "_" + timestamp;
            checkpointRunDir = new File("checkpoints", runFolderName);
        }

        if (!checkpointRunDir.exists() && !checkpointRunDir.mkdirs()) {
            logger.warn("Could not create run checkpoint directory: {}", checkpointRunDir.getAbsolutePath());
        }

        // Save tokenizer & run metadata inside run directory
        try {
            TokenizerFactory.saveTokenizer(tokenizer, tokenizerType, checkpointRunDir);
            saveRunMetadata(checkpointRunDir, datasetPath, tokenizerType.name(), epochs, batchSize, seqLen, embedDim, numLayers, numHeads, feedForwardDim, vocabSize);
        } catch (IOException e) {
            logger.warn("Failed to write run metadata/tokenizer config: {}", e.getMessage());
        }

        // --- Step 7: Single-batch Overfit Check ---
        if (startEpoch == 1) {
            logger.info("5. Running sanity overfit test on single batch...");
            DataLoader overfitLoader = new DataLoader(dataset, 1, seqLen, false);
            trainer.trainOverfit(overfitLoader, 30, 0.5f);
        }

        // --- Step 8: Full Training Loop ---
        logger.info("6. Starting training loop from Epoch {} to Epoch {}...", startEpoch, epochs);
        List<String> evalPrompts = List.of(
            "Once upon a time",
            "One day, a little dog",
            "The cat saw a",
            "Lily went to the park"
        );

        trainer.train(loader, null, startEpoch, epochs, checkpointRunDir, evalPrompts, 50, 100);

        logger.info("==================================================");
        logger.info("TinyGPT Training Pipeline Completed Successfully!");
        logger.info("Checkpoints saved in subfolder: {}", checkpointRunDir.getAbsolutePath());
        logger.info("==================================================");
    }

    private static File findLatestEpochCheckpoint(File runDir) {
        if (runDir == null || !runDir.exists() || !runDir.isDirectory()) return null;
        File best = new File(runDir, "best_checkpoint");
        File[] epochDirs = runDir.listFiles((dir, name) -> name.startsWith("epoch_"));
        if (epochDirs != null && epochDirs.length > 0) {
            File maxEpochDir = null;
            int maxEpoch = -1;
            for (File ed : epochDirs) {
                try {
                    int num = Integer.parseInt(ed.getName().substring(6));
                    if (num > maxEpoch) {
                        maxEpoch = num;
                        maxEpochDir = ed;
                    }
                } catch (Exception ignored) {}
            }
            if (maxEpochDir != null) return maxEpochDir;
        }
        if (best.exists()) return best;
        return null;
    }

    private static void saveRunMetadata(File runDir, String datasetPath, String tokenizerType, int epochs, int batchSize,
                                         int seqLen, int embedDim, int numLayers, int numHeads, int feedForwardDim, int vocabSize) throws IOException {
        Properties props = new Properties();
        props.setProperty("datasetPath", datasetPath);
        props.setProperty("datasetName", Path.of(datasetPath).getFileName().toString());
        props.setProperty("tokenizerType", tokenizerType.toLowerCase());
        props.setProperty("startedAt", new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
        props.setProperty("epochs", String.valueOf(epochs));
        props.setProperty("batchSize", String.valueOf(batchSize));
        props.setProperty("seqLen", String.valueOf(seqLen));
        props.setProperty("embedDim", String.valueOf(embedDim));
        props.setProperty("numLayers", String.valueOf(numLayers));
        props.setProperty("numHeads", String.valueOf(numHeads));
        props.setProperty("feedForwardDim", String.valueOf(feedForwardDim));
        props.setProperty("vocabSize", String.valueOf(vocabSize));

        File metaFile = new File(runDir, "run_info.properties");
        try (FileOutputStream out = new FileOutputStream(metaFile)) {
            props.store(out, "TinyModelZ Training Run Metadata");
        }
    }
}
