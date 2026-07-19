package com.tinymodelz.train;

import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.nn.TinyGPT;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * <h3>TrainTinyStories</h3>
 *
 * <p>Main executable pipeline for training TinyGPT on the <code>TinyStories-valid.txt</code> corpus.</p>
 *
 * <h4>Pipeline Execution Stages:</h4>
 * <ol>
 *   <li><b>Dataset Loading:</b> Reads <code>TinyStories-valid.txt</code> as a continuous UTF-8 token stream.</li>
 *   <li><b>Vocabulary Extraction:</b> Scans character alphabet + registers <code>&lt;|endoftext|&gt;</code> control token.</li>
 *   <li><b>Tokenization & Dataset Construction:</b> Converts corpus to <code>int[]</code> token IDs.</li>
 *   <li><b>DataLoader Initialization:</b> Configures <code>(input, target)</code> token window shifting.</li>
 *   <li><b>TinyGPT Model Construction:</b> Initializes model architecture (embedDim=64, layers=2, heads=2, feedForwardDim=256).</li>
 *   <li><b>Trainer Execution:</b> Runs overfit verification and multi-epoch training loop with metrics & generation sampling.</li>
 * </ol>
 */
public class TrainTinyStories {

    private static final Logger logger = LoggerFactory.getLogger(TrainTinyStories.class);

    public static void main(String[] args) {
        String datasetPath = args.length > 0 ? args[0] : "TinyStories-valid.txt";
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : 5;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : 16;
        int seqLen = args.length > 3 ? Integer.parseInt(args[3]) : 64;
        String deviceArg = args.length > 4 ? args[4] : "gpu";

        if ("gpu".equalsIgnoreCase(deviceArg)) {
            com.tinymodelz.math.DeviceManager.setDevice(com.tinymodelz.math.Device.GPU);
        } else {
            com.tinymodelz.math.DeviceManager.setDevice(com.tinymodelz.math.Device.CPU);
        }

        logger.info("==================================================");
        logger.info("TinyGPT Language Model Training Pipeline");
        logger.info("Compute Device: {}", com.tinymodelz.math.DeviceManager.getSummary());
        logger.info("Dataset Path:   {}", datasetPath);
        logger.info("Epochs:         {}", epochs);
        logger.info("Batch Size:     {}", batchSize);
        logger.info("Context Length: {}", seqLen);
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

        // --- Step 2: Vocabulary & Tokenizer ---
        logger.info("2. Constructing vocabulary and CharacterTokenizer...");
        List<String> customTokens = List.of("<|endoftext|>");
        CharacterTokenizer tokenizer = CharacterTokenizer.fromText(text, customTokens);
        int vocabSize = tokenizer.getVocabSize();
        logger.info("Vocabulary derived: {} tokens (including <|endoftext|> EOS)", vocabSize);

        // --- Step 3 & 4: TextDataset & DataLoader ---
        logger.info("3. Tokenizing dataset into continuous token stream...");
        TextDataset dataset = new TextDataset(text, tokenizer);
        logger.info("Total tokens in dataset: {}", dataset.size());

        DataLoader loader = new DataLoader(dataset, batchSize, seqLen, true);
        logger.info("DataLoader ready: {} batches per epoch", loader.getNumBatches());

        // --- Step 5: TinyGPT Model ---
        logger.info("4. Instantiating TinyGPT model architecture...");
        int embedDim = 64;
        int numLayers = 2;
        int numHeads = 2;
        int feedForwardDim = 256;
        float dropoutProb = 0.1f;
        float learningRate = 3e-4f;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, seqLen, numLayers, numHeads, feedForwardDim, dropoutProb);
        logger.info("Model Summary: {}", model.summary());

        AdamW optimizer = new AdamW(model.getParameters(), learningRate);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Trainer trainer = new Trainer(model, tokenizer, optimizer, lossFn);

        // --- Step 6: Single-batch Overfit Check ---
        logger.info("5. Running sanity overfit test on single batch...");
        DataLoader overfitLoader = new DataLoader(dataset, 1, seqLen, false);
        trainer.trainOverfit(overfitLoader, 30, 0.5f);

        // --- Step 7: Full Training ---
        logger.info("6. Starting full training loop for {} epochs...", epochs);
        File checkpointDir = new File("checkpoints/tinystories");
        trainer.train(loader, epochs, checkpointDir, "Once upon a time", 50, 100);

        logger.info("==================================================");
        logger.info("TinyGPT Training Pipeline Completed Successfully!");
        logger.info("Checkpoints saved to: {}", checkpointDir.getAbsolutePath());
        logger.info("==================================================");
    }
}
