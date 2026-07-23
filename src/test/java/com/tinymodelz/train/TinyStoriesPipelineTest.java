package com.tinymodelz.train;

import com.tinymodelz.TestReporter;
import com.tinymodelz.inference.Generator;
import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.nn.TinyGPT;
import com.tinymodelz.tokenizer.CharacterTokenizer;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import org.springframework.lang.NonNull;

/**
 * <h3>TinyStoriesPipelineTest</h3>
 *
 * <p>
 * Integration unit test suite validating all steps of the language modeling
 * pipeline:
 * <ul>
 * <li>1. Dataset Loading (UTF-8 stream parsing, EOS token detection)</li>
 * <li>2. Vocabulary Building (Automatic dataset character vocabulary
 * generation)</li>
 * <li>3. Tokenization Round-Trip (Exact encode -> decode text matching)</li>
 * <li>4. DataLoader Target Shifting (Input $X_t$, Label $Y_{t+1}$ target
 * alignment)</li>
 * <li>5. TinyGPT Model Dimensions ([B, T, V] shape verification)</li>
 * <li>6. Single-Batch Overfitting (Training loss reduction close to zero)</li>
 * <li>7. Checkpoint Save & Load (Weight restoration and post-reload
 * inference)</li>
 * </ul>
 * </p>
 */
public class TinyStoriesPipelineTest {

    private static final String SAMPLE_CORPUS = "Once upon a time, there was a little dog named Max.\n" +
            "Max loved to play in the park with his ball.\n" +
            "<|endoftext|>\n" +
            "One day, a girl named Mia found a pretty red box.\n" +
            "She opened it and smiled.\n" +
            "<|endoftext|>\n";

    public static void runTests() {
        TestReporter.runTest("1. Dataset loading and UTF-8 stream reading", () -> {
            try {
                testDatasetLoading();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        TestReporter.runTest("2. Automatic vocabulary extraction and tokenization round-trip",
                () -> testTokenizerRoundtrip());
        TestReporter.runTest("3. DataLoader target shifting logic", () -> testDataLoaderShifting());
        TestReporter.runTest("4. TinyGPT output tensor dimensions", () -> testTinyGPTDimensions());
        TestReporter.runTest("5. Single-batch overfit convergence test", () -> testOverfitConvergence());
        TestReporter.runTest("6. Checkpoint save, reload, and post-reload inference", () -> {
            try {
                testCheckpointAndInference();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void assertEquals(Object expected, Object actual, String message) {
        if (!expected.equals(actual)) {
            throw new AssertionError(message + " — Expected: " + expected + ", Got: " + actual);
        }
    }

    /**
     * 1. Verifies dataset loading from file as UTF-8 string.
     */
    private static void testDatasetLoading() throws IOException {
        Path tempFile = Files.createTempFile("tinystories_test", ".txt");
        Files.writeString(tempFile, SAMPLE_CORPUS, StandardCharsets.UTF_8);

        String loadedText = Files.readString(tempFile, StandardCharsets.UTF_8);
        Files.deleteIfExists(tempFile);

        assertEquals(SAMPLE_CORPUS, loadedText, "Loaded UTF-8 dataset content mismatch");
        assertTrue(loadedText.contains("<|endoftext|>"), "Dataset must contain <|endoftext|> EOS markers");

        TestReporter.logMetric("Dataset character count", loadedText.length());
        TestReporter.logMetric("EOS marker count", loadedText.split("<\\|endoftext\\|>").length - 1);
    }

    /**
     * 2. Verifies auto-vocabulary building and exact round-trip encode -> decode.
     */
    private static void testTokenizerRoundtrip() {
        List<String> customTokens = List.of("<|endoftext|>");
        CharacterTokenizer tokenizer = CharacterTokenizer.fromText(SAMPLE_CORPUS, customTokens);

        List<Integer> encoded = tokenizer.encode(SAMPLE_CORPUS);
        String decoded = tokenizer.decode(encoded, false);

        assertEquals(SAMPLE_CORPUS, decoded, "Encode -> Decode round-trip failed to match original text!");

        int eosId = tokenizer.tokenToId("<|endoftext|>");
        assertTrue(eosId != tokenizer.getUnkId(), "<|endoftext|> must be registered as a valid vocabulary ID");
        assertTrue(encoded.contains(eosId), "Encoded sequence must contain <|endoftext|> token ID");

        TestReporter.logMetric("Vocabulary size", tokenizer.getVocabSize());
        TestReporter.logMetric("EOS Token ID", eosId);
        TestReporter.logMetric("Encoded token sequence length", encoded.size());
    }

    /**
     * 3. Verifies DataLoader target shifting: Input[t] -> Label[t+1].
     */
    private static void testDataLoaderShifting() {
        List<String> customTokens = List.of("<|endoftext|>");
        CharacterTokenizer tokenizer = CharacterTokenizer.fromText(SAMPLE_CORPUS, customTokens);
        TextDataset dataset = new TextDataset(SAMPLE_CORPUS, tokenizer);

        int batchSize = 1;
        int seqLen = 4;
        DataLoader loader = new DataLoader(dataset, batchSize, seqLen, false);

        assertTrue(loader.hasNext(), "DataLoader must have at least 1 batch");
        Tensor[] batch = loader.nextBatch();

        Tensor x = batch[0]; // [1, 4]
        Tensor y = batch[1]; // [1, 4]

        float[] xData = x.getData();
        float[] yData = y.getData();

        for (int i = 0; i < seqLen - 1; i++) {
            assertEquals(xData[i + 1], yData[i],
                    "DataLoader target label at index " + i + " must equal input token at index " + (i + 1));
        }

        TestReporter.logMetric("Input batch X", Arrays.toString(xData));
        TestReporter.logMetric("Target batch Y", Arrays.toString(yData));
    }

    /**
     * 4. Verifies TinyGPT forward output shape [B, T, V].
     */
    private static void testTinyGPTDimensions() {
        int vocabSize = 30;
        int embedDim = 16;
        int maxSeqLen = 32;
        int numLayers = 2;
        int numHeads = 2;
        int feedForwardDim = 64;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, feedForwardDim, 0.0f);
        model.eval();

        float[] inputIds = { 1, 2, 3, 4, 5, 6, 7, 8 };
        Tensor x = new Tensor(inputIds, new int[] { 2, 4 }); // [B=2, T=4]

        Tensor logits = model.forward(x);
        int[] expectedShape = { 2, 4, vocabSize };

        assertTrue(Arrays.equals(expectedShape, logits.getShape()),
                "TinyGPT output shape mismatch — expected " + Arrays.toString(expectedShape) + ", got "
                        + Arrays.toString(logits.getShape()));

        TestReporter.logMetric("Input shape", Arrays.toString(x.getShape()));
        TestReporter.logMetric("Output logits shape", Arrays.toString(logits.getShape()));
        TestReporter.logMetric("Model parameter count", model.getParameters().size());
    }

    /**
     * 5. Single-batch overfit convergence test using Trainer.
     */
    private static void testOverfitConvergence() {
        CharacterTokenizer tokenizer = CharacterTokenizer.fromText(SAMPLE_CORPUS, List.of("<|endoftext|>"));
        TextDataset dataset = new TextDataset(SAMPLE_CORPUS, tokenizer);

        DataLoader loader = new DataLoader(dataset, 1, 8, false);

        int vocabSize = tokenizer.getVocabSize();
        int embedDim = 16;
        int maxSeqLen = 16;
        int numLayers = 1;
        int numHeads = 2;
        int feedForwardDim = 64;

        TinyGPT model = new TinyGPT(vocabSize, embedDim, maxSeqLen, numLayers, numHeads, feedForwardDim, 0.0f);
        AdamW optimizer = new AdamW(model.getParameters(), 0.01f);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();

        Trainer trainer = new Trainer(model, tokenizer, optimizer, lossFn);
        float finalLoss = trainer.trainOverfit(loader, 20, 0.5f);

        assertTrue(finalLoss <= 1.0f, "Overfit test failed — final loss " + finalLoss + " > 1.0f");
        TestReporter.logMetric("Achieved final overfit loss", finalLoss);
    }

    /**
     * 6. Verifies checkpoint persistence, reload weight restoration, and generation
     * post-reload.
     */
    private static void testCheckpointAndInference() throws IOException {
        CharacterTokenizer tokenizer = CharacterTokenizer.fromText(SAMPLE_CORPUS, List.of("<|endoftext|>"));
        int vocabSize = tokenizer.getVocabSize();

        TinyGPT originalModel = new TinyGPT(vocabSize, 16, 16, 1, 2, 64, 0.0f);
        File tempDir = Files.createTempDirectory("checkpoint_test").toFile();
        tempDir.deleteOnExit();

        // Save original model checkpoint
        Checkpoint.saveCheckpoint(originalModel, tempDir);

        // Instantiate new model with identical architecture and load weights
        TinyGPT loadedModel = new TinyGPT(vocabSize, 16, 16, 1, 2, 64, 0.0f);
        Checkpoint.loadCheckpoint(loadedModel, tempDir);

        // Verify parameters match exactly
        List<Tensor> origParams = originalModel.getParameters();
        List<Tensor> loadedParams = loadedModel.getParameters();

        for (int i = 0; i < origParams.size(); i++) {
            float[] origData = origParams.get(i).getData();
            float[] loadedData = loadedParams.get(i).getData();
            assertTrue(Arrays.equals(origData, loadedData), "Parameter index " + i + " mismatch after reload!");
        }

        // Perform text generation with loaded model
        Generator generator = new Generator(123L);
        String generated = generator.generate(loadedModel, tokenizer, "Once", 10, 0.7f, 10, 0.9f, 16, null);

        assertTrue(generated != null && generated.length() > 0, "Generated text post-reload must not be empty");
        TestReporter.logMetric("Checkpoint directory", tempDir.getAbsolutePath());
        TestReporter.logMetric("Generated sample text", generated.trim());
    }
}
