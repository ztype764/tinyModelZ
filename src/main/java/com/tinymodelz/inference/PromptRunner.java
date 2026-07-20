package com.tinymodelz.inference;

import com.tinymodelz.math.Device;
import com.tinymodelz.math.DeviceManager;
import com.tinymodelz.nn.TinyGPT;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import com.tinymodelz.tokenizer.Tokenizer;
import com.tinymodelz.tokenizer.TokenizerFactory;
import com.tinymodelz.train.Checkpoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

/**
 * <h3>PromptRunner</h3>
 *
 * <p>Interactive CLI Application for selecting trained TinyGPT model checkpoints,
 * loading matching tokenizers, and running autoregressive inference.</p>
 */
public class PromptRunner {

    private static final Logger logger = LoggerFactory.getLogger(PromptRunner.class);

    public static class CheckpointRunOption {
        public final File dir;
        public final String datasetName;
        public final String tokenizerType;
        public final String startedAt;
        public final List<File> epochCheckpoints;

        public CheckpointRunOption(File dir, String datasetName, String tokenizerType, String startedAt, List<File> epochCheckpoints) {
            this.dir = dir;
            this.datasetName = datasetName;
            this.tokenizerType = tokenizerType;
            this.startedAt = startedAt;
            this.epochCheckpoints = epochCheckpoints;
        }
    }

    public static void main(String[] args) {
        String checkpointPath = null;
        String datasetPath = "TinyStories-valid-reduced.txt";
        if (!Files.exists(Path.of(datasetPath)) && Files.exists(Path.of("TinyStories-valid.txt"))) {
            datasetPath = "TinyStories-valid.txt";
        }
        String prompt = null;
        int maxNewTokens = 50;
        float temperature = 0.7f;
        int topK = 40;
        float topP = 0.9f;
        String deviceArg = "gpu";
        boolean listCheckpoints = false;

        // Parse CLI arguments
        for (int i = 0; i < args.length; i++) {
            if ("--checkpoint".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                checkpointPath = args[++i];
            } else if ("--dataset".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                datasetPath = args[++i];
            } else if ("--prompt".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                prompt = args[++i];
            } else if ("--temp".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                temperature = Float.parseFloat(args[++i]);
            } else if ("--max-tokens".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                maxNewTokens = Integer.parseInt(args[++i]);
            } else if ("--device".equalsIgnoreCase(args[i]) && i + 1 < args.length) {
                deviceArg = args[++i];
            } else if ("--list".equalsIgnoreCase(args[i])) {
                listCheckpoints = true;
            } else if (i == 0 && !args[i].startsWith("--")) {
                checkpointPath = args[0];
            } else if (i == 1 && !args[i].startsWith("--")) {
                datasetPath = args[1];
            } else if (i == 2 && !args[i].startsWith("--")) {
                prompt = args[2];
            }
        }

        if ("cuda".equalsIgnoreCase(deviceArg)) {
            DeviceManager.setDevice(Device.GPU_CUDA);
        } else if ("opencl".equalsIgnoreCase(deviceArg)) {
            DeviceManager.setDevice(Device.GPU_OPENCL);
        } else if ("gpu".equalsIgnoreCase(deviceArg)) {
            DeviceManager.setDevice(Device.GPU);
        } else {
            DeviceManager.setDevice(Device.CPU);
        }

        System.out.println("==================================================");
        System.out.println("🤖 TinyModelZ Model Prompt Inference Runner");
        System.out.println("Compute Device:  " + DeviceManager.getSummary());

        // Scan available checkpoint runs
        List<CheckpointRunOption> availableRuns = discoverCheckpointRuns();

        if (listCheckpoints) {
            printAvailableRuns(availableRuns);
            return;
        }

        File selectedCheckpointDir = null;
        if (checkpointPath != null) {
            selectedCheckpointDir = new File(checkpointPath);
        } else if (!availableRuns.isEmpty()) {
            System.out.println("\n🔍 Discovered Available Checkpoint Runs:");
            printAvailableRuns(availableRuns);
            selectedCheckpointDir = availableRuns.get(0).epochCheckpoints.get(availableRuns.get(0).epochCheckpoints.size() - 1);
            System.out.println("\n-> Selected Default Checkpoint: " + selectedCheckpointDir.getPath());
        } else {
            selectedCheckpointDir = new File("checkpoints/tinystories");
        }

        System.out.println("Selected Checkpoint: " + selectedCheckpointDir.getAbsolutePath());
        System.out.println("==================================================");

        // 1. Load exact tokenizer or construct from dataset
        Tokenizer tokenizer = TokenizerFactory.loadTokenizer(selectedCheckpointDir);
        if (tokenizer == null && selectedCheckpointDir.getParentFile() != null) {
            tokenizer = TokenizerFactory.loadTokenizer(selectedCheckpointDir.getParentFile());
        }
        if (tokenizer == null) {
            tokenizer = loadFallbackTokenizer(datasetPath);
        }

        int vocabSize = tokenizer.getVocabSize();
        int seqLen = 64;
        System.out.println("✅ Tokenizer Loaded: " + tokenizer.getClass().getSimpleName() + " (Vocab Size: " + vocabSize + ")");

        // 2. Instantiate Model Architecture
        TinyGPT model = new TinyGPT(vocabSize, 64, seqLen, 2, 2, 256, 0.1f);

        // 3. Load Model Weights from Checkpoint
        if (selectedCheckpointDir.exists()) {
            try {
                Checkpoint.loadCheckpoint(model, selectedCheckpointDir);
                System.out.println("✅ Checkpoint weights loaded successfully from: " + selectedCheckpointDir.getAbsolutePath());
            } catch (IOException e) {
                System.err.println("⚠️ Warning: Could not load checkpoint from " + selectedCheckpointDir.getAbsolutePath() + ": " + e.getMessage());
                System.err.println("   Using initialized model weights.");
            }
        } else {
            System.out.println("ℹ️ Checkpoint path not found. Using initialized model weights.");
        }

        Generator generator = new Generator(42L);
        Integer eosId = tokenizer.tokenToId("<|endoftext|>");

        // 4. Execution Mode: Single Prompt vs Interactive REPL Loop
        if (prompt != null && !prompt.trim().isEmpty()) {
            generateAndPrint(model, tokenizer, generator, prompt, maxNewTokens, temperature, topK, topP, seqLen, eosId);
        } else {
            runInteractiveRepl(model, tokenizer, generator, maxNewTokens, temperature, topK, topP, seqLen, eosId);
        }
    }

    public static List<CheckpointRunOption> discoverCheckpointRuns() {
        List<CheckpointRunOption> runs = new ArrayList<>();
        File checkpointsBase = new File("checkpoints");
        if (!checkpointsBase.exists() || !checkpointsBase.isDirectory()) return runs;

        File[] subdirs = checkpointsBase.listFiles(File::isDirectory);
        if (subdirs == null) return runs;

        for (File dir : subdirs) {
            File runInfo = new File(dir, "run_info.properties");
            String datasetName = dir.getName();
            String tokenizerType = "character";
            String startedAt = "unknown";

            if (runInfo.exists()) {
                Properties props = new Properties();
                try (FileInputStream in = new FileInputStream(runInfo)) {
                    props.load(in);
                    datasetName = props.getProperty("datasetName", datasetName);
                    tokenizerType = props.getProperty("tokenizerType", tokenizerType);
                    startedAt = props.getProperty("startedAt", startedAt);
                } catch (Exception ignored) {}
            } else if (dir.getName().contains("_")) {
                String[] parts = dir.getName().split("_");
                if (parts.length >= 2) {
                    tokenizerType = parts[1];
                }
            }

            List<File> chkFiles = new ArrayList<>();
            File[] epochDirs = dir.listFiles((d, name) -> name.startsWith("epoch_") || name.equals("best_checkpoint"));
            if (epochDirs != null) {
                for (File ed : epochDirs) chkFiles.add(ed);
            }
            if (chkFiles.isEmpty() && new File(dir, "param_0.tmat").exists()) {
                chkFiles.add(dir);
            }

            if (!chkFiles.isEmpty()) {
                runs.add(new CheckpointRunOption(dir, datasetName, tokenizerType, startedAt, chkFiles));
            }
        }
        return runs;
    }

    private static void printAvailableRuns(List<CheckpointRunOption> runs) {
        if (runs.isEmpty()) {
            System.out.println("No saved training runs found in checkpoints/ directory.");
            return;
        }
        System.out.println("\n--------------------------------------------------");
        System.out.println("Available Training Runs & Checkpoints:");
        System.out.println("--------------------------------------------------");
        for (int i = 0; i < runs.size(); i++) {
            CheckpointRunOption option = runs.get(i);
            System.out.printf("[%d] Run Dir: %s\n", i + 1, option.dir.getName());
            System.out.printf("    Dataset: %s | Tokenizer: %s | Started: %s\n", option.datasetName, option.tokenizerType.toUpperCase(), option.startedAt);
            System.out.print("    Available Checkpoints: ");
            for (File chk : option.epochCheckpoints) {
                System.out.print(chk.getName() + "  ");
            }
            System.out.println("\n--------------------------------------------------");
        }
    }

    private static Tokenizer loadFallbackTokenizer(String datasetPath) {
        Path path = Path.of(datasetPath);
        if (Files.exists(path)) {
            try {
                String text = Files.readString(path, StandardCharsets.UTF_8);
                return CharacterTokenizer.fromText(text, List.of("<|endoftext|>"));
            } catch (IOException e) {
                logger.warn("Could not read dataset file for vocab extraction: {}", e.getMessage());
            }
        }
        List<String> defaultVocab = List.of(
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
                "V", "W", "X", "Y", "Z",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", ".", ",", "!", "?", "'", "\"", "\n",
                "<|endoftext|>");
        return new CharacterTokenizer(defaultVocab);
    }

    private static void runInteractiveRepl(
            TinyGPT model,
            Tokenizer tokenizer,
            Generator generator,
            int defaultMaxTokens,
            float defaultTemp,
            int defaultTopK,
            float defaultTopP,
            int seqLen,
            Integer eosId) {
        Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8);
        System.out.println("\n💡 Interactive REPL Mode Activated.");
        System.out.println("Tokenizer in use: " + tokenizer.getClass().getSimpleName());
        System.out.println("Type your text prompt and press ENTER to generate continuation.");
        System.out.println("Type 'exit' or 'quit' to exit.\n");

        while (true) {
            System.out.print("\nPrompt > ");
            if (!scanner.hasNextLine())
                break;
            String input = scanner.nextLine();
            if (input == null || "exit".equalsIgnoreCase(input.trim()) || "quit".equalsIgnoreCase(input.trim())) {
                System.out.println("Exiting PromptRunner. Goodbye!");
                scanner.close();
                break;
            }

            if (input.trim().isEmpty()) {
                input = "Once upon a time";
            }

            generateAndPrint(model, tokenizer, generator, input, defaultMaxTokens, defaultTemp, defaultTopK,
                    defaultTopP, seqLen, eosId);
        }
    }

    private static void generateAndPrint(
            TinyGPT model,
            Tokenizer tokenizer,
            Generator generator,
            String prompt,
            int maxNewTokens,
            float temperature,
            int topK,
            float topP,
            int seqLen,
            Integer eosId) {
        System.out.println("--------------------------------------------------");
        System.out.println("Input Prompt: \"" + prompt + "\"");
        System.out.println("Generating text using " + tokenizer.getClass().getSimpleName() + "...");

        long startTime = System.nanoTime();
        String generated = generator.generate(model, tokenizer, prompt, maxNewTokens, temperature, topK, topP, seqLen,
                eosId);
        long elapsedMs = (System.nanoTime() - startTime) / 1_000_000;

        int newTokensGenerated = Math.max(1, tokenizer.encode(generated).size() - tokenizer.encode(prompt).size());
        float tokPerSec = (newTokensGenerated / (Math.max(1, elapsedMs) / 1000.0f));

        System.out.println("\n✨ Generated Text Result:\n");
        System.out.println(generated);
        System.out.println("\n--------------------------------------------------");
        System.out.printf("Stats: %d ms | %d new tokens | %.1f tokens/sec\n", elapsedMs, newTokensGenerated, tokPerSec);
        System.out.println("--------------------------------------------------");
    }
}
