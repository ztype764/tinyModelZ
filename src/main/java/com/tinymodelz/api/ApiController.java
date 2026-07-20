package com.tinymodelz.api;

import com.tinymodelz.math.Tensor;
import com.tinymodelz.nn.Embedding;
import com.tinymodelz.nn.TransformerBlock;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import com.tinymodelz.tokenizer.TrieTokenizer;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api")
public class ApiController {

    private final TrieTokenizer trieTokenizer;
    private final CharacterTokenizer characterTokenizer;

    private final List<String> trieVocab = Arrays.asList(
            "hello", "world", "play", "##er", "##ing", "walk", "##s", "##ed", ",", "!", "tiny", "model", "z",
            "transformer", "attention", "engine");

    private final List<String> charVocab = Arrays.asList(
            "h", "e", "l", "o", " ", "w", "r", "d", "!", "p", "a", "y", "i", "n", "g", "t", "m", "z", "c", "s", "v",
            "u", "f", "k", "b", "x", "q", "j");

    private final com.tinymodelz.nn.TinyGPT model;
    private final com.tinymodelz.inference.Generator generator;
    private final Integer eosId;

    public ApiController() {
        this.trieTokenizer = new TrieTokenizer(trieVocab);

        // Load CharacterTokenizer dynamically matching training corpus vocabulary
        CharacterTokenizer loadedCharTok = null;
        String[] possibleDatasetPaths = new String[] {
                "TinyStories-valid-reduced.txt",
                "TinyStories-valid.txt"
        };
        for (String p : possibleDatasetPaths) {
            java.nio.file.Path path = java.nio.file.Path.of(p);
            if (java.nio.file.Files.exists(path)) {
                try {
                    String text = java.nio.file.Files.readString(path, java.nio.charset.StandardCharsets.UTF_8);
                    loadedCharTok = CharacterTokenizer.fromText(text, List.of("<|endoftext|>"));
                    break;
                } catch (Exception ignored) {
                }
            }
        }
        if (loadedCharTok == null) {
            loadedCharTok = new CharacterTokenizer(charVocab);
        }
        this.characterTokenizer = loadedCharTok;

        // Initialize TinyGPT model architecture ONCE at application startup
        int vocabSize = this.characterTokenizer.getVocabSize();
        int seqLen = 64;
        this.model = new com.tinymodelz.nn.TinyGPT(vocabSize, 64, seqLen, 2, 2, 256, 0.1f);
        this.generator = new com.tinymodelz.inference.Generator(42L);
        this.eosId = this.characterTokenizer.tokenToId("<|endoftext|>");

        // Search and load model checkpoint ONCE during startup
        String[] possibleCheckpoints = new String[] {
                "checkpoints/tinystories/best_checkpoint",
                "checkpoints/tinystories/epoch_5",
                "checkpoints/tinystories/epoch_4",
                "checkpoints/tinystories/epoch_3",
                "checkpoints/tinystories/epoch_2",
                "checkpoints/tinystories/epoch_1",
                "checkpoints/best_checkpoint",
                "checkpoints/epoch_5"
        };
        java.io.File selectedChk = null;
        for (String cPath : possibleCheckpoints) {
            java.io.File f = new java.io.File(cPath);
            if (f.exists() && f.isDirectory() && f.list() != null && f.list().length > 0) {
                selectedChk = f;
                break;
            }
        }
        if (selectedChk != null) {
            try {
                com.tinymodelz.train.Checkpoint.loadCheckpoint(this.model, selectedChk);
                org.slf4j.LoggerFactory.getLogger(ApiController.class).info("✅ Loaded checkpoint ONCE into memory from: {}",
                        selectedChk.getAbsolutePath());
            } catch (Exception e) {
                org.slf4j.LoggerFactory.getLogger(ApiController.class).error("Failed to load checkpoint from {}: {}",
                        selectedChk.getAbsolutePath(), e.getMessage());
            }
        } else {
            org.slf4j.LoggerFactory.getLogger(ApiController.class)
                    .warn("No saved checkpoint directory found! Using initialized model weights.");
        }
    }

    @GetMapping("/status")
    public StatusResponse getStatus() {
        return new StatusResponse(
                "TinyModelZ Observability Engine",
                "UP",
                trieTokenizer.getVocabSize(),
                System.getProperty("java.version"));
    }

    @PostMapping("/tokenizer/tokenize")
    public TokenizeResponse tokenize(@RequestBody TokenizeRequest request) {
        String text = request.text() != null ? request.text() : "";
        String type = request.type() != null ? request.type().toLowerCase() : "wordpiece";

        List<String> tokens;
        List<Integer> ids;

        if ("character".equals(type)) {
            tokens = characterTokenizer.tokenize(text);
            ids = characterTokenizer.encode(text);
        } else {
            tokens = trieTokenizer.tokenize(text);
            ids = trieTokenizer.encode(text);
        }

        return new TokenizeResponse(tokens, ids);
    }

    @PostMapping("/tokenizer/decode")
    public DecodeResponse decode(@RequestBody DecodeRequest request) {
        List<Integer> idsList = request.ids() != null ? request.ids() : Collections.emptyList();
        String type = request.type() != null ? request.type().toLowerCase() : "wordpiece";

        String text;
        if ("character".equals(type)) {
            text = characterTokenizer.decode(idsList);
        } else {
            text = trieTokenizer.decode(idsList);
        }

        return new DecodeResponse(text);
    }

    @PostMapping("/transformer/forward")
    public ForwardResponse forward(@RequestBody ForwardRequest request) {
        int embedDim = request.embedDim() > 0 ? request.embedDim() : 8;
        int numHeads = request.numHeads() > 0 ? request.numHeads() : 2;
        String text = request.text() != null ? request.text() : "hello world!";

        // 1. Tokenize input text to get sequence
        List<Integer> tokenIds = trieTokenizer.encode(text);
        int T = tokenIds.size();
        if (T == 0) {
            throw new IllegalArgumentException("Input text tokenized to an empty sequence.");
        }

        long start = System.nanoTime();

        // 2. Setup embedding layer and embed tokens
        int vocabSize = trieTokenizer.getVocabSize();
        Embedding embedding = new Embedding(vocabSize, embedDim);

        float[] idsFloat = new float[T];
        for (int i = 0; i < T; i++) {
            idsFloat[i] = tokenIds.get(i);
        }
        Tensor xIds = new Tensor(idsFloat, new int[] { 1, T });

        // Forward through embedding
        Tensor embedded = embedding.forward(xIds); // [1, T, embedDim]
        embedded.setRequiresGrad(true);

        // 3. Forward through Transformer Block
        TransformerBlock block = new TransformerBlock(embedDim, numHeads, 0.0f);
        block.train();

        // Causal mask
        Tensor mask = com.tinymodelz.nn.MultiHeadAttention.createCausalMask(T);

        Tensor out = block.forward(embedded, mask);
        long forwardEnd = System.nanoTime();

        // 4. Backward pass
        Tensor loss = out.sum();
        loss.backward();
        long backwardEnd = System.nanoTime();

        long forwardTimeMs = (forwardEnd - start) / 1_000_000;
        long backwardTimeMs = (backwardEnd - forwardEnd) / 1_000_000;

        // Collect stats
        float[] outData = out.toContiguous().getData();
        float sum = 0.0f;
        for (float val : outData)
            sum += val;
        float mean = sum / outData.length;

        float variance = 0.0f;
        for (float val : outData) {
            variance += (val - mean) * (val - mean);
        }
        float std = (float) Math.sqrt(variance / outData.length);

        return new ForwardResponse(
                out.getShape(),
                outData,
                mean,
                std,
                forwardTimeMs,
                backwardTimeMs);
    }

    @PostMapping("/generate")
    public GenerateResponse generate(@RequestBody GenerateRequest request) {
        String prompt = (request.prompt() != null && !request.prompt().trim().isEmpty()) ? request.prompt()
                : "Once upon a time";
        int maxNewTokens = request.maxNewTokens() > 0 ? request.maxNewTokens() : 40;
        float temperature = request.temperature() >= 0 ? request.temperature() : 0.7f;
        int topK = request.topK() > 0 ? request.topK() : 40;
        float topP = (request.topP() > 0 && request.topP() <= 1.0f) ? request.topP() : 0.9f;

        long start = System.nanoTime();
        // Reuse in-memory loaded singleton model instance
        String result = generator.generate(model, characterTokenizer, prompt, maxNewTokens, temperature, topK, topP,
                64, eosId);
        long latencyMs = (System.nanoTime() - start) / 1_000_000;

        int newTokens = Math.max(1,
                characterTokenizer.encode(result).size() - characterTokenizer.encode(prompt).size());
        float tokPerSec = newTokens / (Math.max(1, latencyMs) / 1000.0f);

        return new GenerateResponse(result, prompt, newTokens, latencyMs, tokPerSec);
    }

    public record TokenizeRequest(String text, String type) {
    }

    public record TokenizeResponse(List<String> tokens, List<Integer> ids) {
    }

    public record DecodeRequest(List<Integer> ids, String type) {
    }

    public record DecodeResponse(String text) {
    }

    public record ForwardRequest(String text, int embedDim, int numHeads) {
    }

    public record ForwardResponse(
            int[] shape,
            float[] data,
            float mean,
            float std,
            long forwardTimeMs,
            long backwardTimeMs) {
    }

    public record GenerateRequest(
            String prompt,
            int maxNewTokens,
            float temperature,
            int topK,
            float topP) {
    }

    public record GenerateResponse(
            String generatedText,
            String prompt,
            int tokensGenerated,
            long latencyMs,
            float tokensPerSec) {
    }

    public record StatusResponse(String name, String status, int vocabSize, String jdkVersion) {
    }
}
