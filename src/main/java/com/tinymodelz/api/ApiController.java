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
        "hello", "world", "play", "##er", "##ing", "walk", "##s", "##ed", ",", "!", "tiny", "model", "z", "transformer", "attention", "engine"
    );
    
    private final List<String> charVocab = Arrays.asList(
        "h", "e", "l", "o", " ", "w", "r", "d", "!", "p", "a", "y", "i", "n", "g", "t", "m", "z", "c", "s", "v", "u", "f", "k", "b", "x", "q", "j"
    );

    public ApiController() {
        this.trieTokenizer = new TrieTokenizer(trieVocab);
        this.characterTokenizer = new CharacterTokenizer(charVocab);
    }

    @GetMapping("/status")
    public StatusResponse getStatus() {
        return new StatusResponse(
            "TinyModelZ Observability Engine",
            "UP",
            trieTokenizer.getVocabSize(),
            System.getProperty("java.version")
        );
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
        Tensor xIds = new Tensor(idsFloat, new int[]{1, T});
        
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
        for (float val : outData) sum += val;
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
            backwardTimeMs
        );
    }

    public record TokenizeRequest(String text, String type) {}
    public record TokenizeResponse(List<String> tokens, List<Integer> ids) {}
    
    public record DecodeRequest(List<Integer> ids, String type) {}
    public record DecodeResponse(String text) {}
    
    public record ForwardRequest(String text, int embedDim, int numHeads) {}
    public record ForwardResponse(
        int[] shape,
        float[] data,
        float mean,
        float std,
        long forwardTimeMs,
        long backwardTimeMs
    ) {}
    
    public record StatusResponse(String name, String status, int vocabSize, String jdkVersion) {}
}
