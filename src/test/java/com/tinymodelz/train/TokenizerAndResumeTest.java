package com.tinymodelz.train;

import com.tinymodelz.nn.CrossEntropyLoss;
import com.tinymodelz.nn.TinyGPT;
import com.tinymodelz.tokenizer.BPETokenizer;
import com.tinymodelz.tokenizer.CharacterTokenizer;
import com.tinymodelz.tokenizer.Tokenizer;
import com.tinymodelz.tokenizer.TokenizerFactory;
import com.tinymodelz.tokenizer.TokenizerFactory.TokenizerType;
import com.tinymodelz.tokenizer.TrieTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class TokenizerAndResumeTest {

    @Test
    public void testTokenizerFactoryAndSerialization(@TempDir Path tempDir) throws IOException {
        String corpus = "Once upon a time there was a dog named Max. Max loved playing with his ball.";
        List<String> customTokens = List.of("<|endoftext|>");

        // 1. CharacterTokenizer
        Tokenizer charTok = TokenizerFactory.createTokenizer(TokenizerType.CHARACTER, corpus, customTokens);
        assertTrue(charTok instanceof CharacterTokenizer);
        File charDir = tempDir.resolve("char_tok").toFile();
        TokenizerFactory.saveTokenizer(charTok, TokenizerType.CHARACTER, charDir);
        Tokenizer loadedCharTok = TokenizerFactory.loadTokenizer(charDir);
        assertNotNull(loadedCharTok);
        assertEquals(charTok.getVocabSize(), loadedCharTok.getVocabSize());

        // 2. BPETokenizer
        Tokenizer bpeTok = TokenizerFactory.createTokenizer(TokenizerType.BPE, corpus, customTokens);
        assertTrue(bpeTok instanceof BPETokenizer);
        File bpeDir = tempDir.resolve("bpe_tok").toFile();
        TokenizerFactory.saveTokenizer(bpeTok, TokenizerType.BPE, bpeDir);
        Tokenizer loadedBpeTok = TokenizerFactory.loadTokenizer(bpeDir);
        assertNotNull(loadedBpeTok);
        assertTrue(loadedBpeTok instanceof BPETokenizer);
        assertEquals(bpeTok.getVocabSize(), loadedBpeTok.getVocabSize());

        // 3. TrieTokenizer
        Tokenizer trieTok = TokenizerFactory.createTokenizer(TokenizerType.TRIE, corpus, customTokens);
        assertTrue(trieTok instanceof TrieTokenizer);
        File trieDir = tempDir.resolve("trie_tok").toFile();
        TokenizerFactory.saveTokenizer(trieTok, TokenizerType.TRIE, trieDir);
        Tokenizer loadedTrieTok = TokenizerFactory.loadTokenizer(trieDir);
        assertNotNull(loadedTrieTok);
        assertTrue(loadedTrieTok instanceof TrieTokenizer);
        assertEquals(trieTok.getVocabSize(), loadedTrieTok.getVocabSize());
    }

    @Test
    public void testTrainingResumeFromEpoch2(@TempDir Path tempDir) throws IOException {
        String corpus = "Once upon a time there was a little dog named Max. Max loved to play in the park with his ball. <|endoftext|>\n";
        Tokenizer tokenizer = TokenizerFactory.createTokenizer(TokenizerType.BPE, corpus, List.of("<|endoftext|>"));
        TextDataset dataset = new TextDataset(corpus, tokenizer);
        DataLoader loader = new DataLoader(dataset, 2, 16, true);

        TinyGPT model = new TinyGPT(tokenizer.getVocabSize(), 32, 16, 1, 2, 64, 0.1f);
        AdamW optimizer = new AdamW(model.getParameters(), 0.001f);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Trainer trainer = new Trainer(model, tokenizer, optimizer, lossFn);

        // Train epoch 1
        File runDir = tempDir.toFile();
        trainer.train(loader, null, 1, 1, runDir, List.of("Once upon"), 10, 10);
        File epoch1Dir = new File(runDir, "epoch_1");
        assertTrue(new File(epoch1Dir, "param_0.tmat").exists());

        // Now simulate resuming from epoch 2
        TinyGPT resumedModel = new TinyGPT(tokenizer.getVocabSize(), 32, 16, 1, 2, 64, 0.1f);
        AdamW resumedOptimizer = new AdamW(resumedModel.getParameters(), 0.001f);
        Checkpoint.CheckpointState state = Checkpoint.loadCheckpoint(resumedModel, resumedOptimizer, epoch1Dir);

        assertEquals(1, state.epoch);
        Trainer resumedTrainer = new Trainer(resumedModel, tokenizer, resumedOptimizer, lossFn);

        // Resume training for epoch 2 to 3
        resumedTrainer.train(loader, null, state.epoch + 1, 3, runDir, List.of("Once upon"), 10, 10);

        assertTrue(new File(runDir, "epoch_2").exists());
        assertTrue(new File(runDir, "epoch_3").exists());
    }

    public static void runTests() {
        TokenizerAndResumeTest test = new TokenizerAndResumeTest();
        try {
            com.tinymodelz.TestReporter
                    .runTest("Multi-tokenizer creation and checkpoint serialization (Character, BPE, Trie)", () -> {
                        try {
                            Path temp = java.nio.file.Files.createTempDirectory("tok_test");
                            test.testTokenizerFactoryAndSerialization(temp);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });
            com.tinymodelz.TestReporter.runTest("Resuming training pipeline from epoch 2 checkpoint", () -> {
                try {
                    Path temp = java.nio.file.Files.createTempDirectory("resume_test");
                    test.testTrainingResumeFromEpoch2(temp);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
