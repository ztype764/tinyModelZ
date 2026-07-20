package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/**
 * <h3>TokenizerFactory</h3>
 *
 * <p>Unified factory and serialization utility for managing all tokenizer types
 * ({@link CharacterTokenizer}, {@link BPETokenizer}, {@link TrieTokenizer}).</p>
 */
public class TokenizerFactory {

    private static final Logger logger = LoggerFactory.getLogger(TokenizerFactory.class);

    public enum TokenizerType {
        CHARACTER,
        BPE,
        TRIE;

        public static TokenizerType fromString(String type) {
            if (type == null) return CHARACTER;
            String t = type.trim().toLowerCase();
            if (t.equals("bpe")) return BPE;
            if (t.equals("trie") || t.equals("wordpiece") || t.equals("maxmatch")) return TRIE;
            return CHARACTER;
        }

        public String toSubfolderName() {
            switch (this) {
                case BPE: return "bpe";
                case TRIE: return "trie";
                case CHARACTER:
                default: return "char";
            }
        }
    }

    /**
     * Constructs a Tokenizer instance based on requested type and text corpus.
     *
     * @param type         tokenizer algorithm type
     * @param corpusText   raw dataset text
     * @param customTokens custom special tokens (e.g. "&lt;|endoftext|&gt;")
     * @return constructed Tokenizer
     */
    public static Tokenizer createTokenizer(TokenizerType type, String corpusText, List<String> customTokens) {
        if (customTokens == null) {
            customTokens = List.of("<|endoftext|>");
        }
        switch (type) {
            case BPE:
                logger.info("Training BPETokenizer on text corpus (numMerges=300)...");
                return BPETokenizer.trainFromCorpus(corpusText, 300);
            case TRIE:
                logger.info("Constructing TrieTokenizer (WordPiece/MaxMatch) from text corpus...");
                return buildTrieTokenizerFromCorpus(corpusText, customTokens);
            case CHARACTER:
            default:
                logger.info("Constructing CharacterTokenizer from text corpus...");
                return CharacterTokenizer.fromText(corpusText, customTokens);
        }
    }

    /**
     * Builds vocabulary for TrieTokenizer from raw corpus text.
     */
    private static TrieTokenizer buildTrieTokenizerFromCorpus(String corpus, List<String> customTokens) {
        List<String> vocab = new ArrayList<>();
        vocab.add("[PAD]");
        vocab.add("[UNK]");
        vocab.add("[CLS]");
        vocab.add("[SEP]");
        vocab.add("[MASK]");
        if (customTokens != null) {
            for (String c : customTokens) {
                if (!vocab.contains(c)) vocab.add(c);
            }
        }

        String[] rawWords = corpus.split("\\s+");
        Map<String, Integer> wordFreq = new HashMap<>();
        for (String w : rawWords) {
            String cleaned = w.trim();
            if (!cleaned.isEmpty()) {
                wordFreq.put(cleaned, wordFreq.getOrDefault(cleaned, 0) + 1);
            }
        }

        // Add words appearing at least once
        for (Map.Entry<String, Integer> entry : wordFreq.entrySet()) {
            if (!vocab.contains(entry.getKey())) {
                vocab.add(entry.getKey());
            }
        }

        // Add single characters as fallback
        for (int i = 0; i < corpus.length(); i++) {
            String ch = String.valueOf(corpus.charAt(i));
            if (!vocab.contains(ch)) {
                vocab.add(ch);
            }
        }

        return new TrieTokenizer(vocab);
    }

    /**
     * Saves tokenizer metadata, vocabulary, and BPE merges to the specified directory.
     *
     * @param tokenizer tokenizer instance to save
     * @param type      tokenizer algorithm type
     * @param directory directory where tokenizer files will be saved
     * @throws IOException if saving fails
     */
    public static void saveTokenizer(Tokenizer tokenizer, TokenizerType type, File directory) throws IOException {
        if (!directory.exists() && !directory.mkdirs()) {
            throw new IOException("Could not create directory for tokenizer: " + directory.getAbsolutePath());
        }

        Properties props = new Properties();
        props.setProperty("tokenizer.type", type.name().toLowerCase());
        props.setProperty("vocab.size", String.valueOf(tokenizer.getVocabSize()));

        File metaFile = new File(directory, "tokenizer_config.properties");
        try (FileOutputStream out = new FileOutputStream(metaFile)) {
            props.store(out, "TinyModelZ Tokenizer Configuration");
        }

        // Save vocabulary
        List<String> vocabLines = new ArrayList<>();
        for (int i = 0; i < tokenizer.getVocabSize(); i++) {
            String tok = tokenizer.idToToken(i);
            // Escape newlines for line storage
            tok = tok.replace("\n", "\\n").replace("\r", "\\r");
            vocabLines.add(tok);
        }
        Files.write(new File(directory, "vocab.txt").toPath(), vocabLines, StandardCharsets.UTF_8);

        // If BPE, save merge rules
        if (tokenizer instanceof BPETokenizer) {
            BPETokenizer bpe = (BPETokenizer) tokenizer;
            List<String> mergeLines = new ArrayList<>();
            for (Map.Entry<BPETokenizer.Pair, Integer> entry : bpe.getMergeRanks().entrySet()) {
                String first = entry.getKey().first.replace("\n", "\\n").replace("\r", "\\r");
                String second = entry.getKey().second.replace("\n", "\\n").replace("\r", "\\r");
                mergeLines.add(first + " " + second + " " + entry.getValue());
            }
            Files.write(new File(directory, "bpe_merges.txt").toPath(), mergeLines, StandardCharsets.UTF_8);
        }

        logger.info("Saved tokenizer metadata to: {}", directory.getAbsolutePath());
    }

    /**
     * Loads a tokenizer from a saved directory. If saved config is missing, returns null or fallback.
     *
     * @param directory target directory containing tokenizer metadata
     * @return loaded Tokenizer, or null if no valid configuration exists
     */
    public static Tokenizer loadTokenizer(File directory) {
        if (directory == null || !directory.exists()) return null;

        File metaFile = new File(directory, "tokenizer_config.properties");
        File vocabFile = new File(directory, "vocab.txt");

        // If metaFile is in parent run directory, check parent
        if (!metaFile.exists() && directory.getParentFile() != null) {
            File parentMeta = new File(directory.getParentFile(), "tokenizer_config.properties");
            File parentVocab = new File(directory.getParentFile(), "vocab.txt");
            if (parentMeta.exists()) {
                metaFile = parentMeta;
                vocabFile = parentVocab;
                directory = directory.getParentFile();
            }
        }

        if (!metaFile.exists() || !vocabFile.exists()) {
            return null;
        }

        try {
            Properties props = new Properties();
            try (FileInputStream in = new FileInputStream(metaFile)) {
                props.load(in);
            }

            TokenizerType type = TokenizerType.fromString(props.getProperty("tokenizer.type", "character"));
            List<String> rawLines = Files.readAllLines(vocabFile.toPath(), StandardCharsets.UTF_8);
            List<String> vocab = new ArrayList<>();
            for (String line : rawLines) {
                vocab.add(line.replace("\\n", "\n").replace("\\r", "\r"));
            }

            if (type == TokenizerType.BPE) {
                Map<BPETokenizer.Pair, Integer> merges = new HashMap<>();
                File mergesFile = new File(directory, "bpe_merges.txt");
                if (mergesFile.exists()) {
                    List<String> mLines = Files.readAllLines(mergesFile.toPath(), StandardCharsets.UTF_8);
                    for (String line : mLines) {
                        String[] parts = line.split(" ");
                        if (parts.length >= 3) {
                            String f = parts[0].replace("\\n", "\n").replace("\\r", "\r");
                            String s = parts[1].replace("\\n", "\n").replace("\\r", "\r");
                            int rank = Integer.parseInt(parts[2]);
                            merges.put(new BPETokenizer.Pair(f, s), rank);
                        }
                    }
                }
                return new BPETokenizer(vocab, merges);
            } else if (type == TokenizerType.TRIE) {
                return new TrieTokenizer(vocab);
            } else {
                return new CharacterTokenizer(vocab);
            }
        } catch (Exception e) {
            logger.warn("Could not load tokenizer from {}: {}", directory.getAbsolutePath(), e.getMessage());
            return null;
        }
    }
}
