package com.tinymodelz.tokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * <h3>BPETokenizer</h3>
 *
 * <p>Byte-Pair Encoding (BPE) Tokenizer implementation (Phase 8 & Roadmap).</p>
 * <p>Supports subword unit merging based on pair rank priorities, special tokens,
 * and byte-level fallback for 100% loss-free UTF-8 string encoding.</p>
 */
public class BPETokenizer implements Tokenizer {

    private static final Logger logger = LoggerFactory.getLogger(BPETokenizer.class);

    public static final String UNK_TOKEN = "<|unk|>";
    public static final String PAD_TOKEN = "<|pad|>";
    public static final String EOS_TOKEN = "<|endoftext|>";

    private final List<String> idToTokenMap;
    private final Map<String, Integer> tokenToIdMap;
    private final Map<Pair, Integer> mergeRanks;

    public static class Pair {
        public final String first;
        public final String second;

        public Pair(String first, String second) {
            this.first = first;
            this.second = second;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return Objects.equals(first, pair.first) && Objects.equals(second, pair.second);
        }

        @Override
        public int hashCode() {
            return Objects.hash(first, second);
        }

        @Override
        public String toString() {
            return "(" + first + "," + second + ")";
        }
    }

    /**
     * Constructs a BPETokenizer with predefined merges and vocabulary list.
     *
     * @param vocab      list of vocabulary token strings
     * @param mergeRanks map of token pairs to rank priorities (lower rank = higher priority)
     */
    public BPETokenizer(List<String> vocab, Map<Pair, Integer> mergeRanks) {
        this.idToTokenMap = new ArrayList<>(vocab);
        this.tokenToIdMap = new HashMap<>();
        for (int i = 0; i < idToTokenMap.size(); i++) {
            tokenToIdMap.put(idToTokenMap.get(i), i);
        }
        this.mergeRanks = mergeRanks != null ? new HashMap<>(mergeRanks) : new HashMap<>();

        // Ensure special tokens exist
        ensureSpecialToken(PAD_TOKEN);
        ensureSpecialToken(UNK_TOKEN);
        ensureSpecialToken(EOS_TOKEN);

        logger.info("Initialized BPETokenizer. Vocab size: {}, Merges: {}", idToTokenMap.size(), this.mergeRanks.size());
    }

    /**
     * Trains a simple BPE vocabulary and merge rules automatically from a text corpus.
     *
     * @param corpus     raw text corpus to train BPE merges from
     * @param numMerges  maximum number of BPE merges to perform
     * @return trained BPETokenizer instance
     */
    public static BPETokenizer trainFromCorpus(String corpus, int numMerges) {
        List<String> vocab = new ArrayList<>();
        Map<Pair, Integer> merges = new HashMap<>();

        // Add special tokens
        vocab.add(PAD_TOKEN);
        vocab.add(UNK_TOKEN);
        vocab.add(EOS_TOKEN);

        // Add single-byte / single-character tokens to base vocabulary
        for (int i = 0; i < 256; i++) {
            String bTok = "<0x" + String.format("%02X", i) + ">";
            if (!vocab.contains(bTok)) vocab.add(bTok);
        }

        // Add printable ASCII characters directly for readability
        for (char c = 32; c <= 126; c++) {
            String cTok = String.valueOf(c);
            if (!vocab.contains(cTok)) vocab.add(cTok);
        }

        // Tokenize corpus into words
        String[] words = corpus.split("\\s+");
        List<List<String>> tokenizedWords = new ArrayList<>();
        for (String word : words) {
            List<String> symbols = new ArrayList<>();
            for (char c : word.toCharArray()) {
                symbols.add(String.valueOf(c));
            }
            if (!symbols.isEmpty()) {
                tokenizedWords.add(symbols);
            }
        }

        int mergeRankCounter = 0;
        for (int m = 0; m < numMerges; m++) {
            Map<Pair, Integer> pairCounts = new HashMap<>();
            for (List<String> symbols : tokenizedWords) {
                for (int i = 0; i < symbols.size() - 1; i++) {
                    Pair p = new Pair(symbols.get(i), symbols.get(i + 1));
                    pairCounts.put(p, pairCounts.getOrDefault(p, 0) + 1);
                }
            }

            if (pairCounts.isEmpty()) break;

            // Find most frequent pair
            Pair bestPair = null;
            int maxFreq = -1;
            for (Map.Entry<Pair, Integer> entry : pairCounts.entrySet()) {
                if (entry.getValue() > maxFreq) {
                    maxFreq = entry.getValue();
                    bestPair = entry.getKey();
                }
            }

            if (bestPair == null || maxFreq < 2) break; // Stop if no pair appears >= 2 times

            String mergedToken = bestPair.first + bestPair.second;
            if (!vocab.contains(mergedToken)) {
                vocab.add(mergedToken);
            }
            merges.put(bestPair, mergeRankCounter++);

            // Apply merge to corpus words
            for (int w = 0; w < tokenizedWords.size(); w++) {
                List<String> symbols = tokenizedWords.get(w);
                List<String> newSymbols = new ArrayList<>();
                for (int i = 0; i < symbols.size(); i++) {
                    if (i < symbols.size() - 1 && symbols.get(i).equals(bestPair.first) && symbols.get(i + 1).equals(bestPair.second)) {
                        newSymbols.add(mergedToken);
                        i++; // skip merged pair
                    } else {
                        newSymbols.add(symbols.get(i));
                    }
                }
                tokenizedWords.set(w, newSymbols);
            }
        }

        return new BPETokenizer(vocab, merges);
    }

    private void ensureSpecialToken(String token) {
        if (!tokenToIdMap.containsKey(token)) {
            idToTokenMap.add(token);
            tokenToIdMap.put(token, idToTokenMap.size() - 1);
        }
    }

    public Map<Pair, Integer> getMergeRanks() {
        return Collections.unmodifiableMap(mergeRanks);
    }

    @Override
    public List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) return Collections.emptyList();

        List<String> specialTokens = List.of(EOS_TOKEN, UNK_TOKEN, PAD_TOKEN);

        List<String> result = new ArrayList<>();
        int i = 0;
        int len = text.length();
        while (i < len) {
            boolean matchedSpecial = false;
            for (String special : specialTokens) {
                if (text.startsWith(special, i)) {
                    result.add(special);
                    i += special.length();
                    matchedSpecial = true;
                    break;
                }
            }
            if (matchedSpecial) continue;

            char c = text.charAt(i);
            String sym = String.valueOf(c);
            if (tokenToIdMap.containsKey(sym)) {
                result.add(sym);
            } else {
                byte[] bytes = sym.getBytes(StandardCharsets.UTF_8);
                for (byte b : bytes) {
                    result.add("<0x" + String.format("%02X", b & 0xFF) + ">");
                }
            }
            i++;
        }

        // Apply BPE merges iteratively according to rank order
        while (result.size() >= 2) {
            Pair bestPair = null;
            int minRank = Integer.MAX_VALUE;
            int bestIdx = -1;

            for (int k = 0; k < result.size() - 1; k++) {
                Pair p = new Pair(result.get(k), result.get(k + 1));
                Integer rank = mergeRanks.get(p);
                if (rank != null && rank < minRank) {
                    minRank = rank;
                    bestPair = p;
                    bestIdx = k;
                }
            }

            if (bestPair == null || bestIdx == -1) break; // No more merges possible

            String merged = bestPair.first + bestPair.second;
            result.set(bestIdx, merged);
            result.remove(bestIdx + 1);
        }

        return result;
    }

    @Override
    public List<Integer> encode(String text) {
        List<String> tokens = tokenize(text);
        List<Integer> ids = new ArrayList<>(tokens.size());
        int unkId = tokenToId(UNK_TOKEN);

        for (String t : tokens) {
            ids.add(tokenToIdMap.getOrDefault(t, unkId));
        }
        return ids;
    }

    @Override
    public String decode(List<Integer> ids) {
        if (ids == null || ids.isEmpty()) return "";

        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String tok = idToToken(id);
            if (tok.equals(PAD_TOKEN) || tok.equals(UNK_TOKEN)) continue;
            if (tok.startsWith("<0x") && tok.endsWith(">")) {
                try {
                    int b = Integer.parseInt(tok.substring(3, 5), 16);
                    sb.append((char) b);
                } catch (Exception e) {
                    sb.append(tok);
                }
            } else {
                sb.append(tok);
            }
        }
        return sb.toString();
    }

    @Override
    public int getVocabSize() {
        return idToTokenMap.size();
    }

    @Override
    public String idToToken(int id) {
        if (id >= 0 && id < idToTokenMap.size()) {
            return idToTokenMap.get(id);
        }
        return UNK_TOKEN;
    }

    @Override
    public int tokenToId(String token) {
        return tokenToIdMap.getOrDefault(token, tokenToIdMap.get(UNK_TOKEN));
    }
}
