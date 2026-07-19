package com.tinymodelz.tokenizer;

import com.tinymodelz.TestReporter;

import java.util.List;

/**
 * <h3>BPETokenizerTest</h3>
 *
 * <p>Unit tests verifying BPE training, pair merging, subword encoding, and decoding.</p>
 */
public class BPETokenizerTest {

    public static void runTests() {
        TestReporter.runTest("BPE corpus auto-training and subword pair merging", () -> {
            String corpus = "the cat sat on the mat and the cat played with the mat";
            BPETokenizer bpe = BPETokenizer.trainFromCorpus(corpus, 10);

            if (bpe.getVocabSize() <= 256) {
                throw new AssertionError("BPE vocab size should expand after pair merges");
            }

            String sample = "the cat sat on the mat";
            List<String> tokens = bpe.tokenize(sample);
            List<Integer> ids = bpe.encode(sample);
            String decoded = bpe.decode(ids);

            if (!decoded.equals(sample)) {
                throw new AssertionError("BPE decoded string '" + decoded + "' does not match original '" + sample + "'");
            }

            TestReporter.logMetric("BPE Vocab Size", String.valueOf(bpe.getVocabSize()));
            TestReporter.logMetric("Sample Tokens", tokens.toString());
        });
    }
}
