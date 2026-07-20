package com.tinymodelz.api;

import com.tinymodelz.TestReporter;

/**
 * <h3>ApiControllerTest</h3>
 *
 * <p>Unit test for verifying ApiController prompt generation with loaded checkpoints.</p>
 */
public class ApiControllerTest {

    public static void runTests() {
        TestReporter.runTest("ApiController generation with dataset vocabulary and saved checkpoint", () -> {
            ApiController controller = new ApiController();
            ApiController.GenerateRequest request = new ApiController.GenerateRequest(
                "Once upon a time",
                30,
                0.7f,
                40,
                0.9f
            );
            ApiController.GenerateResponse response = controller.generate(request);

            if (response.generatedText() == null || response.generatedText().trim().isEmpty()) {
                throw new AssertionError("Generated text should not be empty");
            }
            if (!response.generatedText().startsWith("Once")) {
                throw new AssertionError("Generated text should start with the prompt: " + response.generatedText());
            }

            TestReporter.logMetric("Prompt", response.prompt());
            TestReporter.logMetric("Generated Output", response.generatedText().replace("\n", "\\n"));
            TestReporter.logMetric("Latency (ms)", String.valueOf(response.latencyMs()));
            TestReporter.logMetric("Throughput (tok/s)", String.format("%.1f", response.tokensPerSec()));
        });
    }
}
