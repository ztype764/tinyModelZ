package com.tinymodelz;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * <h3>TestReporter</h3>
 * 
 * <p>A custom test recording and reporting framework designed to collect execution metrics
 * and generate a high-fidelity HTML5 test report dashboard.</p>
 */
public class TestReporter {

    private static final Logger logger = LoggerFactory.getLogger(TestReporter.class);

    private static final List<Suite> suites = new ArrayList<>();
    private static Suite currentSuite = null;
    private static TestCase currentTestCase = null;
    private static boolean anyFailed = false;

    public static class Metric {
        public String key;
        public String value;

        public Metric(String key, String value) {
            this.key = key;
            this.value = value;
        }
    }

    public static class TestCase {
        public String name;
        public String status; // "PASSED" or "FAILED"
        public long durationMs;
        public String errorType = "";
        public String errorMessage = "";
        public String stackTrace = "";
        public List<Metric> metrics = new ArrayList<>();
    }

    public static class Suite {
        public String name;
        public List<TestCase> tests = new ArrayList<>();
        public long durationMs;
        public boolean hasFailures = false;

        public int getPassedCount() {
            int count = 0;
            for (TestCase tc : tests) {
                if ("PASSED".equals(tc.status))
                    count++;
            }
            return count;
        }

        public int getFailedCount() {
            int count = 0;
            for (TestCase tc : tests) {
                if ("FAILED".equals(tc.status))
                    count++;
            }
            return count;
        }
    }

    /**
     * Starts a new test suite block.
     * 
     * @param name the name of the suite
     */
    public static void startSuite(String name) {
        currentSuite = new Suite();
        currentSuite.name = name;
        logger.info("Test Suite Started: {}", name);
    }

    /**
     * Ends the current test suite block and aggregates duration.
     */
    public static void endSuite() {
        if (currentSuite != null) {
            long suiteDuration = 0;
            for (TestCase tc : currentSuite.tests) {
                suiteDuration += tc.durationMs;
                if ("FAILED".equals(tc.status)) {
                    currentSuite.hasFailures = true;
                }
            }
            currentSuite.durationMs = suiteDuration;
            suites.add(currentSuite);
            logger.info("Test Suite Finished: {} (Passed: {}, Failed: {}, Time: {}ms)",
                    currentSuite.name, currentSuite.getPassedCount(), currentSuite.getFailedCount(), suiteDuration);
            currentSuite = null;
        }
    }

    /**
     * Logs a metadata key-value metric to be visualized inside the current test details card.
     * 
     * @param key the label of the metric
     * @param value the value to be displayed
     */
    public static void logMetric(String key, Object value) {
        if (currentTestCase != null) {
            currentTestCase.metrics.add(new Metric(key, value == null ? "null" : value.toString()));
        }
    }

    /**
     * Executes a unit test block, capturing execution time, pass status, and any errors.
     * 
     * @param testName     the descriptive name of the test
     * @param testRunnable the runnable block containing test assertions
     */
    public static void runTest(String testName, Runnable testRunnable) {
        if (currentSuite == null) {
            startSuite("Default Suite");
        }

        TestCase tc = new TestCase();
        tc.name = testName;
        currentTestCase = tc;
        long start = System.currentTimeMillis();

        try {
            logger.info("  Running test: {}...", testName);
            testRunnable.run();
            tc.status = "PASSED";
            logger.info("  [PASS] {}", testName);
        } catch (Throwable t) {
            tc.status = "FAILED";
            anyFailed = true;
            tc.errorType = t.getClass().getName();
            tc.errorMessage = t.getMessage() != null ? t.getMessage() : t.toString();

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            t.printStackTrace(pw);
            tc.stackTrace = sw.toString();

            logger.error("  [FAIL] {}: {}", testName, tc.errorMessage);
        } finally {
            tc.durationMs = System.currentTimeMillis() - start;
            currentSuite.tests.add(tc);
            currentTestCase = null;
        }
    }

    /**
     * Checks if any test run in any suite has failed.
     * 
     * @return true if there are failures, false otherwise
     */
    public static boolean hasFailures() {
        return anyFailed;
    }

    /**
     * Generates a beautifully styled, responsive, and interactive HTML5 test report dashboard.
     * 
     * @param outputPath the file destination path for the HTML report
     */
    public static void generateReport(String outputPath) {
        logger.info("Generating HTML test visualization report at: {}", outputPath);

        int totalSuites = suites.size();
        int totalTests = 0;
        int passedTests = 0;
        int failedTests = 0;
        long totalDuration = 0;

        for (Suite s : suites) {
            totalTests += s.tests.size();
            passedTests += s.getPassedCount();
            failedTests += s.getFailedCount();
            totalDuration += s.durationMs;
        }

        float passRate = totalTests > 0 ? ((float) passedTests / totalTests) * 100.0f : 0.0f;
        int circumference = 440; // Circ = 2 * PI * r (r=70)
        int strokeOffset = totalTests > 0 ? (int) (circumference * (1.0f - ((float) passedTests / totalTests)))
                : circumference;

        StringBuilder html = new StringBuilder();
        html.append("<!DOCTYPE html>\n")
                .append("<html lang=\"en\">\n")
                .append("<head>\n")
                .append("  <meta charset=\"UTF-8\">\n")
                .append("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
                .append("  <title>TinyModelZ Test Report</title>\n")
                .append("  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n")
                .append("  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n")
                .append("  <link href=\"https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap\" rel=\"stylesheet\">\n")
                .append("  <style>\n")
                .append("    :root {\n")
                .append("      --bg: #070a13;\n")
                .append("      --card-bg: #0e1322;\n")
                .append("      --card-border: #1e293b;\n")
                .append("      --text-main: #f8fafc;\n")
                .append("      --text-muted: #64748b;\n")
                .append("      --accent: #6366f1;\n")
                .append("      --accent-glow: rgba(99, 102, 241, 0.15);\n")
                .append("      --success: #10b981;\n")
                .append("      --success-glow: rgba(16, 185, 129, 0.15);\n")
                .append("      --failure: #ef4444;\n")
                .append("      --failure-glow: rgba(239, 68, 68, 0.15);\n")
                .append("      --metric-bg: #131b2e;\n")
                .append("    }\n")
                .append("    * { box-sizing: border-box; margin: 0; padding: 0; }\n")
                .append("    body {\n")
                .append("      background-color: var(--bg);\n")
                .append("      color: var(--text-main);\n")
                .append("      font-family: 'Plus Jakarta Sans', sans-serif;\n")
                .append("      padding: 2.5rem 1.5rem;\n")
                .append("      line-height: 1.5;\n")
                .append("    }\n")
                .append("    .container {\n")
                .append("      max-width: 1100px;\n")
                .append("      margin: 0 auto;\n")
                .append("    }\n")
                .append("    header {\n")
                .append("      display: flex;\n")
                .append("      justify-content: space-between;\n")
                .append("      align-items: center;\n")
                .append("      margin-bottom: 2.5rem;\n")
                .append("      border-bottom: 1px solid var(--card-border);\n")
                .append("      padding-bottom: 1.5rem;\n")
                .append("    }\n")
                .append("    h1 {\n")
                .append("      font-size: 2rem;\n")
                .append("      font-weight: 700;\n")
                .append("      background: linear-gradient(135deg, #a5b4fc, #6366f1);\n")
                .append("      -webkit-background-clip: text;\n")
                .append("      -webkit-text-fill-color: transparent;\n")
                .append("    }\n")
                .append("    .badge {\n")
                .append("      font-size: 0.85rem;\n")
                .append("      font-weight: 600;\n")
                .append("      padding: 0.35rem 0.85rem;\n")
                .append("      border-radius: 9999px;\n")
                .append("      display: inline-block;\n")
                .append("    }\n")
                .append("    .badge-success {\n")
                .append("      background-color: var(--success-glow);\n")
                .append("      color: var(--success);\n")
                .append("      border: 1px solid rgba(16, 185, 129, 0.3);\n")
                .append("    }\n")
                .append("    .badge-failure {\n")
                .append("      background-color: var(--failure-glow);\n")
                .append("      color: var(--failure);\n")
                .append("      border: 1px solid rgba(239, 68, 68, 0.3);\n")
                .append("    }\n")
                .append("    .dashboard-grid {\n")
                .append("      display: grid;\n")
                .append("      grid-template-columns: 1.5fr 1fr;\n")
                .append("      gap: 2rem;\n")
                .append("      margin-bottom: 2.5rem;\n")
                .append("    }\n")
                .append("    @media (max-width: 768px) {\n")
                .append("      .dashboard-grid { grid-template-columns: 1fr; }\n")
                .append("    }\n")
                .append("    .card {\n" )
                .append("      background-color: var(--card-bg);\n")
                .append("      border: 1px solid var(--card-border);\n")
                .append("      border-radius: 1rem;\n")
                .append("      padding: 1.75rem;\n")
                .append("      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);\n")
                .append("    }\n")
                .append("    .stats-layout {\n")
                .append("      display: grid;\n")
                .append("      grid-template-columns: repeat(2, 1fr);\n")
                .append("      gap: 1.25rem;\n")
                .append("    }\n")
                .append("    .stat-box {\n")
                .append("      background-color: rgba(255, 255, 255, 0.02);\n")
                .append("      border: 1px solid rgba(255, 255, 255, 0.04);\n")
                .append("      border-radius: 0.75rem;\n")
                .append("      padding: 1rem 1.25rem;\n")
                .append("    }\n")
                .append("    .stat-label {\n")
                .append("      font-size: 0.8rem;\n")
                .append("      color: var(--text-muted);\n")
                .append("      text-transform: uppercase;\n")
                .append("      letter-spacing: 0.05em;\n")
                .append("      margin-bottom: 0.25rem;\n")
                .append("    }\n")
                .append("    .stat-value {\n")
                .append("      font-size: 1.75rem;\n")
                .append("      font-weight: 700;\n")
                .append("    }\n")
                .append("    .chart-card {\n")
                .append("      display: flex;\n")
                .append("      flex-direction: column;\n")
                .append("      align-items: center;\n")
                .append("      justify-content: center;\n")
                .append("    }\n")
                .append("    .chart-container {\n")
                .append("      position: relative;\n")
                .append("      width: 160px;\n")
                .append("      height: 160px;\n")
                .append("    }\n")
                .append("    .chart-svg {\n")
                .append("      transform: rotate(-90deg);\n")
                .append("      width: 100%;\n")
                .append("      height: 100%;\n")
                .append("    }\n")
                .append("    .chart-bg { fill: none; stroke: var(--card-border); stroke-width: 16; }\n")
                .append("    .chart-fill {\n")
                .append("      fill: none;\n")
                .append("      stroke: var(--success);\n")
                .append("      stroke-width: 16;\n")
                .append("      stroke-dasharray: 440;\n")
                .append("      stroke-dashoffset: ").append(strokeOffset).append(";\n")
                .append("      stroke-linecap: round;\n")
                .append("      transition: stroke-dashoffset 1s ease-in-out;\n")
                .append("    }\n")
                .append("    .chart-center {\n")
                .append("      position: absolute;\n")
                .append("      top: 50%;\n")
                .append("      left: 50%;\n")
                .append("      transform: translate(-50%, -50%);\n")
                .append("      font-size: 1.5rem;\n")
                .append("      font-weight: 700;\n")
                .append("    }\n")
                .append("    .chart-title {\n")
                .append("      margin-top: 1rem;\n")
                .append("      font-size: 0.95rem;\n")
                .append("      font-weight: 600;\n")
                .append("      color: var(--text-muted);\n")
                .append("    }\n")
                .append("    .controls-container {\n")
                .append("      display: flex;\n")
                .append("      justify-content: space-between;\n")
                .append("      align-items: center;\n")
                .append("      gap: 1rem;\n")
                .append("      margin-bottom: 2rem;\n")
                .append("      flex-wrap: wrap;\n")
                .append("    }\n")
                .append("    .filter-tabs {\n")
                .append("      display: flex;\n")
                .append("      background-color: var(--card-bg);\n")
                .append("      border: 1px solid var(--card-border);\n")
                .append("      padding: 0.25rem;\n")
                .append("      border-radius: 0.5rem;\n")
                .append("    }\n")
                .append("    .filter-btn {\n")
                .append("      background: none;\n")
                .append("      border: none;\n")
                .append("      color: var(--text-muted);\n")
                .append("      padding: 0.5rem 1rem;\n")
                .append("      font-size: 0.875rem;\n")
                .append("      font-weight: 500;\n")
                .append("      cursor: pointer;\n")
                .append("      border-radius: 0.375rem;\n")
                .append("      transition: all 0.2s ease;\n")
                .append("    }\n")
                .append("    .filter-btn.active {\n")
                .append("      background-color: var(--accent);\n")
                .append("      color: #fff;\n")
                .append("    }\n")
                .append("    .search-input {\n")
                .append("      background-color: var(--card-bg);\n")
                .append("      border: 1px solid var(--card-border);\n")
                .append("      color: var(--text-main);\n")
                .append("      padding: 0.5rem 1rem;\n")
                .append("      border-radius: 0.5rem;\n")
                .append("      font-size: 0.875rem;\n")
                .append("      outline: none;\n")
                .append("      width: 280px;\n")
                .append("      transition: border-color 0.2s;\n")
                .append("    }\n")
                .append("    .search-input:focus {\n")
                .append("      border-color: var(--accent);\n")
                .append("    }\n")
                .append("    .suite-section {\n")
                .append("      margin-bottom: 2rem;\n")
                .append("    }\n")
                .append("    .suite-header {\n")
                .append("      display: flex;\n")
                .append("      justify-content: space-between;\n")
                .append("      align-items: center;\n")
                .append("      padding: 0.75rem 0.5rem;\n")
                .append("      border-bottom: 2px solid var(--card-border);\n")
                .append("      margin-bottom: 1rem;\n")
                .append("    }\n")
                .append("    .suite-title {\n")
                .append("      font-size: 1.2rem;\n")
                .append("      font-weight: 600;\n")
                .append("      color: #a5b4fc;\n")
                .append("    }\n")
                .append("    .suite-meta {\n")
                .append("      font-size: 0.85rem;\n")
                .append("      color: var(--text-muted);\n")
                .append("    }\n")
                .append("    .test-list {\n")
                .append("      display: flex;\n")
                .append("      flex-direction: column;\n")
                .append("      gap: 0.75rem;\n")
                .append("    }\n")
                .append("    details {\n")
                .append("      background-color: var(--card-bg);\n")
                .append("      border: 1px solid var(--card-border);\n")
                .append("      border-radius: 0.5rem;\n")
                .append("      overflow: hidden;\n")
                .append("      transition: all 0.2s ease;\n")
                .append("    }\n")
                .append("    details[open] {\n")
                .append("      border-color: rgba(99, 102, 241, 0.4);\n")
                .append("      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.05);\n")
                .append("    }\n")
                .append("    summary {\n")
                .append("      padding: 1rem 1.25rem;\n")
                .append("      cursor: pointer;\n")
                .append("      display: flex;\n")
                .append("      justify-content: space-between;\n")
                .append("      align-items: center;\n")
                .append("      font-weight: 500;\n")
                .append("      user-select: none;\n")
                .append("    }\n")
                .append("    summary::-webkit-details-marker { display: none; }\n")
                .append("    summary::after {\n")
                .append("      content: '►';\n")
                .append("      font-size: 0.75rem;\n")
                .append("      color: var(--text-muted);\n")
                .append("      transition: transform 0.2s ease;\n")
                .append("    }\n")
                .append("    details[open] summary::after {\n")
                .append("      transform: rotate(90deg);\n")
                .append("    }\n")
                .append("    .test-summary-info {\n")
                .append("      display: flex;\n")
                .append("      align-items: center;\n")
                .append("      gap: 0.75rem;\n")
                .append("    }\n")
                .append("    .status-indicator {\n")
                .append("      width: 8px;\n")
                .append("      height: 8px;\n")
                .append("      border-radius: 50%;\n")
                .append("    }\n")
                .append("    .status-pass { background-color: var(--success); box-shadow: 0 0 8px var(--success); }\n")
                .append("    .status-fail { background-color: var(--failure); box-shadow: 0 0 8px var(--failure); }\n")
                .append("    .test-duration {\n")
                .append("      font-size: 0.85rem;\n")
                .append("      color: var(--text-muted);\n")
                .append("    }\n")
                .append("    .test-details {\n")
                .append("      padding: 1.25rem;\n")
                .append("      border-top: 1px solid var(--card-border);\n")
                .append("      background-color: rgba(0, 0, 0, 0.15);\n")
                .append("    }\n")
                .append("    .metrics-grid {\n")
                .append("      display: grid;\n")
                .append("      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));\n")
                .append("      gap: 1rem;\n")
                .append("      margin-top: 0.5rem;\n")
                .append("    }\n")
                .append("    .metric-card {\n")
                .append("      background-color: var(--metric-bg);\n")
                .append("      border: 1px solid var(--card-border);\n")
                .append("      border-radius: 0.5rem;\n")
                .append("      padding: 0.75rem 1rem;\n")
                .append("      display: flex;\n")
                .append("      flex-direction: column;\n")
                .append("      gap: 0.25rem;\n")
                .append("      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);\n")
                .append("    }\n")
                .append("    .metric-key {\n")
                .append("      font-size: 0.75rem;\n")
                .append("      font-weight: 600;\n")
                .append("      color: var(--text-muted);\n")
                .append("      text-transform: uppercase;\n")
                .append("      letter-spacing: 0.05em;\n")
                .append("    }\n")
                .append("    .metric-value {\n")
                .append("      font-size: 0.95rem;\n")
                .append("      font-weight: 500;\n")
                .append("      color: var(--text-main);\n")
                .append("      font-family: 'JetBrains Mono', monospace;\n")
                .append("      word-break: break-all;\n")
                .append("    }\n")
                .append("    .token-badge {\n")
                .append("      background-color: rgba(99, 102, 241, 0.15);\n" )
                .append("      border: 1px solid rgba(99, 102, 241, 0.3);\n")
                .append("      color: #818cf8;\n")
                .append("      font-size: 0.8rem;\n")
                .append("      padding: 0.15rem 0.5rem;\n")
                .append("      border-radius: 0.25rem;\n")
                .append("      margin-right: 0.35rem;\n")
                .append("      margin-bottom: 0.35rem;\n")
                .append("      display: inline-block;\n")
                .append("      font-weight: 500;\n")
                .append("    }\n")
                .append("    .error-header {\n")
                .append("      color: var(--failure);\n")
                .append("      font-weight: 600;\n")
                .append("      margin-bottom: 0.5rem;\n")
                .append("      font-size: 0.95rem;\n")
                .append("    }\n")
                .append("    .stacktrace {\n")
                .append("      background-color: #05070c;\n")
                .append("      border: 1px solid var(--card-border);\n")
                .append("      border-radius: 0.35rem;\n")
                .append("      padding: 1rem;\n")
                .append("      font-family: 'JetBrains Mono', monospace;\n")
                .append("      font-size: 0.85rem;\n")
                .append("      color: #cbd5e1;\n")
                .append("      overflow-x: auto;\n")
                .append("      white-space: pre;\n")
                .append("      max-height: 250px;\n")
                .append("    }\n")
                .append("    .log-entry {\n")
                .append("      font-size: 0.9rem;\n")
                .append("      color: var(--text-muted);\n")
                .append("      margin-bottom: 0.5rem;\n")
                .append("    }\n")
                .append("  </style>\n")
                .append("</head>\n")
                .append("<body>\n")
                .append("  <div class=\"container\">\n")
                .append("    <header>\n")
                .append("      <div>\n")
                .append("        <h1>TinyModelZ Test Report</h1>\n")
                .append("        <p style=\"font-size:0.9rem; color:var(--text-muted); margin-top:0.25rem;\">Automation Test Runner Execution Analytics</p>\n")
                .append("      </div>\n")
                .append("      <div>\n")
                .append("        <span class=\"badge ").append(failedTests > 0 ? "badge-failure" : "badge-success")
                .append("\">\n")
                .append("          ").append(failedTests > 0 ? "FAILED" : "PASSED").append("\n")
                .append("        </span>\n")
                .append("      </div>\n")
                .append("    </header>\n")
                .append("    \n")
                .append("    <div class=\"dashboard-grid\">\n")
                .append("      <div class=\"card stats-layout\">\n")
                .append("        <div class=\"stat-box\">\n")
                .append("          <div class=\"stat-label\">Suites Run</div>\n")
                .append("          <div class=\"stat-value\">").append(totalSuites).append("</div>\n")
                .append("        </div>\n")
                .append("        <div class=\"stat-box\">\n")
                .append("          <div class=\"stat-label\">Total Tests</div>\n")
                .append("          <div class=\"stat-value\">").append(totalTests).append("</div>\n")
                .append("        </div>\n")
                .append("        <div class=\"stat-box\">\n")
                .append("          <div class=\"stat-label\">Passed</div>\n")
                .append("          <div class=\"stat-value\" style=\"color: var(--success);\">").append(passedTests)
                .append("</div>\n")
                .append("        </div>\n")
                .append("        <div class=\"stat-box\">\n")
                .append("          <div class=\"stat-label\">Failed</div>\n")
                .append("          <div class=\"stat-value\" style=\"color: ")
                .append(failedTests > 0 ? "var(--failure)" : "var(--text-main)").append(";\">").append(failedTests)
                .append("</div>\n")
                .append("        </div>\n")
                .append("        <div class=\"stat-box\" style=\"grid-column: span 2;\">\n")
                .append("          <div class=\"stat-label\">Total Duration</div>\n")
                .append("          <div class=\"stat-value\" style=\"color: var(--accent);\">").append(totalDuration)
                .append(" ms</div>\n")
                .append("        </div>\n")
                .append("      </div>\n")
                .append("      \n")
                .append("      <div class=\"card chart-card\">\n")
                .append("        <div class=\"chart-container\">\n")
                .append("          <svg class=\"chart-svg\">\n")
                .append("            <circle class=\"chart-bg\" cx=\"80\" cy=\"80\" r=\"70\"></circle>\n")
                .append("            <circle class=\"chart-fill\" cx=\"80\" cy=\"80\" r=\"70\"></circle>\n")
                .append("          </svg>\n")
                .append("          <div class=\"chart-center\">").append(Math.round(passRate)).append("%</div>\n")
                .append("        </div>\n")
                .append("        <div class=\"chart-title\">Overall Pass Rate</div>\n")
                .append("      </div>\n")
                .append("    </div>\n")
                .append("    \n")
                .append("    <div class=\"controls-container\">\n")
                .append("      <div class=\"filter-tabs\">\n")
                .append("        <button class=\"filter-btn active\" onclick=\"filterTests('all')\">All Tests</button>\n")
                .append("        <button class=\"filter-btn\" onclick=\"filterTests('passed')\">Passed</button>\n")
                .append("        <button class=\"filter-btn\" onclick=\"filterTests('failed')\">Failed</button>\n")
                .append("      </div>\n")
                .append("      <input type=\"text\" class=\"search-input\" placeholder=\"Search tests...\" oninput=\"searchTests(this.value)\">\n")
                .append("    </div>\n");

        for (Suite s : suites) {
            html.append("    <div class=\"suite-section\">\n")
                    .append("      <div class=\"suite-header\">\n")
                    .append("        <div class=\"suite-title\">").append(s.name).append("</div>\n")
                    .append("        <div class=\"suite-meta\">Passed: ").append(s.getPassedCount()).append("/")
                    .append(s.tests.size())
                    .append(" (").append(s.durationMs).append("ms)</div>\n")
                    .append("      </div>\n")
                    .append("      <div class=\"test-list\">\n");

            for (TestCase tc : s.tests) {
                boolean isPass = "PASSED".equals(tc.status);
                html.append("        <details data-status=\"").append(tc.status).append("\">\n")
                        .append("          <summary>\n")
                        .append("            <div class=\"test-summary-info\">\n")
                        .append("              <div class=\"status-indicator ")
                        .append(isPass ? "status-pass" : "status-fail").append("\"></div>\n")
                        .append("              <span>").append(tc.name).append("</span>\n")
                        .append("            </div>\n")
                        .append("            <div class=\"test-duration\">").append(tc.durationMs).append(" ms</div>\n")
                        .append("          </summary>\n")
                        .append("          <div class=\"test-details\">\n");

                if (isPass) {
                    html.append("            <div class=\"log-entry\">Test execution completed successfully in ")
                            .append(tc.durationMs).append(" ms. All assertions passed.</div>\n");
                } else {
                    html.append("            <div class=\"error-header\">").append(tc.errorType).append("</div>\n")
                            .append("            <div class=\"log-entry\" style=\"margin-bottom:0.75rem;\"><b>Error:</b> ")
                            .append(tc.errorMessage).append("</div>\n")
                            .append("            <pre class=\"stacktrace\">").append(tc.stackTrace).append("</pre>\n");
                }

                if (!tc.metrics.isEmpty()) {
                    html.append("            <div class=\"metrics-grid\">\n");
                    for (Metric m : tc.metrics) {
                        html.append("              <div class=\"metric-card\">\n")
                                .append("                <div class=\"metric-key\">").append(m.key).append("</div>\n")
                                .append("                <div class=\"metric-value\">").append(m.value).append("</div>\n")
                                .append("              </div>\n");
                    }
                    html.append("            </div>\n");
                }

                html.append("          </div>\n")
                        .append("        </details>\n");
            }

            html.append("      </div>\n")
                    .append("    </div>\n");
        }

        html.append("  </div>\n")
                .append("  <script>\n")
                .append("    function filterTests(status) {\n")
                .append("      document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));\n")
                .append("      event.target.classList.add('active');\n")
                .append("      window.currentStatusFilter = status;\n")
                .append("      applyFilters();\n")
                .append("    }\n")
                .append("    \n")
                .append("    function searchTests(val) {\n")
                .append("      window.currentSearchQuery = val.toLowerCase();\n")
                .append("      applyFilters();\n")
                .append("    }\n")
                .append("    \n")
                .append("    function applyFilters() {\n")
                .append("      const query = window.currentSearchQuery || '';\n")
                .append("      const status = window.currentStatusFilter || 'all';\n")
                .append("      \n")
                .append("      document.querySelectorAll('.suite-section').forEach(suite => {\n")
                .append("        let visibleInSuite = 0;\n")
                .append("        const details = suite.querySelectorAll('details');\n")
                .append("        details.forEach(d => {\n")
                .append("          const name = d.querySelector('summary span').textContent.toLowerCase();\n")
                .append("          const isPass = d.getAttribute('data-status') === 'PASSED';\n")
                .append("          \n")
                .append("          const matchesQuery = name.includes(query);\n")
                .append("          const matchesStatus = (status === 'all') || \n")
                .append("                                (status === 'passed' && isPass) || \n")
                .append("                                (status === 'failed' && !isPass);\n")
                .append("                                \n")
                .append("          if (matchesQuery && matchesStatus) {\n")
                .append("            d.style.display = 'block';\n")
                .append("            visibleInSuite++;\n")
                .append("          } else {\n")
                .append("            d.style.display = 'none';\n")
                .append("          }\n")
                .append("        });\n")
                .append("        \n")
                .append("        if (visibleInSuite === 0 && details.length > 0) {\n")
                .append("          suite.style.display = 'none';\n")
                .append("        } else {\n")
                .append("          suite.style.display = 'block';\n")
                .append("        }\n")
                .append("      });\n")
                .append("    }\n")
                .append("    window.currentStatusFilter = 'all';\n")
                .append("    window.currentSearchQuery = '';\n")
                .append("    \n")
                .append("    document.addEventListener('DOMContentLoaded', () => {\n")
                .append("      document.querySelectorAll('.metric-value').forEach(el => {\n")
                .append("        const txt = el.textContent.trim();\n")
                .append("        if (txt.startsWith('[') && txt.endsWith(']')) {\n")
                .append("          try {\n")
                .append("            const arr = JSON.parse(txt);\n")
                .append("            if (Array.isArray(arr)) {\n")
                .append("              el.innerHTML = '';\n")
                .append("              arr.forEach(val => {\n")
                .append("                const span = document.createElement('span');\n")
                .append("                span.className = 'token-badge';\n")
                .append("                span.textContent = val;\n")
                .append("                el.appendChild(span);\n")
                .append("              });\n")
                .append("            }\n")
                .append("          } catch(e) {\n")
                .append("            const items = txt.slice(1, -1).split(',');\n")
                .append("            el.innerHTML = '';\n")
                .append("            items.forEach(val => {\n")
                .append("              const span = document.createElement('span');\n")
                .append("              span.className = 'token-badge';\n")
                .append("              span.textContent = val.trim();\n")
                .append("              el.appendChild(span);\n")
                .append("            });\n")
                .append("          }\n")
                .append("        }\n")
                .append("      });\n")
                .append("    });\n")
                .append("  </script>\n")
                .append("</body>\n")
                .append("</html>\n");

        try (FileWriter fw = new FileWriter(outputPath)) {
            fw.write(html.toString());
            logger.info("Test report HTML file written successfully to: {}", outputPath);
        } catch (IOException e) {
            logger.error("Failed to write HTML test report file", e);
        }
    }
}
