package com.tinymodelz;

/**
 * <h3>VersionTest</h3>
 *
 * <p>Validates Semantic Versioning constants and format string output.</p>
 */
public class VersionTest {

    public static void runTests() {
        TestReporter.runTest("Semantic versioning format validation", () -> {
            String ver = Version.getVersion();
            if (!ver.matches("^\\d+\\.\\d+\\.\\d+.*$")) {
                throw new AssertionError("Invalid semver version string: " + ver);
            }
            if (Version.MAJOR < 1) {
                throw new AssertionError("Major version should be at least 1");
            }
            TestReporter.logMetric("Semantic Version", Version.getFullVersionString());
        });
    }
}
