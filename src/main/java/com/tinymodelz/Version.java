package com.tinymodelz;

/**
 * <h3>Version</h3>
 *
 * <p>Semantic Versioning metadata (SemVer 2.0.0) for TinyModelZ.</p>
 */
public final class Version {

    public static final int MAJOR = 1;
    public static final int MINOR = 0;
    public static final int PATCH = 0;

    public static final String VERSION = "1.0.0";
    public static final String NAME = "TinyModelZ Engine";

    private Version() {
        // Utility class
    }

    /**
     * Gets the full semantic version string.
     *
     * @return semantic version string (e.g. "1.0.0")
     */
    public static String getVersion() {
        return VERSION;
    }

    /**
     * Prints version summary details.
     */
    public static String getFullVersionString() {
        return NAME + " v" + VERSION + " (Java " + System.getProperty("java.version") + ")";
    }
}
