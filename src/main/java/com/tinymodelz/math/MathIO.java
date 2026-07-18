package com.tinymodelz.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * <h3>MathIO</h3>
 * 
 * <p>Handles binary serialization and deserialization of {@link Matrix} and {@link Vector} instances
 * using a custom, portable, high-performance binary format (TMA1).</p>
 * 
 * <h4>TMA1 Binary Format Specification:</h4>
 * <table>
 *   <tr>
 *     <th>Field</th>
 *     <th>Size (Bytes)</th>
 *     <th>Type</th>
 *     <th>Description</th>
 *   </tr>
 *   <tr>
 *     <td>Magic Bytes</td>
 *     <td>4</td>
 *     <td>char[]</td>
 *     <td>Identifies format: {'T', 'M', 'A', '1'}</td>
 *   </tr>
 *   <tr>
 *     <td>Type Flag</td>
 *     <td>1</td>
 *     <td>byte</td>
 *     <td>0x01 = Vector, 0x02 = Matrix</td>
 *   </tr>
 *   <tr>
 *     <td>Dimensions</td>
 *     <td>4 or 8</td>
 *     <td>int / (int, int)</td>
 *     <td>Vector: size. Matrix: (rows, cols)</td>
 *   </tr>
 *   <tr>
 *     <td>Payload</td>
 *     <td>4 * N</td>
 *     <td>float[]</td>
 *     <td>Flat sequence of floating-point values</td>
 *   </tr>
 * </table>
 * 
 * <h4>Complexity Analysis:</h4>
 * <ul>
 *   <li><b>Time Complexity:</b> $O(N)$ for writing/reading $N$ float values.</li>
 *   <li><b>Space Complexity:</b> $O(1)$ auxiliary space during streaming.</li>
 * </ul>
 */
public class MathIO {

    private static final Logger logger = LoggerFactory.getLogger(MathIO.class);

    private static final byte[] MAGIC_BYTES = {'T', 'M', 'A', '1'};
    private static final byte TYPE_VECTOR = 0x01;
    private static final byte TYPE_MATRIX = 0x02;

    /**
     * Saves a {@link Vector} to a file.
     * 
     * @param vector the vector to save
     * @param file the destination file
     * @throws IOException if an I/O error occurs
     */
    public static void saveVector(Vector vector, File file) throws IOException {
        if (vector == null || file == null) {
            throw new IllegalArgumentException("Vector and file arguments cannot be null");
        }
        logger.debug("Saving Vector of size {} to file: {}", vector.size(), file.getAbsolutePath());

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            dos.write(MAGIC_BYTES);
            dos.writeByte(TYPE_VECTOR);
            dos.writeInt(vector.size());
            float[] data = vector.getDataCopy();
            for (float val : data) {
                dos.writeFloat(val);
            }
        }
    }

    /**
     * Loads a {@link Vector} from a file.
     * 
     * @param file the source file
     * @return the loaded Vector
     * @throws IOException if an I/O error occurs or format validation fails
     */
    public static Vector loadVector(File file) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("File argument cannot be null");
        }
        logger.debug("Loading Vector from file: {}", file.getAbsolutePath());

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            validateMagicBytes(dis);
            byte type = dis.readByte();
            if (type != TYPE_VECTOR) {
                throw new IOException("Expected Vector type flag (0x01), but found: " + String.format("0x%02X", type));
            }
            int size = dis.readInt();
            if (size <= 0) {
                throw new IOException("Invalid vector size: " + size);
            }
            float[] data = new float[size];
            for (int i = 0; i < size; i++) {
                data[i] = dis.readFloat();
            }
            return new Vector(data);
        }
    }

    /**
     * Saves a {@link Matrix} to a file.
     * 
     * @param matrix the matrix to save
     * @param file the destination file
     * @throws IOException if an I/O error occurs
     */
    public static void saveMatrix(Matrix matrix, File file) throws IOException {
        if (matrix == null || file == null) {
            throw new IllegalArgumentException("Matrix and file arguments cannot be null");
        }
        logger.debug("Saving Matrix of shape {}x{} to file: {}", matrix.rows(), matrix.cols(), file.getAbsolutePath());

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            dos.write(MAGIC_BYTES);
            dos.writeByte(TYPE_MATRIX);
            dos.writeInt(matrix.rows());
            dos.writeInt(matrix.cols());
            float[][] data = matrix.getDataCopy();
            for (int i = 0; i < matrix.rows(); i++) {
                for (int j = 0; j < matrix.cols(); j++) {
                    dos.writeFloat(data[i][j]);
                }
            }
        }
    }

    /**
     * Loads a {@link Matrix} from a file.
     * 
     * @param file the source file
     * @return the loaded Matrix
     * @throws IOException if an I/O error occurs or format validation fails
     */
    public static Matrix loadMatrix(File file) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("File argument cannot be null");
        }
        logger.debug("Loading Matrix from file: {}", file.getAbsolutePath());

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            validateMagicBytes(dis);
            byte type = dis.readByte();
            if (type != TYPE_MATRIX) {
                throw new IOException("Expected Matrix type flag (0x02), but found: " + String.format("0x%02X", type));
            }
            int rows = dis.readInt();
            int cols = dis.readInt();
            if (rows <= 0 || cols <= 0) {
                throw new IOException("Invalid matrix shape: " + rows + "x" + cols);
            }
            float[][] data = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    data[i][j] = dis.readFloat();
                }
            }
            return new Matrix(data);
        }
    }

    private static void validateMagicBytes(DataInputStream dis) throws IOException {
        byte[] magic = new byte[4];
        dis.readFully(magic);
        for (int i = 0; i < 4; i++) {
            if (magic[i] != MAGIC_BYTES[i]) {
                throw new IOException("Invalid Magic Bytes. Not a TMA1 file.");
            }
        }
    }
}
