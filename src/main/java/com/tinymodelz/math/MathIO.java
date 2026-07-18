package com.tinymodelz.math;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

/**
 * <h3>MathIO</h3>
 * 
 * <p>Handles high-performance binary serialization and deserialization of {@link Matrix}, {@link Vector}, and {@link Tensor} instances
 * using Java NIO {@link FileChannel} and {@link ByteBuffer} chunking.</p>
 */
public class MathIO {

    private static final Logger logger = LoggerFactory.getLogger(MathIO.class);

    private static final byte[] MAGIC_BYTES = {'T', 'M', 'A', '1'};
    private static final byte[] MAGIC_TENSOR = {'T', 'M', 'A', 'T'};
    private static final byte TYPE_VECTOR = 0x01;
    private static final byte TYPE_MATRIX = 0x02;

    // Buffer size optimized for cache and memory limits (256 KB)
    private static final int BUFFER_SIZE = 256 * 1024;

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

        try (FileOutputStream fos = new FileOutputStream(file);
             FileChannel channel = fos.getChannel()) {
            
            // Header: Magic (4) + Type (1) + Size (4) = 9 bytes
            ByteBuffer headerBuf = ByteBuffer.allocate(9).order(ByteOrder.BIG_ENDIAN);
            headerBuf.put(MAGIC_BYTES);
            headerBuf.put(TYPE_VECTOR);
            headerBuf.putInt(vector.size());
            headerBuf.flip();
            channel.write(headerBuf);

            float[] data = vector.getData();
            int totalFloats = data.length;
            ByteBuffer dataBuf = ByteBuffer.allocate(Math.min(totalFloats * 4, BUFFER_SIZE)).order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuf = dataBuf.asFloatBuffer();

            int offset = 0;
            while (offset < totalFloats) {
                int chunk = Math.min(totalFloats - offset, floatBuf.remaining());
                floatBuf.put(data, offset, chunk);
                
                dataBuf.position(0);
                dataBuf.limit(chunk * 4);
                channel.write(dataBuf);
                
                dataBuf.clear();
                floatBuf.clear();
                offset += chunk;
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

        try (FileInputStream fis = new FileInputStream(file);
             FileChannel channel = fis.getChannel()) {
            
            ByteBuffer headerBuf = ByteBuffer.allocate(9).order(ByteOrder.BIG_ENDIAN);
            while (headerBuf.hasRemaining()) {
                if (channel.read(headerBuf) == -1) {
                    throw new IOException("Unexpected end of file while reading header");
                }
            }
            headerBuf.flip();
            
            byte[] magic = new byte[4];
            headerBuf.get(magic);
            for (int i = 0; i < 4; i++) {
                if (magic[i] != MAGIC_BYTES[i]) {
                    throw new IOException("Invalid Magic Bytes. Not a TMA1 file.");
                }
            }
            
            byte type = headerBuf.get();
            if (type != TYPE_VECTOR) {
                throw new IOException("Expected Vector type flag (0x01), but found: " + String.format("0x%02X", type));
            }
            
            int size = headerBuf.getInt();
            if (size <= 0) {
                throw new IOException("Invalid vector size: " + size);
            }
            
            float[] data = new float[size];
            ByteBuffer dataBuf = ByteBuffer.allocate(Math.min(size * 4, BUFFER_SIZE)).order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuf = dataBuf.asFloatBuffer();
            
            int offset = 0;
            while (offset < size) {
                int chunkInBytes = Math.min((size - offset) * 4, dataBuf.capacity());
                dataBuf.limit(chunkInBytes);
                dataBuf.position(0);
                
                while (dataBuf.hasRemaining()) {
                    if (channel.read(dataBuf) == -1) {
                        throw new IOException("Unexpected end of file while reading data");
                    }
                }
                
                dataBuf.flip();
                floatBuf.clear();
                int chunk = chunkInBytes / 4;
                floatBuf.get(data, offset, chunk);
                offset += chunk;
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

        try (FileOutputStream fos = new FileOutputStream(file);
             FileChannel channel = fos.getChannel()) {
            
            // Header: Magic (4) + Type (1) + Rows (4) + Cols (4) = 13 bytes
            ByteBuffer headerBuf = ByteBuffer.allocate(13).order(ByteOrder.BIG_ENDIAN);
            headerBuf.put(MAGIC_BYTES);
            headerBuf.put(TYPE_MATRIX);
            headerBuf.putInt(matrix.rows());
            headerBuf.putInt(matrix.cols());
            headerBuf.flip();
            channel.write(headerBuf);

            float[][] data = matrix.getData();
            int rows = matrix.rows();
            int cols = matrix.cols();

            ByteBuffer dataBuf = ByteBuffer.allocate(Math.min(cols * 4, BUFFER_SIZE)).order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuf = dataBuf.asFloatBuffer();

            for (int i = 0; i < rows; i++) {
                float[] row = data[i];
                int offset = 0;
                while (offset < cols) {
                    int chunk = Math.min(cols - offset, floatBuf.remaining());
                    floatBuf.put(row, offset, chunk);
                    
                    dataBuf.position(0);
                    dataBuf.limit(chunk * 4);
                    channel.write(dataBuf);
                    
                    dataBuf.clear();
                    floatBuf.clear();
                    offset += chunk;
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

        try (FileInputStream fis = new FileInputStream(file);
             FileChannel channel = fis.getChannel()) {
            
            ByteBuffer headerBuf = ByteBuffer.allocate(13).order(ByteOrder.BIG_ENDIAN);
            while (headerBuf.hasRemaining()) {
                if (channel.read(headerBuf) == -1) {
                    throw new IOException("Unexpected end of file while reading header");
                }
            }
            headerBuf.flip();
            
            byte[] magic = new byte[4];
            headerBuf.get(magic);
            for (int i = 0; i < 4; i++) {
                if (magic[i] != MAGIC_BYTES[i]) {
                    throw new IOException("Invalid Magic Bytes. Not a TMA1 file.");
                }
            }
            
            byte type = headerBuf.get();
            if (type != TYPE_MATRIX) {
                throw new IOException("Expected Matrix type flag (0x02), but found: " + String.format("0x%02X", type));
            }
            
            int rows = headerBuf.getInt();
            int cols = headerBuf.getInt();
            if (rows <= 0 || cols <= 0) {
                throw new IOException("Invalid matrix shape: " + rows + "x" + cols);
            }
            
            float[][] data = new float[rows][cols];
            ByteBuffer dataBuf = ByteBuffer.allocate(Math.min(cols * 4, BUFFER_SIZE)).order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuf = dataBuf.asFloatBuffer();
            
            for (int i = 0; i < rows; i++) {
                float[] row = data[i];
                int offset = 0;
                while (offset < cols) {
                    int chunkInBytes = Math.min((cols - offset) * 4, dataBuf.capacity());
                    dataBuf.limit(chunkInBytes);
                    dataBuf.position(0);
                    
                    while (dataBuf.hasRemaining()) {
                        if (channel.read(dataBuf) == -1) {
                            throw new IOException("Unexpected end of file while reading data");
                        }
                    }
                    
                    dataBuf.flip();
                    floatBuf.clear();
                    int chunk = chunkInBytes / 4;
                    floatBuf.get(row, offset, chunk);
                    offset += chunk;
                }
            }
            return new Matrix(data);
        }
    }

    /**
     * Saves a {@link Tensor} to a file using the TMAT format.
     * 
     * @param tensor the tensor to save
     * @param file the destination file
     * @throws IOException if an I/O error occurs
     */
    public static void saveTensor(Tensor tensor, File file) throws IOException {
        if (tensor == null || file == null) {
            throw new IllegalArgumentException("Tensor and file arguments cannot be null");
        }
        logger.debug("Saving Tensor of shape {} to file: {}", java.util.Arrays.toString(tensor.getShape()), file.getAbsolutePath());

        try (FileOutputStream fos = new FileOutputStream(file);
             FileChannel channel = fos.getChannel()) {
            
            int[] shape = tensor.getShape();
            int rank = shape.length;
            
            // Header: Magic (4) + Rank (4) + Shape (4 * rank)
            ByteBuffer headerBuf = ByteBuffer.allocate(8 + 4 * rank).order(ByteOrder.BIG_ENDIAN);
            headerBuf.put(MAGIC_TENSOR);
            headerBuf.putInt(rank);
            for (int dim : shape) {
                headerBuf.putInt(dim);
            }
            headerBuf.flip();
            channel.write(headerBuf);

            // Get contiguous view of data
            Tensor contiguous = tensor.toContiguous();
            float[] data = contiguous.getData();
            int totalFloats = contiguous.size();
            
            ByteBuffer dataBuf = ByteBuffer.allocate(Math.min(totalFloats * 4, BUFFER_SIZE)).order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuf = dataBuf.asFloatBuffer();

            int offset = 0;
            while (offset < totalFloats) {
                int chunk = Math.min(totalFloats - offset, floatBuf.remaining());
                floatBuf.put(data, offset, chunk);
                
                dataBuf.position(0);
                dataBuf.limit(chunk * 4);
                channel.write(dataBuf);
                
                dataBuf.clear();
                floatBuf.clear();
                offset += chunk;
            }
        }
    }

    /**
     * Loads a {@link Tensor} from a TMAT format file.
     * 
     * @param file the source file
     * @return the loaded Tensor
     * @throws IOException if an I/O error occurs or format validation fails
     */
    public static Tensor loadTensor(File file) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("File argument cannot be null");
        }
        logger.debug("Loading Tensor from file: {}", file.getAbsolutePath());

        try (FileInputStream fis = new FileInputStream(file);
             FileChannel channel = fis.getChannel()) {
            
            ByteBuffer magicAndRankBuf = ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN);
            while (magicAndRankBuf.hasRemaining()) {
                if (channel.read(magicAndRankBuf) == -1) {
                    throw new IOException("Unexpected end of file while reading magic/rank");
                }
            }
            magicAndRankBuf.flip();
            
            byte[] magic = new byte[4];
            magicAndRankBuf.get(magic);
            for (int i = 0; i < 4; i++) {
                if (magic[i] != MAGIC_TENSOR[i]) {
                    throw new IOException("Invalid Magic Bytes. Not a TMAT tensor file.");
                }
            }
            
            int rank = magicAndRankBuf.getInt();
            if (rank <= 0) {
                throw new IOException("Invalid Tensor rank: " + rank);
            }
            
            ByteBuffer shapeBuf = ByteBuffer.allocate(4 * rank).order(ByteOrder.BIG_ENDIAN);
            while (shapeBuf.hasRemaining()) {
                if (channel.read(shapeBuf) == -1) {
                    throw new IOException("Unexpected end of file while reading shape");
                }
            }
            shapeBuf.flip();
            
            int[] shape = new int[rank];
            int size = 1;
            for (int i = 0; i < rank; i++) {
                shape[i] = shapeBuf.getInt();
                if (shape[i] <= 0) {
                    throw new IOException("Invalid dimension size: " + shape[i]);
                }
                size *= shape[i];
            }
            
            float[] data = new float[size];
            ByteBuffer dataBuf = ByteBuffer.allocate(Math.min(size * 4, BUFFER_SIZE)).order(ByteOrder.BIG_ENDIAN);
            FloatBuffer floatBuf = dataBuf.asFloatBuffer();
            
            int offset = 0;
            while (offset < size) {
                int chunkInBytes = Math.min((size - offset) * 4, dataBuf.capacity());
                dataBuf.limit(chunkInBytes);
                dataBuf.position(0);
                
                while (dataBuf.hasRemaining()) {
                    if (channel.read(dataBuf) == -1) {
                        throw new IOException("Unexpected end of file while reading data");
                    }
                }
                
                dataBuf.flip();
                floatBuf.clear();
                int chunk = chunkInBytes / 4;
                floatBuf.get(data, offset, chunk);
                offset += chunk;
            }
            return new Tensor(data, shape);
        }
    }
}
