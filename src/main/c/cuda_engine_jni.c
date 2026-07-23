#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef unsigned long long CUdeviceptr;

static void* g_cudaHandle = NULL;
static CUcontext g_ctx = NULL;
static CUmodule g_module = NULL;
static CUstream g_defaultStream = NULL;
static char g_deviceName[256] = "Unknown CUDA Device";
static int g_initialized = 0;

// Kernel Functions
static CUfunction g_tiledMatmulFunc = NULL;
static CUfunction g_batchedTiledMatmulFunc = NULL;
static CUfunction g_embeddingFwdFunc = NULL;
static CUfunction g_embeddingBwdFunc = NULL;
static CUfunction g_geluFwdFunc = NULL;
static CUfunction g_geluBwdFunc = NULL;
static CUfunction g_layerNormFwdFunc = NULL;
static CUfunction g_elementwiseAddFunc = NULL;
static CUfunction g_vecAccFunc = NULL;
static CUfunction g_vecFillFunc = NULL;
static CUfunction g_adamwFunc = NULL;

// Driver API Function Pointers
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemAllocHost_t)(void**, size_t);
typedef CUresult (*cuMemFreeHost_t)(void*);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyHtoDAsync_t)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_t)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoDAsync_t)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, CUstream, void**, void**);
typedef CUresult (*cuStreamCreate_t)(CUstream*, unsigned int);
typedef CUresult (*cuStreamDestroy_t)(CUstream);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuGetErrorString_t)(CUresult, const char**);
typedef CUresult (*cuGetErrorName_t)(CUresult, const char**);
typedef CUresult (*cuStreamBeginCapture_t)(CUstream, int);
typedef CUresult (*cuStreamEndCapture_t)(CUstream, CUgraph*);
typedef CUresult (*cuGraphInstantiate_t)(CUgraphExec*, CUgraph, void*, char*, size_t);
typedef CUresult (*cuGraphLaunch_t)(CUgraphExec, CUstream);
typedef CUresult (*cuGraphDestroy_t)(CUgraph);
typedef CUresult (*cuGraphExecDestroy_t)(CUgraphExec);
typedef CUresult (*cuOccupancyMaxPotentialBlockSize_t)(int*, int*, CUfunction, void*, size_t, int);

static cuInit_t f_cuInit = NULL;
static cuDeviceGet_t f_cuDeviceGet = NULL;
static cuDeviceGetName_t f_cuDeviceGetName = NULL;
static cuCtxCreate_t f_cuCtxCreate = NULL;
static cuCtxDestroy_t f_cuCtxDestroy = NULL;
static cuModuleLoadData_t f_cuModuleLoadData = NULL;
static cuModuleGetFunction_t f_cuModuleGetFunction = NULL;
static cuMemAlloc_t f_cuMemAlloc = NULL;
static cuMemFree_t f_cuMemFree = NULL;
static cuMemAllocHost_t f_cuMemAllocHost = NULL;
static cuMemFreeHost_t f_cuMemFreeHost = NULL;
static cuMemcpyHtoD_t f_cuMemcpyHtoD = NULL;
static cuMemcpyDtoH_t f_cuMemcpyDtoH = NULL;
static cuMemcpyHtoDAsync_t f_cuMemcpyHtoDAsync = NULL;
static cuMemcpyDtoHAsync_t f_cuMemcpyDtoHAsync = NULL;
static cuMemcpyDtoDAsync_t f_cuMemcpyDtoDAsync = NULL;
static cuLaunchKernel_t f_cuLaunchKernel = NULL;
static cuStreamCreate_t f_cuStreamCreate = NULL;
static cuStreamDestroy_t f_cuStreamDestroy = NULL;
static cuStreamSynchronize_t f_cuStreamSynchronize = NULL;
static cuCtxSynchronize_t f_cuCtxSynchronize = NULL;
static cuGetErrorString_t f_cuGetErrorString = NULL;
static cuGetErrorName_t f_cuGetErrorName = NULL;
static cuStreamBeginCapture_t f_cuStreamBeginCapture = NULL;
static cuStreamEndCapture_t f_cuStreamEndCapture = NULL;
static cuGraphInstantiate_t f_cuGraphInstantiate = NULL;
static cuGraphLaunch_t f_cuGraphLaunch = NULL;
static cuGraphDestroy_t f_cuGraphDestroy = NULL;
static cuGraphExecDestroy_t f_cuGraphExecDestroy = NULL;
static cuOccupancyMaxPotentialBlockSize_t f_cuOccupancyMaxPotentialBlockSize = NULL;

static void logCudaError(CUresult res, const char* funcName, const char* file, int line) {
    if (res != 0) {
        const char* errName = "UNKNOWN_ERROR";
        const char* errStr = "No error description available";
        if (f_cuGetErrorName) f_cuGetErrorName(res, &errName);
        if (f_cuGetErrorString) f_cuGetErrorString(res, &errStr);
        fprintf(stderr, "[CUDA ERROR %d] %s at %s:%d: %s - %s\n", res, funcName, file, line, errName, errStr);
    }
}

#define CUDA_CHECK(call) do { \
    CUresult res = (call); \
    if (res != 0) { \
        logCudaError(res, #call, __FILE__, __LINE__); \
        return JNI_FALSE; \
    } \
} while(0)

#define CUDA_CHECK_NULL(call) do { \
    CUresult res = (call); \
    if (res != 0) { \
        logCudaError(res, #call, __FILE__, __LINE__); \
        return 0; \
    } \
} while(0)

static char* loadPtxFromFile(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, size, f);
    fclose(f);
    buf[size] = '\0';
    return buf;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nInit(JNIEnv *env, jclass cls) {
    if (g_initialized) return JNI_TRUE;

    g_cudaHandle = dlopen("libcuda.so", RTLD_LAZY);
    if (!g_cudaHandle) {
        g_cudaHandle = dlopen("/lib64/libcuda.so", RTLD_LAZY);
    }
    if (!g_cudaHandle) return JNI_FALSE;

    f_cuInit = (cuInit_t)dlsym(g_cudaHandle, "cuInit");
    f_cuDeviceGet = (cuDeviceGet_t)dlsym(g_cudaHandle, "cuDeviceGet");
    f_cuDeviceGetName = (cuDeviceGetName_t)dlsym(g_cudaHandle, "cuDeviceGetName");
    f_cuCtxCreate = (cuCtxCreate_t)dlsym(g_cudaHandle, "cuCtxCreate_v2");
    f_cuCtxDestroy = (cuCtxDestroy_t)dlsym(g_cudaHandle, "cuCtxDestroy_v2");
    f_cuModuleLoadData = (cuModuleLoadData_t)dlsym(g_cudaHandle, "cuModuleLoadData");
    f_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(g_cudaHandle, "cuModuleGetFunction");
    f_cuMemAlloc = (cuMemAlloc_t)dlsym(g_cudaHandle, "cuMemAlloc_v2");
    f_cuMemFree = (cuMemFree_t)dlsym(g_cudaHandle, "cuMemFree_v2");
    f_cuMemAllocHost = (cuMemAllocHost_t)dlsym(g_cudaHandle, "cuMemAllocHost_v2");
    f_cuMemFreeHost = (cuMemFreeHost_t)dlsym(g_cudaHandle, "cuMemFreeHost_v2");
    f_cuMemcpyHtoD = (cuMemcpyHtoD_t)dlsym(g_cudaHandle, "cuMemcpyHtoD_v2");
    f_cuMemcpyDtoH = (cuMemcpyDtoH_t)dlsym(g_cudaHandle, "cuMemcpyDtoH_v2");
    f_cuMemcpyHtoDAsync = (cuMemcpyHtoDAsync_t)dlsym(g_cudaHandle, "cuMemcpyHtoDAsync_v2");
    f_cuMemcpyDtoHAsync = (cuMemcpyDtoHAsync_t)dlsym(g_cudaHandle, "cuMemcpyDtoHAsync_v2");
    f_cuMemcpyDtoDAsync = (cuMemcpyDtoDAsync_t)dlsym(g_cudaHandle, "cuMemcpyDtoDAsync_v2");
    f_cuLaunchKernel = (cuLaunchKernel_t)dlsym(g_cudaHandle, "cuLaunchKernel");
    f_cuStreamCreate = (cuStreamCreate_t)dlsym(g_cudaHandle, "cuStreamCreate");
    f_cuStreamDestroy = (cuStreamDestroy_t)dlsym(g_cudaHandle, "cuStreamDestroy");
    f_cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(g_cudaHandle, "cuStreamSynchronize");
    f_cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(g_cudaHandle, "cuCtxSynchronize");
    f_cuGetErrorString = (cuGetErrorString_t)dlsym(g_cudaHandle, "cuGetErrorString");
    f_cuGetErrorName = (cuGetErrorName_t)dlsym(g_cudaHandle, "cuGetErrorName");
    f_cuStreamBeginCapture = (cuStreamBeginCapture_t)dlsym(g_cudaHandle, "cuStreamBeginCapture");
    f_cuStreamEndCapture = (cuStreamEndCapture_t)dlsym(g_cudaHandle, "cuStreamEndCapture");
    f_cuGraphInstantiate = (cuGraphInstantiate_t)dlsym(g_cudaHandle, "cuGraphInstantiate_v2");
    f_cuGraphLaunch = (cuGraphLaunch_t)dlsym(g_cudaHandle, "cuGraphLaunch");
    f_cuGraphDestroy = (cuGraphDestroy_t)dlsym(g_cudaHandle, "cuGraphDestroy");
    f_cuGraphExecDestroy = (cuGraphExecDestroy_t)dlsym(g_cudaHandle, "cuGraphExecDestroy");
    f_cuOccupancyMaxPotentialBlockSize = (cuOccupancyMaxPotentialBlockSize_t)dlsym(g_cudaHandle, "cuOccupancyMaxPotentialBlockSize");

    if (!f_cuInit || f_cuInit(0) != 0) return JNI_FALSE;

    CUdevice dev;
    if (f_cuDeviceGet(&dev, 0) != 0) return JNI_FALSE;

    f_cuDeviceGetName(g_deviceName, sizeof(g_deviceName), dev);
    strcat(g_deviceName, " (CUDA Async Driver)");

    if (f_cuCtxCreate(&g_ctx, 0, dev) != 0) return JNI_FALSE;

    if (f_cuStreamCreate) {
        f_cuStreamCreate(&g_defaultStream, 0);
    }

    char* ptxCode = loadPtxFromFile("src/main/resources/cuda/kernels.ptx");
    if (!ptxCode) ptxCode = loadPtxFromFile("target/classes/cuda/kernels.ptx");
    if (!ptxCode) ptxCode = loadPtxFromFile("../src/main/resources/cuda/kernels.ptx");
    if (!ptxCode) ptxCode = loadPtxFromFile("kernels.ptx");

    if (ptxCode) {
        CUresult res = f_cuModuleLoadData(&g_module, ptxCode);
        free(ptxCode);
        if (res != 0) {
            logCudaError(res, "cuModuleLoadData", __FILE__, __LINE__);
            return JNI_FALSE;
        }
    } else {
        fprintf(stderr, "[CUDA ERROR] Could not locate kernels.ptx file in resources or CWD.\n");
        return JNI_FALSE;
    }

    f_cuModuleGetFunction(&g_tiledMatmulFunc, g_module, "tiled_matmul_kernel");
    f_cuModuleGetFunction(&g_batchedTiledMatmulFunc, g_module, "batched_tiled_matmul_kernel");
    f_cuModuleGetFunction(&g_embeddingFwdFunc, g_module, "embedding_forward_kernel");
    f_cuModuleGetFunction(&g_embeddingBwdFunc, g_module, "embedding_backward_kernel");
    f_cuModuleGetFunction(&g_geluFwdFunc, g_module, "gelu_forward_kernel");
    f_cuModuleGetFunction(&g_geluBwdFunc, g_module, "gelu_backward_kernel");
    f_cuModuleGetFunction(&g_layerNormFwdFunc, g_module, "layernorm_forward_kernel");
    f_cuModuleGetFunction(&g_elementwiseAddFunc, g_module, "elementwise_add_kernel");
    f_cuModuleGetFunction(&g_vecAccFunc, g_module, "vec_accumulate_kernel");
    f_cuModuleGetFunction(&g_vecFillFunc, g_module, "vec_fill_kernel");
    f_cuModuleGetFunction(&g_adamwFunc, g_module, "adamw_step_kernel");

    g_initialized = 1;
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nIsAvailable(JNIEnv *env, jclass cls) {
    return g_initialized ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nGetDeviceName(JNIEnv *env, jclass cls) {
    return (*env)->NewStringUTF(env, g_deviceName);
}

// Memory Allocation & Destruction APIs
JNIEXPORT jlong JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nAllocBuffer(JNIEnv *env, jclass cls, jlong sizeBytes) {
    if (!g_initialized) return 0;
    CUdeviceptr d_ptr = 0;
    if (f_cuMemAlloc(&d_ptr, (size_t)sizeBytes) == 0) {
        return (jlong)d_ptr;
    }
    return 0;
}

JNIEXPORT void JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nFreeBuffer(JNIEnv *env, jclass cls, jlong handle) {
    if (g_initialized && handle != 0) {
        f_cuMemFree((CUdeviceptr)handle);
    }
}

JNIEXPORT jlong JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nAllocHostBuffer(JNIEnv *env, jclass cls, jlong sizeBytes) {
    if (!g_initialized || !f_cuMemAllocHost) return 0;
    void* h_ptr = NULL;
    if (f_cuMemAllocHost(&h_ptr, (size_t)sizeBytes) == 0) {
        return (jlong)(uintptr_t)h_ptr;
    }
    return 0;
}

JNIEXPORT void JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nFreeHostBuffer(JNIEnv *env, jclass cls, jlong handle) {
    if (g_initialized && handle != 0 && f_cuMemFreeHost) {
        f_cuMemFreeHost((void*)(uintptr_t)handle);
    }
}

// Memory Transfers (Sync and Async Streams)
JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nCopyToGPU(JNIEnv *env, jclass cls, jlong handle, jfloatArray hData, jlong sizeBytes) {
    if (!g_initialized || handle == 0 || hData == NULL) return JNI_FALSE;
    jfloat *ptr = (*env)->GetPrimitiveArrayCritical(env, hData, NULL);
    if (!ptr) return JNI_FALSE;
    CUresult res = f_cuMemcpyHtoD((CUdeviceptr)handle, ptr, (size_t)sizeBytes);
    (*env)->ReleasePrimitiveArrayCritical(env, hData, ptr, JNI_ABORT);
    return res == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nCopyFromGPU(JNIEnv *env, jclass cls, jfloatArray hData, jlong handle, jlong sizeBytes) {
    if (!g_initialized || handle == 0 || hData == NULL) return JNI_FALSE;
    jfloat *ptr = (*env)->GetPrimitiveArrayCritical(env, hData, NULL);
    if (!ptr) return JNI_FALSE;
    CUresult res = f_cuMemcpyDtoH(ptr, (CUdeviceptr)handle, (size_t)sizeBytes);
    (*env)->ReleasePrimitiveArrayCritical(env, hData, ptr, 0);
    return res == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nCopyGPUToGPU(JNIEnv *env, jclass cls, jlong destHandle, jlong srcHandle, jlong sizeBytes) {
    if (!g_initialized || destHandle == 0 || srcHandle == 0) return JNI_FALSE;
    if (f_cuMemcpyDtoDAsync) {
        CUresult res = f_cuMemcpyDtoDAsync((CUdeviceptr)destHandle, (CUdeviceptr)srcHandle, (size_t)sizeBytes, g_defaultStream);
        return res == 0 ? JNI_TRUE : JNI_FALSE;
    }
    return JNI_FALSE;
}

// GPU-Resident Tiled MatMul and Batched MatMul
JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nMatMulResident(
        JNIEnv *env, jclass cls,
        jlong aHandle, jlong bHandle, jlong cHandle,
        jint m, jint n, jint k) {

    if (!g_initialized || !g_tiledMatmulFunc || aHandle == 0 || bHandle == 0 || cHandle == 0) {
        return JNI_FALSE;
    }

    int blockDimX = 16;
    int blockDimY = 16;
    int gridDimX = (n + 15) / 16;
    int gridDimY = (m + 15) / 16;

    void* args[] = { &aHandle, &bHandle, &cHandle, &m, &n, &k };

    CUDA_CHECK(f_cuLaunchKernel(g_tiledMatmulFunc, gridDimX, gridDimY, 1, blockDimX, blockDimY, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nBatchedMatMulResident(
        JNIEnv *env, jclass cls,
        jlong aHandle, jlong bHandle, jlong cHandle,
        jint numBatches, jint m, jint n, jint k) {

    if (!g_initialized || !g_batchedTiledMatmulFunc || aHandle == 0 || bHandle == 0 || cHandle == 0) {
        return JNI_FALSE;
    }

    int blockDimX = 16;
    int blockDimY = 16;
    int gridDimX = (n + 15) / 16;
    int gridDimY = (m + 15) / 16;
    int gridDimZ = numBatches;

    void* args[] = { &aHandle, &bHandle, &cHandle, &m, &n, &k };

    CUDA_CHECK(f_cuLaunchKernel(g_batchedTiledMatmulFunc, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

// Fallback arrays-based MatMul for backwards compatibility
JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nMatMul(
        JNIEnv *env, jclass cls,
        jfloatArray a, jfloatArray b, jfloatArray c,
        jint m, jint n, jint k) {

    if (!g_initialized) return JNI_FALSE;
    size_t sizeA = (size_t)m * k * sizeof(float);
    size_t sizeB = (size_t)k * n * sizeof(float);
    size_t sizeC = (size_t)m * n * sizeof(float);

    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    if (f_cuMemAlloc(&d_a, sizeA) != 0 || f_cuMemAlloc(&d_b, sizeB) != 0 || f_cuMemAlloc(&d_c, sizeC) != 0) {
        if (d_a) f_cuMemFree(d_a);
        if (d_b) f_cuMemFree(d_b);
        if (d_c) f_cuMemFree(d_c);
        return JNI_FALSE;
    }

    jfloat *h_a = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *h_b = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
    jfloat *h_c = (*env)->GetPrimitiveArrayCritical(env, c, NULL);

    f_cuMemcpyHtoD(d_a, h_a, sizeA);
    f_cuMemcpyHtoD(d_b, h_b, sizeB);

    jboolean ok = Java_com_tinymodelz_gpu_CUDAMathEngine_nMatMulResident(env, cls, (jlong)d_a, (jlong)d_b, (jlong)d_c, m, n, k);

    if (ok) {
        if (f_cuStreamSynchronize) f_cuStreamSynchronize(g_defaultStream);
        f_cuMemcpyDtoH(h_c, d_c, sizeC);
    }

    (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);

    f_cuMemFree(d_a);
    f_cuMemFree(d_b);
    f_cuMemFree(d_c);

    return ok;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nBatchedMatMul(
        JNIEnv *env, jclass cls,
        jfloatArray a, jfloatArray b, jfloatArray c,
        jint numBatches, jint m, jint n, jint k) {

    if (!g_initialized) return JNI_FALSE;
    size_t sizeA = (size_t)numBatches * m * k * sizeof(float);
    size_t sizeB = (size_t)numBatches * k * n * sizeof(float);
    size_t sizeC = (size_t)numBatches * m * n * sizeof(float);

    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    if (f_cuMemAlloc(&d_a, sizeA) != 0 || f_cuMemAlloc(&d_b, sizeB) != 0 || f_cuMemAlloc(&d_c, sizeC) != 0) {
        if (d_a) f_cuMemFree(d_a);
        if (d_b) f_cuMemFree(d_b);
        if (d_c) f_cuMemFree(d_c);
        return JNI_FALSE;
    }

    jfloat *h_a = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *h_b = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
    jfloat *h_c = (*env)->GetPrimitiveArrayCritical(env, c, NULL);

    f_cuMemcpyHtoD(d_a, h_a, sizeA);
    f_cuMemcpyHtoD(d_b, h_b, sizeB);

    jboolean ok = Java_com_tinymodelz_gpu_CUDAMathEngine_nBatchedMatMulResident(env, cls, (jlong)d_a, (jlong)d_b, (jlong)d_c, numBatches, m, n, k);

    if (ok) {
        if (f_cuStreamSynchronize) f_cuStreamSynchronize(g_defaultStream);
        f_cuMemcpyDtoH(h_c, d_c, sizeC);
    }

    (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);

    f_cuMemFree(d_a);
    f_cuMemFree(d_b);
    f_cuMemFree(d_c);

    return ok;
}

// GPU Resident Kernel Launches (Embedding, GELU, LayerNorm, Add, AdamW)
JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nEmbeddingForward(
        JNIEnv *env, jclass cls,
        jlong wHandle, jlong tokensHandle, jlong outHandle,
        jint batchSeq, jint embedDim) {

    if (!g_initialized || !g_embeddingFwdFunc) return JNI_FALSE;
    int size = batchSeq * embedDim;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &wHandle, &tokensHandle, &outHandle, &batchSeq, &embedDim };
    CUDA_CHECK(f_cuLaunchKernel(g_embeddingFwdFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nGeluForward(
        JNIEnv *env, jclass cls,
        jlong inHandle, jlong outHandle, jint size) {

    if (!g_initialized || !g_geluFwdFunc) return JNI_FALSE;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &inHandle, &outHandle, &size };
    CUDA_CHECK(f_cuLaunchKernel(g_geluFwdFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nGeluBackward(
        JNIEnv *env, jclass cls,
        jlong gradInHandle, jlong gradOutHandle, jlong inHandle, jint size) {

    if (!g_initialized || !g_geluBwdFunc) return JNI_FALSE;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &gradInHandle, &gradOutHandle, &inHandle, &size };
    CUDA_CHECK(f_cuLaunchKernel(g_geluBwdFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nLayerNormForward(
        JNIEnv *env, jclass cls,
        jlong inHandle, jlong gammaHandle, jlong betaHandle, jlong outHandle,
        jlong meanHandle, jlong rstdHandle, jint numRows, jint normDim, jfloat eps) {

    if (!g_initialized || !g_layerNormFwdFunc) return JNI_FALSE;
    int blocksPerGrid = numRows;
    int threadsPerBlock = 1;
    void* args[] = { &inHandle, &gammaHandle, &betaHandle, &outHandle, &meanHandle, &rstdHandle, &numRows, &normDim, &eps };
    CUDA_CHECK(f_cuLaunchKernel(g_layerNormFwdFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nElementwiseAdd(
        JNIEnv *env, jclass cls,
        jlong aHandle, jlong bHandle, jlong cHandle, jint size) {

    if (!g_initialized || !g_elementwiseAddFunc) return JNI_FALSE;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &aHandle, &bHandle, &cHandle, &size };
    CUDA_CHECK(f_cuLaunchKernel(g_elementwiseAddFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nVecAccumulate(
        JNIEnv *env, jclass cls,
        jlong destHandle, jlong srcHandle, jint size) {

    if (!g_initialized || !g_vecAccFunc) return JNI_FALSE;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &destHandle, &srcHandle, &size };
    CUDA_CHECK(f_cuLaunchKernel(g_vecAccFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nVecFill(
        JNIEnv *env, jclass cls,
        jlong destHandle, jfloat value, jint size) {

    if (!g_initialized || !g_vecFillFunc) return JNI_FALSE;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &destHandle, &value, &size };
    CUDA_CHECK(f_cuLaunchKernel(g_vecFillFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nAdamWStep(
        JNIEnv *env, jclass cls,
        jlong pHandle, jlong gHandle, jlong mHandle, jlong vHandle,
        jint size, jfloat lr, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay,
        jfloat bc1, jfloat bc2) {

    if (!g_initialized || !g_adamwFunc || pHandle == 0 || gHandle == 0 || mHandle == 0 || vHandle == 0) {
        return JNI_FALSE;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &pHandle, &gHandle, &mHandle, &vHandle, &size, &lr, &beta1, &beta2, &eps, &weightDecay, &bc1, &bc2 };

    CUDA_CHECK(f_cuLaunchKernel(g_adamwFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, g_defaultStream, args, NULL));
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nSynchronizeStream(JNIEnv *env, jclass cls) {
    if (g_initialized && f_cuStreamSynchronize) {
        f_cuStreamSynchronize(g_defaultStream);
    }
}

JNIEXPORT void JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nSynchronizeContext(JNIEnv *env, jclass cls) {
    if (g_initialized && f_cuCtxSynchronize) {
        f_cuCtxSynchronize();
    }
}

JNIEXPORT void JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nShutdown(JNIEnv *env, jclass cls) {
    if (g_initialized) {
        if (g_defaultStream && f_cuStreamDestroy) {
            f_cuStreamDestroy(g_defaultStream);
            g_defaultStream = NULL;
        }
        if (g_ctx && f_cuCtxDestroy) {
            f_cuCtxDestroy(g_ctx);
            g_ctx = NULL;
        }
        g_initialized = 0;
    }
}
