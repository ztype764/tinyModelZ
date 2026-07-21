#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;

typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*fn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (*fn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_program (*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel (*fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (*fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_int, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (*fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_int, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (*fn_clFinish)(cl_command_queue);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem);

static void* lib_handle = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel_matmul = NULL;
static cl_kernel kernel_add = NULL;
static char device_name[256] = "Unknown GPU";
static int gpu_initialized = 0;

// Persistent GPU buffer cache to eliminate allocation overhead (Phase 2 & 3)
static cl_mem bufA = NULL;
static cl_mem bufB = NULL;
static cl_mem bufC = NULL;
static size_t capA = 0;
static size_t capB = 0;
static size_t capC = 0;

static fn_clCreateBuffer p_clCreateBuffer;
static fn_clEnqueueWriteBuffer p_clEnqueueWriteBuffer;
static fn_clEnqueueReadBuffer p_clEnqueueReadBuffer;
static fn_clSetKernelArg p_clSetKernelArg;
static fn_clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel;
static fn_clFinish p_clFinish;
static fn_clReleaseMemObject p_clReleaseMemObject;

static cl_kernel kernel_batched_matmul = NULL;

// Tiled Local Memory OpenCL Matrix Multiplication Kernel (Phase 2 Performance Upgrade)
const char* gpu_source = 
"#define TILE_SIZE 16\n"
"__kernel void matmul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    float sum = 0.0f;\n"
"    if (row < M && col < N) {\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n"
"__kernel void batched_matmul(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    int batch = get_global_id(2);\n"
"    int offA = batch * M * K;\n"
"    int offB = batch * K * N;\n"
"    int offC = batch * M * N;\n"
"    float sum = 0.0f;\n"
"    if (row < M && col < N) {\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[offA + row * K + k] * B[offB + k * N + col];\n"
"        }\n"
"        C[offC + row * N + col] = sum;\n"
"    }\n"
"}\n"
"__kernel void vec_add(__global const float* A, __global const float* B, __global float* C, int size) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < size) C[idx] = A[idx] + B[idx];\n"
"}\n"
"__kernel void adamw_step(__global float* p, __global const float* g, __global float* m, __global float* v, int size, float lr, float beta1, float beta2, float eps, float weightDecay, float bc1, float bc2) {\n"
"    int i = get_global_id(0);\n"
"    if (i < size) {\n"
"        float w = p[i];\n"
"        float grad = g[i];\n"
"        float mVal = beta1 * m[i] + (1.0f - beta1) * grad;\n"
"        m[i] = mVal;\n"
"        float vVal = beta2 * v[i] + (1.0f - beta2) * grad * grad;\n"
"        v[i] = vVal;\n"
"        float mHat = mVal / bc1;\n"
"        float vHat = vVal / bc2;\n"
"        float adamStep = (lr / (sqrt(vHat) + eps)) * mHat;\n"
"        float decayStep = (weightDecay != 0.0f) ? lr * weightDecay * w : 0.0f;\n"
"        p[i] = w - adamStep - decayStep;\n"
"    }\n"
"}\n";

static cl_kernel kernel_adamw = NULL;

static int internal_gpu_init() {
    if (gpu_initialized) return 1;

    lib_handle = dlopen("/usr/lib64/libOpenCL.so.1", RTLD_LAZY);
    if (!lib_handle) lib_handle = dlopen("/usr/lib/libOpenCL.so.1", RTLD_LAZY);
    if (!lib_handle) lib_handle = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!lib_handle) return 0;

    fn_clGetPlatformIDs p_clGetPlatformIDs = (fn_clGetPlatformIDs) dlsym(lib_handle, "clGetPlatformIDs");
    fn_clGetDeviceIDs p_clGetDeviceIDs = (fn_clGetDeviceIDs) dlsym(lib_handle, "clGetDeviceIDs");
    fn_clGetDeviceInfo p_clGetDeviceInfo = (fn_clGetDeviceInfo) dlsym(lib_handle, "clGetDeviceInfo");
    fn_clCreateContext p_clCreateContext = (fn_clCreateContext) dlsym(lib_handle, "clCreateContext");
    fn_clCreateCommandQueue p_clCreateCommandQueue = (fn_clCreateCommandQueue) dlsym(lib_handle, "clCreateCommandQueue");
    fn_clCreateProgramWithSource p_clCreateProgramWithSource = (fn_clCreateProgramWithSource) dlsym(lib_handle, "clCreateProgramWithSource");
    fn_clBuildProgram p_clBuildProgram = (fn_clBuildProgram) dlsym(lib_handle, "clBuildProgram");
    fn_clCreateKernel p_clCreateKernel = (fn_clCreateKernel) dlsym(lib_handle, "clCreateKernel");

    p_clCreateBuffer = (fn_clCreateBuffer) dlsym(lib_handle, "clCreateBuffer");
    p_clEnqueueWriteBuffer = (fn_clEnqueueWriteBuffer) dlsym(lib_handle, "clEnqueueWriteBuffer");
    p_clEnqueueReadBuffer = (fn_clEnqueueReadBuffer) dlsym(lib_handle, "clEnqueueReadBuffer");
    p_clSetKernelArg = (fn_clSetKernelArg) dlsym(lib_handle, "clSetKernelArg");
    p_clEnqueueNDRangeKernel = (fn_clEnqueueNDRangeKernel) dlsym(lib_handle, "clEnqueueNDRangeKernel");
    p_clFinish = (fn_clFinish) dlsym(lib_handle, "clFinish");
    p_clReleaseMemObject = (fn_clReleaseMemObject) dlsym(lib_handle, "clReleaseMemObject");

    if (!p_clGetPlatformIDs || !p_clGetDeviceIDs || !p_clCreateContext || !p_clCreateBuffer) return 0;

    cl_platform_id platform;
    cl_uint num_platforms;
    if (p_clGetPlatformIDs(1, &platform, &num_platforms) != 0 || num_platforms == 0) return 0;

    cl_device_id device;
    cl_uint num_devices;
    if (p_clGetDeviceIDs(platform, 0xFFFFFFFF, 1, &device, &num_devices) != 0 || num_devices == 0) return 0;

    if (p_clGetDeviceInfo) {
        p_clGetDeviceInfo(device, 0x102B, sizeof(device_name), device_name, NULL);
    }

    cl_int err;
    context = p_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != 0) return 0;

    queue = p_clCreateCommandQueue(context, device, 0, &err);
    if (err != 0) return 0;

    cl_program program = p_clCreateProgramWithSource(context, 1, &gpu_source, NULL, &err);
    if (err != 0) return 0;

    err = p_clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != 0) return 0;

    kernel_matmul = p_clCreateKernel(program, "matmul", &err);
    kernel_batched_matmul = p_clCreateKernel(program, "batched_matmul", &err);
    kernel_add = p_clCreateKernel(program, "vec_add", &err);
    kernel_adamw = p_clCreateKernel(program, "adamw_step", &err);

    if (err != 0 || !kernel_matmul || !kernel_batched_matmul) return 0;

    gpu_initialized = 1;
    return 1;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nInit(JNIEnv *env, jclass cls) {
    return internal_gpu_init() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nIsAvailable(JNIEnv *env, jclass cls) {
    return (gpu_initialized || internal_gpu_init()) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nGetDeviceName(JNIEnv *env, jclass cls) {
    if (!gpu_initialized) internal_gpu_init();
    return (*env)->NewStringUTF(env, device_name);
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nMatMul(JNIEnv *env, jclass cls, jfloatArray a, jfloatArray b, jfloatArray c, jint m, jint n, jint k) {
    if (!gpu_initialized && !internal_gpu_init()) return JNI_FALSE;

    jfloat *ptrA = (*env)->GetFloatArrayElements(env, a, NULL);
    jfloat *ptrB = (*env)->GetFloatArrayElements(env, b, NULL);
    jfloat *ptrC = (*env)->GetFloatArrayElements(env, c, NULL);

    cl_int err;
    size_t sizeA = (size_t)m * k * sizeof(float);
    size_t sizeB = (size_t)k * n * sizeof(float);
    size_t sizeC = (size_t)m * n * sizeof(float);

    // Persistent OpenCL Buffer Allocation Pooling (Phase 2 & 3 Optimization)
    if (!bufA || capA < sizeA) {
        if (bufA) p_clReleaseMemObject(bufA);
        bufA = p_clCreateBuffer(context, 1, sizeA, NULL, &err);
        capA = sizeA;
    }
    if (!bufB || capB < sizeB) {
        if (bufB) p_clReleaseMemObject(bufB);
        bufB = p_clCreateBuffer(context, 1, sizeB, NULL, &err);
        capB = sizeB;
    }
    if (!bufC || capC < sizeC) {
        if (bufC) p_clReleaseMemObject(bufC);
        bufC = p_clCreateBuffer(context, 2, sizeC, NULL, &err);
        capC = sizeC;
    }

    p_clEnqueueWriteBuffer(queue, bufA, 1, 0, sizeA, ptrA, 0, NULL, NULL);
    p_clEnqueueWriteBuffer(queue, bufB, 1, 0, sizeB, ptrB, 0, NULL, NULL);

    p_clSetKernelArg(kernel_matmul, 0, sizeof(cl_mem), &bufA);
    p_clSetKernelArg(kernel_matmul, 1, sizeof(cl_mem), &bufB);
    p_clSetKernelArg(kernel_matmul, 2, sizeof(cl_mem), &bufC);
    p_clSetKernelArg(kernel_matmul, 3, sizeof(int), &m);
    p_clSetKernelArg(kernel_matmul, 4, sizeof(int), &n);
    p_clSetKernelArg(kernel_matmul, 5, sizeof(int), &k);

    size_t global[2] = { (size_t)m, (size_t)n };
    p_clEnqueueNDRangeKernel(queue, kernel_matmul, 2, NULL, global, NULL, 0, NULL, NULL);

    p_clEnqueueReadBuffer(queue, bufC, 1, 0, sizeC, ptrC, 0, NULL, NULL);
    p_clFinish(queue);

    (*env)->ReleaseFloatArrayElements(env, a, ptrA, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, b, ptrB, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, c, ptrC, 0);

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nBatchedMatMul(JNIEnv *env, jclass cls, jfloatArray a, jfloatArray b, jfloatArray c, jint numBatches, jint m, jint n, jint k) {
    if (!gpu_initialized && !internal_gpu_init()) return JNI_FALSE;

    jfloat *ptrA = (*env)->GetFloatArrayElements(env, a, NULL);
    jfloat *ptrB = (*env)->GetFloatArrayElements(env, b, NULL);
    jfloat *ptrC = (*env)->GetFloatArrayElements(env, c, NULL);

    cl_int err;
    size_t batchBytesA = (size_t)m * k * sizeof(float);
    size_t batchBytesB = (size_t)k * n * sizeof(float);
    size_t batchBytesC = (size_t)m * n * sizeof(float);

    size_t totalBytesA = batchBytesA * numBatches;
    size_t totalBytesB = batchBytesB * numBatches;
    size_t totalBytesC = batchBytesC * numBatches;

    if (!bufA || capA < totalBytesA) {
        if (bufA) p_clReleaseMemObject(bufA);
        bufA = p_clCreateBuffer(context, 1, totalBytesA, NULL, &err);
        capA = totalBytesA;
    }
    if (!bufB || capB < totalBytesB) {
        if (bufB) p_clReleaseMemObject(bufB);
        bufB = p_clCreateBuffer(context, 1, totalBytesB, NULL, &err);
        capB = totalBytesB;
    }
    if (!bufC || capC < totalBytesC) {
        if (bufC) p_clReleaseMemObject(bufC);
        bufC = p_clCreateBuffer(context, 2, totalBytesC, NULL, &err);
        capC = totalBytesC;
    }

    // Single PCIe Host -> GPU DMA transfer for all batch matrices
    p_clEnqueueWriteBuffer(queue, bufA, 1, 0, totalBytesA, ptrA, 0, NULL, NULL);
    p_clEnqueueWriteBuffer(queue, bufB, 1, 0, totalBytesB, ptrB, 0, NULL, NULL);

    p_clSetKernelArg(kernel_batched_matmul, 0, sizeof(cl_mem), &bufA);
    p_clSetKernelArg(kernel_batched_matmul, 1, sizeof(cl_mem), &bufB);
    p_clSetKernelArg(kernel_batched_matmul, 2, sizeof(cl_mem), &bufC);
    p_clSetKernelArg(kernel_batched_matmul, 3, sizeof(int), &m);
    p_clSetKernelArg(kernel_batched_matmul, 4, sizeof(int), &n);
    p_clSetKernelArg(kernel_batched_matmul, 5, sizeof(int), &k);

    size_t global[3] = { (size_t)m, (size_t)n, (size_t)numBatches };
    p_clEnqueueNDRangeKernel(queue, kernel_batched_matmul, 3, NULL, global, NULL, 0, NULL, NULL);

    p_clEnqueueReadBuffer(queue, bufC, 1, 0, totalBytesC, ptrC, 0, NULL, NULL);
    p_clFinish(queue);

    (*env)->ReleaseFloatArrayElements(env, a, ptrA, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, b, ptrB, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, c, ptrC, 0);

    return JNI_TRUE;
}

JNIEXPORT jlong JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nAllocBuffer(JNIEnv *env, jclass cls, jlong sizeBytes) {
    if (!gpu_initialized && !internal_gpu_init()) return 0;
    cl_int err;
    cl_mem buf = p_clCreateBuffer(context, 1, (size_t)sizeBytes, NULL, &err);
    if (err == 0 && buf != NULL) {
        return (jlong)buf;
    }
    return 0;
}

JNIEXPORT void JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nFreeBuffer(JNIEnv *env, jclass cls, jlong handle) {
    if (gpu_initialized && handle != 0) {
        p_clReleaseMemObject((cl_mem)handle);
    }
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nCopyToGPU(JNIEnv *env, jclass cls, jlong handle, jfloatArray hData, jlong sizeBytes) {
    if ((!gpu_initialized && !internal_gpu_init()) || handle == 0 || hData == NULL) return JNI_FALSE;
    jfloat *ptr = (*env)->GetPrimitiveArrayCritical(env, hData, NULL);
    if (!ptr) return JNI_FALSE;
    cl_int err = p_clEnqueueWriteBuffer(queue, (cl_mem)handle, 1, 0, (size_t)sizeBytes, ptr, 0, NULL, NULL);
    (*env)->ReleasePrimitiveArrayCritical(env, hData, ptr, JNI_ABORT);
    return err == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nCopyFromGPU(JNIEnv *env, jclass cls, jlong handle, jfloatArray hData, jlong sizeBytes) {
    if ((!gpu_initialized && !internal_gpu_init()) || handle == 0 || hData == NULL) return JNI_FALSE;
    jfloat *ptr = (*env)->GetPrimitiveArrayCritical(env, hData, NULL);
    if (!ptr) return JNI_FALSE;
    cl_int err = p_clEnqueueReadBuffer(queue, (cl_mem)handle, 1, 0, (size_t)sizeBytes, ptr, 0, NULL, NULL);
    (*env)->ReleasePrimitiveArrayCritical(env, hData, ptr, 0);
    return err == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_GPUMathEngine_nAdamWStep(
        JNIEnv *env, jclass cls,
        jlong pHandle, jlong gHandle, jlong mHandle, jlong vHandle,
        jint size, jfloat lr, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay,
        jfloat bc1, jfloat bc2) {

    if ((!gpu_initialized && !internal_gpu_init()) || !kernel_adamw || pHandle == 0 || gHandle == 0 || mHandle == 0 || vHandle == 0) {
        return JNI_FALSE;
    }

    cl_mem pBuf = (cl_mem)pHandle;
    cl_mem gBuf = (cl_mem)gHandle;
    cl_mem mBuf = (cl_mem)mHandle;
    cl_mem vBuf = (cl_mem)vHandle;

    p_clSetKernelArg(kernel_adamw, 0, sizeof(cl_mem), &pBuf);
    p_clSetKernelArg(kernel_adamw, 1, sizeof(cl_mem), &gBuf);
    p_clSetKernelArg(kernel_adamw, 2, sizeof(cl_mem), &mBuf);
    p_clSetKernelArg(kernel_adamw, 3, sizeof(cl_mem), &vBuf);
    p_clSetKernelArg(kernel_adamw, 4, sizeof(int), &size);
    p_clSetKernelArg(kernel_adamw, 5, sizeof(float), &lr);
    p_clSetKernelArg(kernel_adamw, 6, sizeof(float), &beta1);
    p_clSetKernelArg(kernel_adamw, 7, sizeof(float), &beta2);
    p_clSetKernelArg(kernel_adamw, 8, sizeof(float), &eps);
    p_clSetKernelArg(kernel_adamw, 9, sizeof(float), &weightDecay);
    p_clSetKernelArg(kernel_adamw, 10, sizeof(float), &bc1);
    p_clSetKernelArg(kernel_adamw, 11, sizeof(float), &bc2);

    size_t global = (size_t)size;
    cl_int err = p_clEnqueueNDRangeKernel(queue, kernel_adamw, 1, NULL, &global, NULL, 0, NULL, NULL);
    return err == 0 ? JNI_TRUE : JNI_FALSE;
}

