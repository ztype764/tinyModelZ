#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef unsigned long long CUdeviceptr;

static void* g_cudaHandle = NULL;
static CUcontext g_ctx = NULL;
static CUmodule g_module = NULL;
static CUfunction g_matmulFunc = NULL;
static CUfunction g_batchedMatmulFunc = NULL;
static char g_deviceName[256] = "Unknown CUDA Device";
static int g_initialized = 0;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, void*, void**, void**);
typedef CUresult (*cuCtxSynchronize_t)(void);

static cuInit_t f_cuInit = NULL;
static cuDeviceGet_t f_cuDeviceGet = NULL;
static cuDeviceGetName_t f_cuDeviceGetName = NULL;
static cuCtxCreate_t f_cuCtxCreate = NULL;
static cuModuleLoadData_t f_cuModuleLoadData = NULL;
static cuModuleGetFunction_t f_cuModuleGetFunction = NULL;
static cuMemAlloc_t f_cuMemAlloc = NULL;
static cuMemFree_t f_cuMemFree = NULL;
static cuMemcpyHtoD_t f_cuMemcpyHtoD = NULL;
static cuMemcpyDtoH_t f_cuMemcpyDtoH = NULL;
static cuLaunchKernel_t f_cuLaunchKernel = NULL;
static cuCtxSynchronize_t f_cuCtxSynchronize = NULL;

const char* cuda_ptx_code = 
".version 6.5\n"
".target sm_30\n"
".address_size 64\n"
"\n"
".visible .entry matmul_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
") {\n"
"    .reg .b32 %r<16>;\n"
"    .reg .b64 %rd<16>;\n"
"    .reg .f32 %f<8>;\n"
"    .reg .pred %p<4>;\n"
"\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    mov.u32 %r1, %ntid.x;\n"
"    mov.u32 %r2, %tid.x;\n"
"    mad.lo.s32 %r3, %r0, %r1, %r2;\n"

"    mov.u32 %r4, %ctaid.y;\n"
"    mov.u32 %r5, %ntid.y;\n"
"    mov.u32 %r6, %tid.y;\n"
"    mad.lo.s32 %r7, %r4, %r5, %r6;\n"

"    ld.param.u32 %r8, [param_M];\n"
"    ld.param.u32 %r9, [param_N];\n"
"    ld.param.u32 %r10, [param_K];\n"

"    setp.ge.s32 %p0, %r7, %r8;\n"
"    @%p0 bra END;\n"
"    setp.ge.s32 %p1, %r3, %r9;\n"
"    @%p1 bra END;\n"

"    ld.param.u64 %rd0, [param_A];\n"
"    ld.param.u64 %rd1, [param_B];\n"
"    ld.param.u64 %rd2, [param_C];\n"

"    mov.f32 %f0, 0f00000000;\n"
"    mov.u32 %r11, 0;\n"

"LOOP:\n"
"    setp.ge.s32 %p2, %r11, %r10;\n"
"    @%p2 bra WRITE;\n"

"    mad.lo.s32 %r12, %r7, %r10, %r11;\n"
"    mul.wide.s32 %rd3, %r12, 4;\n"
"    add.s64 %rd4, %rd0, %rd3;\n"
"    ld.global.f32 %f1, [%rd4];\n"

"    mad.lo.s32 %r13, %r11, %r9, %r3;\n"
"    mul.wide.s32 %rd5, %r13, 4;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f2, [%rd6];\n"

"    fma.rn.f32 %f0, %f1, %f2, %f0;\n"
"    add.s32 %r11, %r11, 1;\n"
"    bra LOOP;\n"

"WRITE:\n"
"    mad.lo.s32 %r14, %r7, %r9, %r3;\n"
"    mul.wide.s32 %rd7, %r14, 4;\n"
"    add.s64 %rd8, %rd2, %rd7;\n"
"    st.global.f32 [%rd8], %f0;\n"

"END:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry batched_matmul_kernel(\n"
"    .param .u64 param_A,\n"
"    .param .u64 param_B,\n"
"    .param .u64 param_C,\n"
"    .param .u32 param_M,\n"
"    .param .u32 param_N,\n"
"    .param .u32 param_K\n"
") {\n"
"    .reg .b32 %r<20>;\n"
"    .reg .b64 %rd<20>;\n"
"    .reg .f32 %f<8>;\n"
"    .reg .pred %p<4>;\n"

"    mov.u32 %r0, %ctaid.x;\n"
"    mov.u32 %r1, %ntid.x;\n"
"    mov.u32 %r2, %tid.x;\n"
"    mad.lo.s32 %r3, %r0, %r1, %r2;\n" // col

"    mov.u32 %r4, %ctaid.y;\n"
"    mov.u32 %r5, %ntid.y;\n"
"    mov.u32 %r6, %tid.y;\n"
"    mad.lo.s32 %r7, %r4, %r5, %r6;\n" // row

"    mov.u32 %r15, %ctaid.z;\n" // batch_idx

"    ld.param.u32 %r8, [param_M];\n"
"    ld.param.u32 %r9, [param_N];\n"
"    ld.param.u32 %r10, [param_K];\n"

"    setp.ge.s32 %p0, %r7, %r8;\n"
"    @%p0 bra BEND;\n"
"    setp.ge.s32 %p1, %r3, %r9;\n"
"    @%p1 bra BEND;\n"

"    ld.param.u64 %rd0, [param_A];\n"
"    ld.param.u64 %rd1, [param_B];\n"
"    ld.param.u64 %rd2, [param_C];\n"

    // Batch offsets
"    mul.lo.s32 %r16, %r8, %r10;\n" // M * K
"    mul.lo.s32 %r17, %r10, %r9;\n" // K * N
"    mul.lo.s32 %r18, %r8, %r9;\n"  // M * N

"    mad.lo.s32 %r16, %r15, %r16, 0;\n"
"    mul.wide.s32 %rd10, %r16, 4;\n"
"    add.s64 %rd0, %rd0, %rd10;\n"

"    mad.lo.s32 %r17, %r15, %r17, 0;\n"
"    mul.wide.s32 %rd11, %r17, 4;\n"
"    add.s64 %rd1, %rd1, %rd11;\n"

"    mad.lo.s32 %r18, %r15, %r18, 0;\n"
"    mul.wide.s32 %rd12, %r18, 4;\n"
"    add.s64 %rd2, %rd2, %rd12;\n"

"    mov.f32 %f0, 0f00000000;\n"
"    mov.u32 %r11, 0;\n"

"BLOOP:\n"
"    setp.ge.s32 %p2, %r11, %r10;\n"
"    @%p2 bra BWRITE;\n"

"    mad.lo.s32 %r12, %r7, %r10, %r11;\n"
"    mul.wide.s32 %rd3, %r12, 4;\n"
"    add.s64 %rd4, %rd0, %rd3;\n"
"    ld.global.f32 %f1, [%rd4];\n"

"    mad.lo.s32 %r13, %r11, %r9, %r3;\n"
"    mul.wide.s32 %rd5, %r13, 4;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f2, [%rd6];\n"

"    fma.rn.f32 %f0, %f1, %f2, %f0;\n"
"    add.s32 %r11, %r11, 1;\n"
"    bra BLOOP;\n"

"BWRITE:\n"
"    mad.lo.s32 %r14, %r7, %r9, %r3;\n"
"    mul.wide.s32 %rd7, %r14, 4;\n"
"    add.s64 %rd8, %rd2, %rd7;\n"
"    st.global.f32 [%rd8], %f0;\n"

"BEND:\n"
"    ret;\n"
"}\n";

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nInit(JNIEnv *env, jclass cls) {
    if (g_initialized) return JNI_TRUE;

    g_cudaHandle = dlopen("libcuda.so", RTLD_LAZY);
    if (!g_cudaHandle) return JNI_FALSE;

    f_cuInit = (cuInit_t)dlsym(g_cudaHandle, "cuInit");
    f_cuDeviceGet = (cuDeviceGet_t)dlsym(g_cudaHandle, "cuDeviceGet");
    f_cuDeviceGetName = (cuDeviceGetName_t)dlsym(g_cudaHandle, "cuDeviceGetName");
    f_cuCtxCreate = (cuCtxCreate_t)dlsym(g_cudaHandle, "cuCtxCreate_v2");
    f_cuModuleLoadData = (cuModuleLoadData_t)dlsym(g_cudaHandle, "cuModuleLoadData");
    f_cuModuleGetFunction = (cuModuleGetFunction_t)dlsym(g_cudaHandle, "cuModuleGetFunction");
    f_cuMemAlloc = (cuMemAlloc_t)dlsym(g_cudaHandle, "cuMemAlloc_v2");
    f_cuMemFree = (cuMemFree_t)dlsym(g_cudaHandle, "cuMemFree_v2");
    f_cuMemcpyHtoD = (cuMemcpyHtoD_t)dlsym(g_cudaHandle, "cuMemcpyHtoD_v2");
    f_cuMemcpyDtoH = (cuMemcpyDtoH_t)dlsym(g_cudaHandle, "cuMemcpyDtoH_v2");
    f_cuLaunchKernel = (cuLaunchKernel_t)dlsym(g_cudaHandle, "cuLaunchKernel");
    f_cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(g_cudaHandle, "cuCtxSynchronize");

    if (!f_cuInit || f_cuInit(0) != 0) return JNI_FALSE;

    CUdevice dev;
    if (f_cuDeviceGet(&dev, 0) != 0) return JNI_FALSE;

    f_cuDeviceGetName(g_deviceName, sizeof(g_deviceName), dev);
    strcat(g_deviceName, " (CUDA Native Driver)");

    if (f_cuCtxCreate(&g_ctx, 0, dev) != 0) return JNI_FALSE;

    if (f_cuModuleLoadData(&g_module, cuda_ptx_code) != 0) return JNI_FALSE;

    f_cuModuleGetFunction(&g_matmulFunc, g_module, "matmul_kernel");
    f_cuModuleGetFunction(&g_batchedMatmulFunc, g_module, "batched_matmul_kernel");

    g_initialized = 1;
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nIsAvailable(JNIEnv *env, jclass cls) {
    return g_initialized ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nGetDeviceName(JNIEnv *env, jclass cls) {
    return (*env)->NewStringUTF(env, g_deviceName);
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nMatMul(
        JNIEnv *env, jclass cls,
        jfloatArray a, jfloatArray b, jfloatArray c,
        jint m, jint n, jint k) {

    if (!g_initialized || !g_matmulFunc) return JNI_FALSE;

    jfloat *h_a = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *h_b = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
    jfloat *h_c = (*env)->GetPrimitiveArrayCritical(env, c, NULL);

    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    f_cuMemAlloc(&d_a, sizeA);
    f_cuMemAlloc(&d_b, sizeB);
    f_cuMemAlloc(&d_c, sizeC);

    f_cuMemcpyHtoD(d_a, h_a, sizeA);
    f_cuMemcpyHtoD(d_b, h_b, sizeB);

    int gridDimX = (n + 15) / 16;
    int gridDimY = (m + 15) / 16;
    void* args[] = { &d_a, &d_b, &d_c, &m, &n, &k };

    f_cuLaunchKernel(g_matmulFunc, gridDimX, gridDimY, 1, 16, 16, 1, 0, NULL, args, NULL);
    f_cuCtxSynchronize();

    f_cuMemcpyDtoH(h_c, d_c, sizeC);

    f_cuMemFree(d_a);
    f_cuMemFree(d_b);
    f_cuMemFree(d_c);

    (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nBatchedMatMul(
        JNIEnv *env, jclass cls,
        jfloatArray a, jfloatArray b, jfloatArray c,
        jint numBatches, jint m, jint n, jint k) {

    if (!g_initialized || !g_batchedMatmulFunc) return JNI_FALSE;

    jfloat *h_a = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *h_b = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
    jfloat *h_c = (*env)->GetPrimitiveArrayCritical(env, c, NULL);

    size_t sizeA = numBatches * m * k * sizeof(float);
    size_t sizeB = numBatches * k * n * sizeof(float);
    size_t sizeC = numBatches * m * n * sizeof(float);

    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    f_cuMemAlloc(&d_a, sizeA);
    f_cuMemAlloc(&d_b, sizeB);
    f_cuMemAlloc(&d_c, sizeC);

    f_cuMemcpyHtoD(d_a, h_a, sizeA);
    f_cuMemcpyHtoD(d_b, h_b, sizeB);

    int gridDimX = (n + 15) / 16;
    int gridDimY = (m + 15) / 16;
    int gridDimZ = numBatches;
    void* args[] = { &d_a, &d_b, &d_c, &m, &n, &k };

    f_cuLaunchKernel(g_batchedMatmulFunc, gridDimX, gridDimY, gridDimZ, 16, 16, 1, 0, NULL, args, NULL);
    f_cuCtxSynchronize();

    f_cuMemcpyDtoH(h_c, d_c, sizeC);

    f_cuMemFree(d_a);
    f_cuMemFree(d_b);
    f_cuMemFree(d_c);

    (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);

    return JNI_TRUE;
}
