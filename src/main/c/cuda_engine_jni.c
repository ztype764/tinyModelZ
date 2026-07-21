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
"}\n"
"\n"
".visible .entry adamw_step_kernel(\n"
"    .param .u64 param_P,\n"
"    .param .u64 param_G,\n"
"    .param .u64 param_M,\n"
"    .param .u64 param_V,\n"
"    .param .u32 param_size,\n"
"    .param .f32 param_lr,\n"
"    .param .f32 param_beta1,\n"
"    .param .f32 param_beta2,\n"
"    .param .f32 param_eps,\n"
"    .param .f32 param_weightDecay,\n"
"    .param .f32 param_bc1,\n"
"    .param .f32 param_bc2\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<12>;\n"
"    .reg .f32 %f<25>;\n"
"    .reg .pred %p<2>;\n"
"\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    mov.u32 %r1, %ntid.x;\n"
"    mov.u32 %r2, %tid.x;\n"
"    mad.lo.s32 %r3, %r0, %r1, %r2;\n"
"\n"
"    ld.param.u32 %r4, [param_size];\n"
"    setp.ge.s32 %p0, %r3, %r4;\n"
"    @%p0 bra AEND;\n"
"\n"
"    ld.param.u64 %rd0, [param_P];\n"
"    ld.param.u64 %rd1, [param_G];\n"
"    ld.param.u64 %rd2, [param_M];\n"
"    ld.param.u64 %rd3, [param_V];\n"
"\n"
"    mul.wide.s32 %rd4, %r3, 4;\n"
"    add.s64 %rd5, %rd0, %rd4;\n"
"    add.s64 %rd6, %rd1, %rd4;\n"
"    add.s64 %rd7, %rd2, %rd4;\n"
"    add.s64 %rd8, %rd3, %rd4;\n"
"\n"
"    ld.global.f32 %f0, [%rd5];\n"
"    ld.global.f32 %f1, [%rd6];\n"
"    ld.global.f32 %f2, [%rd7];\n"
"    ld.global.f32 %f3, [%rd8];\n"
"\n"
"    ld.param.f32 %f4, [param_lr];\n"
"    ld.param.f32 %f5, [param_beta1];\n"
"    ld.param.f32 %f6, [param_beta2];\n"
"    ld.param.f32 %f7, [param_eps];\n"
"    ld.param.f32 %f8, [param_weightDecay];\n"
"    ld.param.f32 %f9, [param_bc1];\n"
"    ld.param.f32 %f10, [param_bc2];\n"
"\n"
"    mov.f32 %f11, 1.0;\n"
"    sub.f32 %f11, %f11, %f5;\n"
"    mul.f32 %f12, %f5, %f2;\n"
"    fma.rn.f32 %f13, %f11, %f1, %f12;\n"
"    st.global.f32 [%rd7], %f13;\n"
"\n"
"    mov.f32 %f14, 1.0;\n"
"    sub.f32 %f14, %f14, %f6;\n"
"    mul.f32 %f15, %f1, %f1;\n"
"    mul.f32 %f16, %f6, %f3;\n"
"    fma.rn.f32 %f17, %f14, %f15, %f16;\n"
"    st.global.f32 [%rd8], %f17;\n"
"\n"
"    div.rn.f32 %f18, %f13, %f9;\n"
"    div.rn.f32 %f19, %f17, %f10;\n"
"\n"
"    sqrt.rn.f32 %f20, %f19;\n"
"    add.f32 %f20, %f20, %f7;\n"
"    div.rn.f32 %f21, %f4, %f20;\n"
"    mul.f32 %f21, %f21, %f18;\n"
"\n"
"    mul.f32 %f22, %f4, %f8;\n"
"    mul.f32 %f22, %f22, %f0;\n"
"\n"
"    sub.f32 %f0, %f0, %f21;\n"
"    sub.f32 %f0, %f0, %f22;\n"
"    st.global.f32 [%rd5], %f0;\n"
"\n"
"AEND:\n"
"    ret;\n"
"}\n";

static CUfunction g_adamwFunc = NULL;

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

static CUdeviceptr g_bufA = 0;
static CUdeviceptr g_bufB = 0;
static CUdeviceptr g_bufC = 0;
static size_t g_capA = 0;
static size_t g_capB = 0;
static size_t g_capC = 0;

JNIEXPORT jboolean JNICALL Java_com_tinymodelz_gpu_CUDAMathEngine_nMatMul(
        JNIEnv *env, jclass cls,
        jfloatArray a, jfloatArray b, jfloatArray c,
        jint m, jint n, jint k) {

    if (!g_initialized || !g_matmulFunc) return JNI_FALSE;

    jfloat *h_a = (*env)->GetPrimitiveArrayCritical(env, a, NULL);
    jfloat *h_b = (*env)->GetPrimitiveArrayCritical(env, b, NULL);
    jfloat *h_c = (*env)->GetPrimitiveArrayCritical(env, c, NULL);

    size_t sizeA = (size_t)m * k * sizeof(float);
    size_t sizeB = (size_t)k * n * sizeof(float);
    size_t sizeC = (size_t)m * n * sizeof(float);

    if (!g_bufA || g_capA < sizeA) {
        if (g_bufA) f_cuMemFree(g_bufA);
        if (f_cuMemAlloc(&g_bufA, sizeA) != 0) {
            (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, JNI_ABORT);
            return JNI_FALSE;
        }
        g_capA = sizeA;
    }
    if (!g_bufB || g_capB < sizeB) {
        if (g_bufB) f_cuMemFree(g_bufB);
        if (f_cuMemAlloc(&g_bufB, sizeB) != 0) {
            (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, JNI_ABORT);
            return JNI_FALSE;
        }
        g_capB = sizeB;
    }
    if (!g_bufC || g_capC < sizeC) {
        if (g_bufC) f_cuMemFree(g_bufC);
        if (f_cuMemAlloc(&g_bufC, sizeC) != 0) {
            (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, JNI_ABORT);
            return JNI_FALSE;
        }
        g_capC = sizeC;
    }

    f_cuMemcpyHtoD(g_bufA, h_a, sizeA);
    f_cuMemcpyHtoD(g_bufB, h_b, sizeB);

    int gridDimX = (n + 15) / 16;
    int gridDimY = (m + 15) / 16;
    void* args[] = { &g_bufA, &g_bufB, &g_bufC, &m, &n, &k };

    f_cuLaunchKernel(g_matmulFunc, gridDimX, gridDimY, 1, 16, 16, 1, 0, NULL, args, NULL);
    f_cuCtxSynchronize();

    f_cuMemcpyDtoH(h_c, g_bufC, sizeC);

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

    size_t sizeA = (size_t)numBatches * m * k * sizeof(float);
    size_t sizeB = (size_t)numBatches * k * n * sizeof(float);
    size_t sizeC = (size_t)numBatches * m * n * sizeof(float);

    if (!g_bufA || g_capA < sizeA) {
        if (g_bufA) f_cuMemFree(g_bufA);
        if (f_cuMemAlloc(&g_bufA, sizeA) != 0) {
            (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, JNI_ABORT);
            return JNI_FALSE;
        }
        g_capA = sizeA;
    }
    if (!g_bufB || g_capB < sizeB) {
        if (g_bufB) f_cuMemFree(g_bufB);
        if (f_cuMemAlloc(&g_bufB, sizeB) != 0) {
            (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, JNI_ABORT);
            return JNI_FALSE;
        }
        g_capB = sizeB;
    }
    if (!g_bufC || g_capC < sizeC) {
        if (g_bufC) f_cuMemFree(g_bufC);
        if (f_cuMemAlloc(&g_bufC, sizeC) != 0) {
            (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
            (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, JNI_ABORT);
            return JNI_FALSE;
        }
        g_capC = sizeC;
    }

    f_cuMemcpyHtoD(g_bufA, h_a, sizeA);
    f_cuMemcpyHtoD(g_bufB, h_b, sizeB);

    int gridDimX = (n + 15) / 16;
    int gridDimY = (m + 15) / 16;
    int gridDimZ = numBatches;
    void* args[] = { &g_bufA, &g_bufB, &g_bufC, &m, &n, &k };

    f_cuLaunchKernel(g_batchedMatmulFunc, gridDimX, gridDimY, gridDimZ, 16, 16, 1, 0, NULL, args, NULL);
    f_cuCtxSynchronize();

    f_cuMemcpyDtoH(h_c, g_bufC, sizeC);

    (*env)->ReleasePrimitiveArrayCritical(env, c, h_c, 0);
    (*env)->ReleasePrimitiveArrayCritical(env, b, h_b, JNI_ABORT);
    (*env)->ReleasePrimitiveArrayCritical(env, a, h_a, JNI_ABORT);

    return JNI_TRUE;
}

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

    f_cuLaunchKernel(g_adamwFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, NULL, args, NULL);
    return JNI_TRUE;
}

