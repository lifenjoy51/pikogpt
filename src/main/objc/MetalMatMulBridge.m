// MetalMatMulBridge.m
//
// pikogpt — Mac M3 GPU 가속. MatMul forward + backward 3 kernel을 Metal로 수행.
//
// 개선 누적:
//   A. MTLBuffer 풀 — size-bucket pool acquire/release.
//      GetPrimitiveArrayCritical로 JVM↔GPU memcpy 시간 단축.
//   B. Tiled kernel (custom, 16×16 threadgroup mem) — naive 대비 arithmetic intensity ↑.
//   E. MPSMatrixMultiplication (Apple vendor-tuned) — Metal Performance Shaders 프레임워크 호출.
//      simdgroup_matrix mma + shape별 best kernel 자동 선택. M3에서 단순 tiled 대비 2~5× 추가.
//      min(M,N,K) ≥ 16일 때 MPS, 그 외(매우 작은 shape)는 naive.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <jni.h>
#include <string.h>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeFwd       = nil;  // naive
static id<MTLComputePipelineState> g_pipeBwdA      = nil;  // naive
static id<MTLComputePipelineState> g_pipeBwdB      = nil;  // naive

// MPS는 매 호출 init하면 비싸므로 shape별 캐싱.
// key: "M_K_N_transL_transR_betaInt" (betaInt = 0 또는 1)
static NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *g_mpsOps = nil;

// MPS 사용 임계값. 한 차원이라도 16 미만이면 init overhead 대비 compute가 작아 naive로.
static const NSUInteger kMpsMin = 16;

// size-bucket MTLBuffer 풀. key=NSNumber(NSUInteger byte size), value=NSMutableArray.
// 한 버킷 최대 32개 cap. parallel trainer 대비 NSLock 보호.
static NSLock *g_poolLock = nil;
static NSMutableDictionary<NSNumber *, NSMutableArray<id<MTLBuffer>> *> *g_pool = nil;

static id<MTLBuffer> acquireBuffer(NSUInteger sizeBytes) {
    id<MTLBuffer> buf = nil;
    [g_poolLock lock];
    NSNumber *key = @(sizeBytes);
    NSMutableArray *arr = g_pool[key];
    if (arr != nil && arr.count > 0) {
        buf = arr.lastObject;
        [arr removeLastObject];
    }
    [g_poolLock unlock];
    if (buf == nil) {
        buf = [g_device newBufferWithLength:sizeBytes options:MTLResourceStorageModeShared];
    }
    return buf;
}

static void releaseBuffer(id<MTLBuffer> buf, NSUInteger sizeBytes) {
    if (buf == nil) return;
    [g_poolLock lock];
    NSNumber *key = @(sizeBytes);
    NSMutableArray *arr = g_pool[key];
    if (arr == nil) {
        arr = [NSMutableArray array];
        g_pool[key] = arr;
    }
    if (arr.count < 32) {
        [arr addObject:buf];
    }
    [g_poolLock unlock];
}

// naive shader source (작은 shape fallback). MPS init overhead가 compute보다 클 때 사용.
//   Forward:   C[i,j]  = Σ_k A[i,k] * B[k,j]
//   BackwardA: dA[i,k] += Σ_j gy[i,j] * B[k,j]  (turbo 동등성 — 누적)
//   BackwardB: dB[k,j] += Σ_i A[i,k]  * gy[i,j] (turbo 동등성 — 누적)
static NSString * const kShaderSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"\n"
    @"kernel void matmul_forward(\n"
    @"    device const float* A   [[buffer(0)]],\n"
    @"    device const float* B   [[buffer(1)]],\n"
    @"    device float*       C   [[buffer(2)]],\n"
    @"    constant uint3&     dims[[buffer(3)]],\n"
    @"    uint2 gid [[thread_position_in_grid]])\n"
    @"{\n"
    @"    uint M = dims.x, K = dims.y, N = dims.z;\n"
    @"    uint i = gid.y, j = gid.x;\n"
    @"    if (i >= M || j >= N) return;\n"
    @"    float acc = 0.0f;\n"
    @"    for (uint k = 0; k < K; ++k) {\n"
    @"        acc += A[i*K + k] * B[k*N + j];\n"
    @"    }\n"
    @"    C[i*N + j] = acc;\n"
    @"}\n"
    @"\n"
    @"kernel void matmul_backward_a(\n"
    @"    device const float* B   [[buffer(0)]],\n"
    @"    device const float* gy  [[buffer(1)]],\n"
    @"    device float*       dA  [[buffer(2)]],\n"
    @"    constant uint3&     dims[[buffer(3)]],\n"
    @"    uint2 gid [[thread_position_in_grid]])\n"
    @"{\n"
    @"    uint M = dims.x, K = dims.y, N = dims.z;\n"
    @"    uint i = gid.y, kk = gid.x;\n"
    @"    if (i >= M || kk >= K) return;\n"
    @"    float acc = 0.0f;\n"
    @"    for (uint j = 0; j < N; ++j) {\n"
    @"        acc += gy[i*N + j] * B[kk*N + j];\n"
    @"    }\n"
    @"    dA[i*K + kk] += acc;\n"
    @"}\n"
    @"\n"
    @"kernel void matmul_backward_b(\n"
    @"    device const float* A   [[buffer(0)]],\n"
    @"    device const float* gy  [[buffer(1)]],\n"
    @"    device float*       dB  [[buffer(2)]],\n"
    @"    constant uint3&     dims[[buffer(3)]],\n"
    @"    uint2 gid [[thread_position_in_grid]])\n"
    @"{\n"
    @"    uint M = dims.x, K = dims.y, N = dims.z;\n"
    @"    uint kk = gid.y, j = gid.x;\n"
    @"    if (kk >= K || j >= N) return;\n"
    @"    float acc = 0.0f;\n"
    @"    for (uint i = 0; i < M; ++i) {\n"
    @"        acc += A[i*K + kk] * gy[i*N + j];\n"
    @"    }\n"
    @"    dB[kk*N + j] += acc;\n"
    @"}\n"
    ;

static MTLSize pickThreadgroup(id<MTLComputePipelineState> pipe, NSUInteger gridX, NSUInteger gridY) {
    NSUInteger w = pipe.threadExecutionWidth;
    NSUInteger maxT = pipe.maxTotalThreadsPerThreadgroup;
    NSUInteger h = maxT / w;
    if (w > gridX && gridX > 0) w = gridX;
    if (h > gridY && gridY > 0) h = gridY;
    if (w == 0) w = 1;
    if (h == 0) h = 1;
    return MTLSizeMake(w, h, 1);
}

static void copyJavaToBuffer(JNIEnv *env, jfloatArray jArr, id<MTLBuffer> buf, NSUInteger sizeBytes) {
    void *src = (*env)->GetPrimitiveArrayCritical(env, jArr, NULL);
    memcpy([buf contents], src, sizeBytes);
    (*env)->ReleasePrimitiveArrayCritical(env, jArr, src, JNI_ABORT);
}

static void copyBufferToJava(JNIEnv *env, jfloatArray jArr, id<MTLBuffer> buf, NSUInteger sizeBytes) {
    void *dst = (*env)->GetPrimitiveArrayCritical(env, jArr, NULL);
    memcpy(dst, [buf contents], sizeBytes);
    (*env)->ReleasePrimitiveArrayCritical(env, jArr, dst, 0);
}

static id<MTLComputePipelineState> makePipeline(id<MTLLibrary> lib, NSString *name, NSError **err) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) {
        NSLog(@"[mps] function not found: %@", name);
        return nil;
    }
    id<MTLComputePipelineState> pipe = [g_device newComputePipelineStateWithFunction:fn error:err];
    if (pipe == nil) {
        NSLog(@"[mps] pipeline %@: %@", name, *err);
    }
    return pipe;
}

// MPSMatrixMultiplication 캐시. shape + transpose flag + beta로 key.
//
// forward    : transL=NO,  transR=NO,  beta=0.0  (C = A·B)
// backwardA  : transL=NO,  transR=YES, beta=1.0  (dA += gy · B^T)
// backwardB  : transL=YES, transR=NO,  beta=1.0  (dB += A^T · gy)
//
// resultRows / interiorColumns / resultColumns는 transpose 적용 *후*의 dimension.
static MPSMatrixMultiplication *acquireMpsOp(
    NSUInteger M, NSUInteger K, NSUInteger N,
    BOOL transL, BOOL transR, double beta)
{
    NSString *key = [NSString stringWithFormat:@"%lu_%lu_%lu_%d_%d_%d",
        (unsigned long)M, (unsigned long)K, (unsigned long)N,
        (int)transL, (int)transR, beta > 0.5 ? 1 : 0];
    [g_poolLock lock];
    MPSMatrixMultiplication *op = g_mpsOps[key];
    [g_poolLock unlock];
    if (op != nil) return op;

    NSUInteger resultRows = 0, interior = 0, resultCols = 0;
    if (!transL && !transR) {
        // A[M,K] · B[K,N] = C[M,N]
        resultRows = M; interior = K; resultCols = N;
    } else if (!transL && transR) {
        // gy[M,N] · B[K,N]^T = dA[M,K]
        resultRows = M; interior = N; resultCols = K;
    } else if (transL && !transR) {
        // A[M,K]^T · gy[M,N] = dB[K,N]
        resultRows = K; interior = M; resultCols = N;
    } else {
        // 미사용 (transL && transR)
        resultRows = K; interior = N; resultCols = M;
    }

    op = [[MPSMatrixMultiplication alloc] initWithDevice:g_device
        transposeLeft:transL
        transposeRight:transR
        resultRows:resultRows
        resultColumns:resultCols
        interiorColumns:interior
        alpha:1.0
        beta:beta];

    [g_poolLock lock];
    if (g_mpsOps[key] == nil) g_mpsOps[key] = op;
    else op = g_mpsOps[key];  // race: 다른 thread가 동시에 만들었으면 그걸 사용.
    [g_poolLock unlock];
    return op;
}

// JNI buffer를 MPSMatrix로 감싼다 (lightweight wrapper).
static MPSMatrix *wrapMatrix(id<MTLBuffer> buf, NSUInteger rows, NSUInteger cols) {
    MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                     columns:cols
                                                                    rowBytes:cols * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
}

JNIEXPORT jboolean JNICALL
Java_mps_jni_MetalMatMulBridge_nativeInit(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device == nil) {
            NSLog(@"[mps] MTLCreateSystemDefaultDevice returned nil");
            return JNI_FALSE;
        }
        g_queue = [g_device newCommandQueue];
        if (g_queue == nil) {
            NSLog(@"[mps] newCommandQueue failed");
            return JNI_FALSE;
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:kShaderSource options:nil error:&err];
        if (lib == nil) {
            NSLog(@"[mps] shader compile failed: %@", err);
            return JNI_FALSE;
        }
        g_pipeFwd  = makePipeline(lib, @"matmul_forward",    &err); if (!g_pipeFwd)  return JNI_FALSE;
        g_pipeBwdA = makePipeline(lib, @"matmul_backward_a", &err); if (!g_pipeBwdA) return JNI_FALSE;
        g_pipeBwdB = makePipeline(lib, @"matmul_backward_b", &err); if (!g_pipeBwdB) return JNI_FALSE;

        g_poolLock = [[NSLock alloc] init];
        g_pool = [NSMutableDictionary dictionary];
        g_mpsOps = [NSMutableDictionary dictionary];
        return JNI_TRUE;
    }
}

// naive kernel dispatch.
static void dispatchNaive(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pipe,
                          NSUInteger gridX, NSUInteger gridY) {
    [enc setComputePipelineState:pipe];
    MTLSize tg = pickThreadgroup(pipe, gridX, gridY);
    [enc dispatchThreads:MTLSizeMake(gridX, gridY, 1) threadsPerThreadgroup:tg];
}

static BOOL canUseMps(NSUInteger M, NSUInteger K, NSUInteger N) {
    return M >= kMpsMin && K >= kMpsMin && N >= kMpsMin;
}

// nativeMatmul(a, b, m, k, n, c) — C = A·B (덮어쓰기).
JNIEXPORT void JNICALL
Java_mps_jni_MetalMatMulBridge_nativeMatmul(JNIEnv *env, jclass cls,
    jfloatArray jA, jfloatArray jB, jint m, jint k, jint n, jfloatArray jC) {
    (void)cls;
    @autoreleasepool {
        const NSUInteger M = (NSUInteger)m, K = (NSUInteger)k, N = (NSUInteger)n;
        const NSUInteger sizeA = M * K * sizeof(float);
        const NSUInteger sizeB = K * N * sizeof(float);
        const NSUInteger sizeC = M * N * sizeof(float);

        id<MTLBuffer> bufA = acquireBuffer(sizeA);
        id<MTLBuffer> bufB = acquireBuffer(sizeB);
        id<MTLBuffer> bufC = acquireBuffer(sizeC);

        copyJavaToBuffer(env, jA, bufA, sizeA);
        copyJavaToBuffer(env, jB, bufB, sizeB);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        if (canUseMps(M, K, N)) {
            // MPS path. beta=0 → C 덮어쓰기 (C 초기화 불필요).
            MPSMatrixMultiplication *op = acquireMpsOp(M, K, N, NO, NO, 0.0);
            MPSMatrix *matA = wrapMatrix(bufA, M, K);
            MPSMatrix *matB = wrapMatrix(bufB, K, N);
            MPSMatrix *matC = wrapMatrix(bufC, M, N);
            [op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        } else {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            uint32_t dims[3] = { (uint32_t)M, (uint32_t)K, (uint32_t)N };
            [enc setBytes:dims length:sizeof(dims) atIndex:3];
            dispatchNaive(enc, g_pipeFwd, N, M);
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        copyBufferToJava(env, jC, bufC, sizeC);

        releaseBuffer(bufA, sizeA);
        releaseBuffer(bufB, sizeB);
        releaseBuffer(bufC, sizeC);
    }
}

// nativeMatmulBackwardA(b, gy, m, k, n, dA) — dA += gy · B^T (누적).
JNIEXPORT void JNICALL
Java_mps_jni_MetalMatMulBridge_nativeMatmulBackwardA(JNIEnv *env, jclass cls,
    jfloatArray jB, jfloatArray jGy, jint m, jint k, jint n, jfloatArray jDA) {
    (void)cls;
    @autoreleasepool {
        const NSUInteger M = (NSUInteger)m, K = (NSUInteger)k, N = (NSUInteger)n;
        const NSUInteger sizeB  = K * N * sizeof(float);
        const NSUInteger sizeGy = M * N * sizeof(float);
        const NSUInteger sizeDA = M * K * sizeof(float);

        id<MTLBuffer> bufB  = acquireBuffer(sizeB);
        id<MTLBuffer> bufGy = acquireBuffer(sizeGy);
        id<MTLBuffer> bufDA = acquireBuffer(sizeDA);

        copyJavaToBuffer(env, jB,  bufB,  sizeB);
        copyJavaToBuffer(env, jGy, bufGy, sizeGy);
        copyJavaToBuffer(env, jDA, bufDA, sizeDA);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        if (canUseMps(M, K, N)) {
            // beta=1 → dA += gy · B^T (누적)
            MPSMatrixMultiplication *op = acquireMpsOp(M, K, N, NO, YES, 1.0);
            MPSMatrix *matGy = wrapMatrix(bufGy, M, N);
            MPSMatrix *matB  = wrapMatrix(bufB,  K, N);
            MPSMatrix *matDA = wrapMatrix(bufDA, M, K);
            [op encodeToCommandBuffer:cmd leftMatrix:matGy rightMatrix:matB resultMatrix:matDA];
        } else {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:bufB  offset:0 atIndex:0];
            [enc setBuffer:bufGy offset:0 atIndex:1];
            [enc setBuffer:bufDA offset:0 atIndex:2];
            uint32_t dims[3] = { (uint32_t)M, (uint32_t)K, (uint32_t)N };
            [enc setBytes:dims length:sizeof(dims) atIndex:3];
            dispatchNaive(enc, g_pipeBwdA, K, M);
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        copyBufferToJava(env, jDA, bufDA, sizeDA);

        releaseBuffer(bufB,  sizeB);
        releaseBuffer(bufGy, sizeGy);
        releaseBuffer(bufDA, sizeDA);
    }
}

// fp16 mixed precision helpers. Apple Silicon clang에서 __fp16은 NEON half native.
// -O3에서 단순 cast loop가 자동으로 NEON `vcvt_f32_f16` SIMD로 변환됨.
static inline void fp32ToFp16(const float *src, __fp16 *dst, NSUInteger count) {
    for (NSUInteger i = 0; i < count; ++i) dst[i] = (__fp16)src[i];
}

static inline void fp16ToFp32(const __fp16 *src, float *dst, NSUInteger count) {
    for (NSUInteger i = 0; i < count; ++i) dst[i] = (float)src[i];
}

// nativeMatmulFp16 — forward 한정 fp16 mixed precision. backward는 fp32 그대로.
// 학습 안정성 위험 큼 (특히 SwiGLU/RMSNorm 등) — 사용자가 첫 100 iter loss curve로 검증 후 채택.
JNIEXPORT void JNICALL
Java_mps_jni_MetalMatMulBridge_nativeMatmulFp16(JNIEnv *env, jclass cls,
    jfloatArray jA, jfloatArray jB, jint m, jint k, jint n, jfloatArray jC) {
    (void)cls;
    @autoreleasepool {
        const NSUInteger M = (NSUInteger)m, K = (NSUInteger)k, N = (NSUInteger)n;
        const NSUInteger sizeA = M * K * sizeof(__fp16);
        const NSUInteger sizeB = K * N * sizeof(__fp16);
        const NSUInteger sizeC = M * N * sizeof(__fp16);

        id<MTLBuffer> bufA = acquireBuffer(sizeA);
        id<MTLBuffer> bufB = acquireBuffer(sizeB);
        id<MTLBuffer> bufC = acquireBuffer(sizeC);

        // fp32 → fp16 (input A, B). Critical 영역은 cast 동안만.
        {
            void *src = (*env)->GetPrimitiveArrayCritical(env, jA, NULL);
            fp32ToFp16((const float *)src, (__fp16 *)[bufA contents], M * K);
            (*env)->ReleasePrimitiveArrayCritical(env, jA, src, JNI_ABORT);
        }
        {
            void *src = (*env)->GetPrimitiveArrayCritical(env, jB, NULL);
            fp32ToFp16((const float *)src, (__fp16 *)[bufB contents], K * N);
            (*env)->ReleasePrimitiveArrayCritical(env, jB, src, JNI_ABORT);
        }

        // MPS dispatch — fp16 dtype. MPSMatrixMultiplication 객체는 dtype 무관, descriptor가 결정.
        MPSMatrixMultiplication *op = acquireMpsOp(M, K, N, NO, NO, 0.0);
        MPSMatrixDescriptor *aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                          columns:K
                                                                         rowBytes:K * sizeof(__fp16)
                                                                         dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                          columns:N
                                                                         rowBytes:N * sizeof(__fp16)
                                                                         dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                          columns:N
                                                                         rowBytes:N * sizeof(__fp16)
                                                                         dataType:MPSDataTypeFloat16];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:aDesc];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:bDesc];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:cDesc];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        [op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];

        // fp16 → fp32 (output C).
        {
            void *dst = (*env)->GetPrimitiveArrayCritical(env, jC, NULL);
            fp16ToFp32((const __fp16 *)[bufC contents], (float *)dst, M * N);
            (*env)->ReleasePrimitiveArrayCritical(env, jC, dst, 0);
        }

        releaseBuffer(bufA, sizeA);
        releaseBuffer(bufB, sizeB);
        releaseBuffer(bufC, sizeC);
    }
}

// nativeMatmulBackwardB(a, gy, m, k, n, dB) — dB += A^T · gy (누적).
JNIEXPORT void JNICALL
Java_mps_jni_MetalMatMulBridge_nativeMatmulBackwardB(JNIEnv *env, jclass cls,
    jfloatArray jA, jfloatArray jGy, jint m, jint k, jint n, jfloatArray jDB) {
    (void)cls;
    @autoreleasepool {
        const NSUInteger M = (NSUInteger)m, K = (NSUInteger)k, N = (NSUInteger)n;
        const NSUInteger sizeA  = M * K * sizeof(float);
        const NSUInteger sizeGy = M * N * sizeof(float);
        const NSUInteger sizeDB = K * N * sizeof(float);

        id<MTLBuffer> bufA  = acquireBuffer(sizeA);
        id<MTLBuffer> bufGy = acquireBuffer(sizeGy);
        id<MTLBuffer> bufDB = acquireBuffer(sizeDB);

        copyJavaToBuffer(env, jA,  bufA,  sizeA);
        copyJavaToBuffer(env, jGy, bufGy, sizeGy);
        copyJavaToBuffer(env, jDB, bufDB, sizeDB);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        if (canUseMps(M, K, N)) {
            // beta=1 → dB += A^T · gy (누적)
            MPSMatrixMultiplication *op = acquireMpsOp(M, K, N, YES, NO, 1.0);
            MPSMatrix *matA  = wrapMatrix(bufA,  M, K);
            MPSMatrix *matGy = wrapMatrix(bufGy, M, N);
            MPSMatrix *matDB = wrapMatrix(bufDB, K, N);
            [op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matGy resultMatrix:matDB];
        } else {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:bufA  offset:0 atIndex:0];
            [enc setBuffer:bufGy offset:0 atIndex:1];
            [enc setBuffer:bufDB offset:0 atIndex:2];
            uint32_t dims[3] = { (uint32_t)M, (uint32_t)K, (uint32_t)N };
            [enc setBytes:dims length:sizeof(dims) atIndex:3];
            dispatchNaive(enc, g_pipeBwdB, N, K);
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        copyBufferToJava(env, jDB, bufDB, sizeDB);

        releaseBuffer(bufA,  sizeA);
        releaseBuffer(bufGy, sizeGy);
        releaseBuffer(bufDB, sizeDB);
    }
}
