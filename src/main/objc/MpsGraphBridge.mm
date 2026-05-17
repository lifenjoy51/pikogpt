// MpsGraphBridge.mm — MPSGraph 기반 GPU 100% residence 학습 backend의 JNI bridge.
//
// Phase 1: 인프라 (완료)
// Phase 2.1: weight registration (진행 중) — turbo와 같은 paramIndex 순서로 GPU MTLBuffer에 저장.
// Phase 2.2~5: forward/backward/AdamW graph (예정)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <jni.h>
#include <stdint.h>

// ============================================================================
// WeightSlot — GPU resident weight (MPSGraphTensorData)
// ============================================================================
@interface PikoWeightSlot : NSObject
@property (nonatomic, strong) id<MTLBuffer> buffer;       // weight (GPU resident)
@property (nonatomic, strong) id<MTLBuffer> gradBuffer;   // gradient (GPU resident, lazy)
@property (nonatomic, strong) id<MTLBuffer> mBuffer;      // AdamW m state (lazy)
@property (nonatomic, strong) id<MTLBuffer> vBuffer;      // AdamW v state (lazy)
@property (nonatomic, strong) NSArray<NSNumber *> *shape;
@property (nonatomic, assign) NSUInteger numel;
@end
@implementation PikoWeightSlot
@end

// ============================================================================
// PikoMpsGraphSession
// ============================================================================
@interface PikoMpsGraphSession : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) MPSGraph *graph;

// Phase 6 — cached step graph (built once, reused per iter)
@property (nonatomic, strong) MPSGraph *stepGraph;
@property (nonatomic, assign) NSInteger cachedT;  // 0 = not built
@property (nonatomic, strong) MPSGraphTensor *stepIdsPh;
@property (nonatomic, strong) MPSGraphTensor *stepTgtPh;
@property (nonatomic, strong) MPSGraphTensor *stepCosPh;
@property (nonatomic, strong) MPSGraphTensor *stepSinPh;
@property (nonatomic, strong) MPSGraphTensor *stepMaskPh;
@property (nonatomic, strong) MPSGraphTensor *stepLRPh;
@property (nonatomic, strong) MPSGraphTensor *stepBc1Ph;
@property (nonatomic, strong) MPSGraphTensor *stepBc2Ph;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepWPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepMPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepVPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepWNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepMNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepVNew;
@property (nonatomic, strong) MPSGraphTensor *stepLoss;

// Weights: paramIndex → GPU buffer + shape
@property (nonatomic, strong) NSMutableArray *weights;  // PikoWeightSlot* or NSNull placeholder

// 모델 hyperparam
@property (nonatomic, assign) int numLayers;
@property (nonatomic, assign) int embedDim;
@property (nonatomic, assign) int numHeads;
@property (nonatomic, assign) int blockSize;
@property (nonatomic, assign) int vocab;
@property (nonatomic, assign) int batchSize;
@property (nonatomic, assign) BOOL useSwiglu;
@property (nonatomic, assign) BOOL useRope;
@property (nonatomic, assign) BOOL tieWeights;
@end
@implementation PikoMpsGraphSession
@end

// ============================================================================
// Helpers
// ============================================================================
static PikoMpsGraphSession *sessionFromHandle(jlong handle) {
    if (handle == 0) return nil;
    return (__bridge PikoMpsGraphSession *)(void *)(intptr_t)handle;
}

// ============================================================================
// JNI exports — package mps.MpsGraphSession
// ============================================================================
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jboolean JNICALL
Java_mps_MpsGraphSession_nativeInit(JNIEnv *env, jclass clazz) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return JNI_FALSE;
        MPSGraph *g = [[MPSGraph alloc] init];
        if (!g) return JNI_FALSE;
        return JNI_TRUE;
    }
}

JNIEXPORT jlong JNICALL
Java_mps_MpsGraphSession_nativeCreateSession(
    JNIEnv *env, jclass clazz,
    jint numLayers, jint embedDim, jint numHeads, jint blockSize,
    jint vocab, jint batchSize,
    jboolean useSwiglu, jboolean useRope, jboolean tieWeights) {
    @autoreleasepool {
        PikoMpsGraphSession *s = [[PikoMpsGraphSession alloc] init];
        s.device = MTLCreateSystemDefaultDevice();
        if (!s.device) return 0L;
        s.commandQueue = [s.device newCommandQueue];
        if (!s.commandQueue) return 0L;
        s.graph = [[MPSGraph alloc] init];
        if (!s.graph) return 0L;

        s.weights = [NSMutableArray array];
        s.numLayers = numLayers;
        s.embedDim = embedDim;
        s.numHeads = numHeads;
        s.blockSize = blockSize;
        s.vocab = vocab;
        s.batchSize = batchSize;
        s.useSwiglu = useSwiglu == JNI_TRUE;
        s.useRope = useRope == JNI_TRUE;
        s.tieWeights = tieWeights == JNI_TRUE;

        return (jlong)(intptr_t)CFBridgingRetain(s);
    }
}

JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeDestroySession(JNIEnv *env, jclass clazz, jlong handle) {
    @autoreleasepool {
        if (handle == 0) return;
        PikoMpsGraphSession *s = (__bridge_transfer PikoMpsGraphSession *)(void *)(intptr_t)handle;
        (void)s;
    }
}

/**
 * Weight를 GPU resident MTLBuffer에 저장. paramIndex는 turbo `TurboPikoGPT.parameters()` 순서와 동일.
 *
 * shape는 int[] 형태로 받음 (rank ≤ 4 가정). MPSGraph 사용 시 shape 정보가 필요해서 함께 보관.
 *
 * 호출 시 weights 배열을 paramIndex까지 grow하고 NULL slot 채움 → set slot.
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeLoadWeights(
    JNIEnv *env, jclass clazz, jlong handle,
    jint paramIndex, jfloatArray data, jintArray shapeArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;

        jsize dataLen = env->GetArrayLength(data);
        jsize shapeLen = env->GetArrayLength(shapeArr);
        jint *shapeData = env->GetIntArrayElements(shapeArr, NULL);

        NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:shapeLen];
        NSUInteger numel = 1;
        for (jsize i = 0; i < shapeLen; i++) {
            int d = shapeData[i];
            [shape addObject:@(d)];
            numel *= (NSUInteger)d;
        }
        env->ReleaseIntArrayElements(shapeArr, shapeData, JNI_ABORT);

        if (numel != (NSUInteger)dataLen) {
            env->ThrowNew(
                env->FindClass("java/lang/IllegalArgumentException"),
                "data length != product(shape)");
            return;
        }

        NSUInteger byteLen = numel * sizeof(float);
        id<MTLBuffer> buf = [s.device newBufferWithLength:byteLen
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> grad = [s.device newBufferWithLength:byteLen
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> m = [s.device newBufferWithLength:byteLen
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> v = [s.device newBufferWithLength:byteLen
                                                options:MTLResourceStorageModeShared];
        if (!buf || !grad || !m || !v) {
            env->ThrowNew(
                env->FindClass("java/lang/OutOfMemoryError"),
                "MTLBuffer alloc failed");
            return;
        }
        jfloat *dataPtr = (jfloat *)env->GetPrimitiveArrayCritical(data, NULL);
        memcpy([buf contents], dataPtr, byteLen);
        env->ReleasePrimitiveArrayCritical(data, dataPtr, JNI_ABORT);
        memset([grad contents], 0, byteLen);
        memset([m contents], 0, byteLen);
        memset([v contents], 0, byteLen);

        while ((NSInteger)s.weights.count <= paramIndex) {
            [s.weights addObject:[NSNull null]];
        }
        PikoWeightSlot *slot = [[PikoWeightSlot alloc] init];
        slot.buffer = buf;
        slot.gradBuffer = grad;
        slot.mBuffer = m;
        slot.vBuffer = v;
        slot.shape = [shape copy];
        slot.numel = numel;
        s.weights[paramIndex] = slot;
    }
}

JNIEXPORT jint JNICALL
Java_mps_MpsGraphSession_nativeWeightCount(JNIEnv *env, jclass clazz, jlong handle) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return 0;
        return (jint)s.weights.count;
    }
}

// ============================================================================
// Phase 2.2 — Forward graph build (단계 1: embedding lookup만)
//
// 이 단계의 graph:
//   tokenIds [B*T] int32 → gather → output [B*T, embedDim]
//   tokenEmbedding weight: paramIndex 0
//
// Phase 2.3+ 에서 LayerNorm/MHA/SwiGLU/lm_head 추가하며 graph 확장.
// ============================================================================

/**
 * Embedding lookup만 graph build + run. Phase 2.2 step 1 검증용.
 *
 * input:  tokenIds (int32 array, length = T or B*T)
 * output: float array (T*embedDim 또는 B*T*embedDim)
 *
 * 매 호출마다 graph 생성/실행 (cache 없음 — Phase 2.5에서 executable cache 추가).
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunEmbeddingForward(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jfloatArray outputArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) {
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "session null");
            return;
        }
        if (s.weights.count == 0 || s.weights[0] == [NSNull null]) {
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                          "tokenEmbedding (paramIndex 0) not loaded");
            return;
        }

        PikoWeightSlot *embSlot = (PikoWeightSlot *)s.weights[0];
        jsize tokLen = env->GetArrayLength(tokenIdsArr);
        jsize outLen = env->GetArrayLength(outputArr);
        NSUInteger embedDim = [embSlot.shape[1] unsignedIntegerValue];
        if ((NSUInteger)outLen != (NSUInteger)tokLen * embedDim) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "output size != tokenIds * embedDim");
            return;
        }

        MPSGraph *g = [[MPSGraph alloc] init];

        // Placeholders
        MPSGraphTensor *embPh = [g placeholderWithShape:embSlot.shape
                                                dataType:MPSDataTypeFloat32
                                                    name:@"emb"];
        MPSGraphTensor *idsPh = [g placeholderWithShape:@[@(tokLen)]
                                                dataType:MPSDataTypeInt32
                                                    name:@"ids"];

        // gather(emb, ids, axis=0)
        MPSGraphTensor *out = [g gatherWithUpdatesTensor:embPh
                                          indicesTensor:idsPh
                                                   axis:0
                                        batchDimensions:0
                                                   name:nil];

        // Feeds
        MPSGraphTensorData *embData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:embSlot.buffer
                                                    shape:embSlot.shape
                                                 dataType:MPSDataTypeFloat32];

        // tokenIds buffer 생성 + copy
        NSUInteger idsBytes = tokLen * sizeof(int32_t);
        id<MTLBuffer> idsBuf = [s.device newBufferWithLength:idsBytes
                                                     options:MTLResourceStorageModeShared];
        {
            jint *idsPtr = (jint *)env->GetPrimitiveArrayCritical(tokenIdsArr, NULL);
            memcpy([idsBuf contents], idsPtr, idsBytes);
            env->ReleasePrimitiveArrayCritical(tokenIdsArr, idsPtr, JNI_ABORT);
        }
        MPSGraphTensorData *idsData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf
                                                    shape:@[@(tokLen)]
                                                 dataType:MPSDataTypeInt32];

        // Output buffer
        NSUInteger outBytes = outLen * sizeof(float);
        id<MTLBuffer> outBuf = [s.device newBufferWithLength:outBytes
                                                     options:MTLResourceStorageModeShared];
        MPSGraphTensorData *outData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                    shape:@[@(tokLen), @(embedDim)]
                                                 dataType:MPSDataTypeFloat32];

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            @{embPh: embData, idsPh: idsData};
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *targets =
            @{out: outData};

        [g runWithMTLCommandQueue:s.commandQueue
                            feeds:feeds
                 targetOperations:nil
                resultsDictionary:targets];

        // Copy out
        jfloat *outPtr = (jfloat *)env->GetPrimitiveArrayCritical(outputArr, NULL);
        memcpy(outPtr, [outBuf contents], outBytes);
        env->ReleasePrimitiveArrayCritical(outputArr, outPtr, 0);
    }
}

// ============================================================================
// Phase 2.2 step 2 — LayerNorm forward (단위 검증)
//
// y = gamma * (x - mean) / sqrt(var + eps) + beta  (axis=-1 per row)
// MPSGraph mean/variance는 axes 따라 자동 reduction.
// ============================================================================
static MPSGraphTensor *buildLayerNorm(MPSGraph *g,
                                      MPSGraphTensor *x,
                                      MPSGraphTensor *gamma,
                                      MPSGraphTensor *beta,
                                      float eps,
                                      NSInteger axisDim) {
    NSArray<NSNumber *> *axes = @[@(axisDim)];
    MPSGraphTensor *mean = [g meanOfTensor:x axes:axes name:nil];
    MPSGraphTensor *diff = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
    MPSGraphTensor *diffSq = [g squareWithTensor:diff name:nil];
    MPSGraphTensor *variance = [g meanOfTensor:diffSq axes:axes name:nil];
    return [g normalizationWithTensor:x
                           meanTensor:mean
                       varianceTensor:variance
                          gammaTensor:gamma
                           betaTensor:beta
                              epsilon:eps
                                 name:nil];
}

/**
 * LayerNorm forward 단독 (gamma=paramGamma, beta=paramBeta).
 * input shape: [T, C]. output shape: [T, C].
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunLayerNormForward(
    JNIEnv *env, jclass clazz, jlong handle,
    jint paramGamma, jint paramBeta,
    jint T, jint C, jfloat eps,
    jfloatArray inputArr, jfloatArray outputArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) {
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "session null");
            return;
        }
        if ((NSInteger)s.weights.count <= paramGamma || (NSInteger)s.weights.count <= paramBeta) {
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                          "gamma/beta not loaded");
            return;
        }
        PikoWeightSlot *gSlot = (PikoWeightSlot *)s.weights[paramGamma];
        PikoWeightSlot *bSlot = (PikoWeightSlot *)s.weights[paramBeta];

        MPSGraph *g = [[MPSGraph alloc] init];
        MPSGraphTensor *xPh = [g placeholderWithShape:@[@(T), @(C)]
                                             dataType:MPSDataTypeFloat32 name:@"x"];
        MPSGraphTensor *gPh = [g placeholderWithShape:gSlot.shape
                                             dataType:MPSDataTypeFloat32 name:@"gamma"];
        MPSGraphTensor *bPh = [g placeholderWithShape:bSlot.shape
                                             dataType:MPSDataTypeFloat32 name:@"beta"];
        MPSGraphTensor *out = buildLayerNorm(g, xPh, gPh, bPh, (float)eps, /*axisDim=*/1);

        // Feeds: input buffer
        NSUInteger inputBytes = (NSUInteger)T * C * sizeof(float);
        id<MTLBuffer> inBuf = [s.device newBufferWithLength:inputBytes
                                                    options:MTLResourceStorageModeShared];
        {
            jfloat *inPtr = (jfloat *)env->GetPrimitiveArrayCritical(inputArr, NULL);
            memcpy([inBuf contents], inPtr, inputBytes);
            env->ReleasePrimitiveArrayCritical(inputArr, inPtr, JNI_ABORT);
        }
        MPSGraphTensorData *xData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:inBuf shape:@[@(T), @(C)] dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *gData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:gSlot.buffer shape:gSlot.shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *bData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bSlot.buffer shape:bSlot.shape dataType:MPSDataTypeFloat32];

        id<MTLBuffer> outBuf = [s.device newBufferWithLength:inputBytes
                                                     options:MTLResourceStorageModeShared];
        MPSGraphTensorData *outData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outBuf shape:@[@(T), @(C)] dataType:MPSDataTypeFloat32];

        [g runWithMTLCommandQueue:s.commandQueue
                            feeds:@{xPh: xData, gPh: gData, bPh: bData}
                 targetOperations:nil
                resultsDictionary:@{out: outData}];

        jfloat *outPtr = (jfloat *)env->GetPrimitiveArrayCritical(outputArr, NULL);
        memcpy(outPtr, [outBuf contents], inputBytes);
        env->ReleasePrimitiveArrayCritical(outputArr, outPtr, 0);
    }
}

// ============================================================================
// Phase 2.2 step 3 — Linear projection forward (turbo TurboLinear와 동등)
//
// y = x @ weight.T + bias
// weight shape: [outF, inF]. transpose 후 matmul.
// ============================================================================
static MPSGraphTensor *buildLinear(MPSGraph *g,
                                   MPSGraphTensor *x,         // [T, inF]
                                   MPSGraphTensor *weight,    // [outF, inF]
                                   MPSGraphTensor *bias) {    // [outF] or nil
    MPSGraphTensor *wT = [g transposeTensor:weight
                                  dimension:0
                              withDimension:1
                                       name:nil];
    MPSGraphTensor *y = [g matrixMultiplicationWithPrimaryTensor:x
                                                 secondaryTensor:wT
                                                            name:nil];
    if (bias) {
        y = [g additionWithPrimaryTensor:y secondaryTensor:bias name:nil];
    }
    return y;
}

/**
 * Linear forward 단독. y = x @ weight.T + bias.
 *   weight: paramWeight (shape [outF, inF])
 *   bias:   paramBias (shape [outF]) 또는 -1이면 없음
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunLinearForward(
    JNIEnv *env, jclass clazz, jlong handle,
    jint paramWeight, jint paramBias,
    jint T, jint inF, jint outF,
    jfloatArray inputArr, jfloatArray outputArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *wSlot = (PikoWeightSlot *)s.weights[paramWeight];
        PikoWeightSlot *bSlot = (paramBias >= 0) ? (PikoWeightSlot *)s.weights[paramBias] : nil;

        MPSGraph *g = [[MPSGraph alloc] init];
        MPSGraphTensor *xPh = [g placeholderWithShape:@[@(T), @(inF)]
                                             dataType:MPSDataTypeFloat32 name:@"x"];
        MPSGraphTensor *wPh = [g placeholderWithShape:wSlot.shape
                                             dataType:MPSDataTypeFloat32 name:@"w"];
        MPSGraphTensor *bPh = nil;
        if (bSlot) {
            bPh = [g placeholderWithShape:bSlot.shape
                                 dataType:MPSDataTypeFloat32 name:@"b"];
        }
        MPSGraphTensor *out = buildLinear(g, xPh, wPh, bPh);

        NSUInteger inBytes = (NSUInteger)T * inF * sizeof(float);
        NSUInteger outBytes = (NSUInteger)T * outF * sizeof(float);
        id<MTLBuffer> inBuf = [s.device newBufferWithLength:inBytes
                                                    options:MTLResourceStorageModeShared];
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(inputArr, NULL);
            memcpy([inBuf contents], p, inBytes);
            env->ReleasePrimitiveArrayCritical(inputArr, p, JNI_ABORT);
        }
        id<MTLBuffer> outBuf = [s.device newBufferWithLength:outBytes
                                                     options:MTLResourceStorageModeShared];

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[xPh] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:inBuf shape:@[@(T), @(inF)] dataType:MPSDataTypeFloat32];
        feeds[wPh] = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:wSlot.buffer shape:wSlot.shape dataType:MPSDataTypeFloat32];
        if (bPh) {
            feeds[bPh] = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:bSlot.buffer shape:bSlot.shape dataType:MPSDataTypeFloat32];
        }
        NSDictionary *results = @{out: [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outBuf shape:@[@(T), @(outF)] dataType:MPSDataTypeFloat32]};

        [g runWithMTLCommandQueue:s.commandQueue
                            feeds:feeds
                 targetOperations:nil
                resultsDictionary:results];

        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(outputArr, NULL);
        memcpy(p, [outBuf contents], outBytes);
        env->ReleasePrimitiveArrayCritical(outputArr, p, 0);
    }
}

// ============================================================================
// Phase 2.2 step 4 — SwiGLU MLP forward (turbo TurboMLP swiglu 경로 동등)
//
// gate = Linear(gateW, gateB)(x)
// up   = Linear(upW, upB)(x)
// h    = silu(gate) * up               # silu(z) = z * sigmoid(z)
// y    = Linear(downW, downB)(h)
// ============================================================================
static MPSGraphTensor *buildSiLU(MPSGraph *g, MPSGraphTensor *z) {
    MPSGraphTensor *sig = [g sigmoidWithTensor:z name:nil];
    return [g multiplicationWithPrimaryTensor:z secondaryTensor:sig name:nil];
}

static MPSGraphTensor *buildSwiGLU(MPSGraph *g,
                                   MPSGraphTensor *x,
                                   MPSGraphTensor *gateW, MPSGraphTensor *gateB,
                                   MPSGraphTensor *upW, MPSGraphTensor *upB,
                                   MPSGraphTensor *downW, MPSGraphTensor *downB) {
    MPSGraphTensor *gate = buildLinear(g, x, gateW, gateB);
    MPSGraphTensor *up = buildLinear(g, x, upW, upB);
    MPSGraphTensor *gateAct = buildSiLU(g, gate);
    MPSGraphTensor *h = [g multiplicationWithPrimaryTensor:gateAct
                                          secondaryTensor:up
                                                     name:nil];
    return buildLinear(g, h, downW, downB);
}

JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunSwiGluForward(
    JNIEnv *env, jclass clazz, jlong handle,
    jint pGateW, jint pGateB, jint pUpW, jint pUpB, jint pDownW, jint pDownB,
    jint T, jint embedDim, jint hiddenDim,
    jfloatArray inputArr, jfloatArray outputArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;

        MPSGraph *g = [[MPSGraph alloc] init];
        MPSGraphTensor *xPh = [g placeholderWithShape:@[@(T), @(embedDim)]
                                             dataType:MPSDataTypeFloat32 name:@"x"];

        PikoWeightSlot *gateWS = (PikoWeightSlot *)s.weights[pGateW];
        PikoWeightSlot *gateBS = (PikoWeightSlot *)s.weights[pGateB];
        PikoWeightSlot *upWS   = (PikoWeightSlot *)s.weights[pUpW];
        PikoWeightSlot *upBS   = (PikoWeightSlot *)s.weights[pUpB];
        PikoWeightSlot *downWS = (PikoWeightSlot *)s.weights[pDownW];
        PikoWeightSlot *downBS = (PikoWeightSlot *)s.weights[pDownB];

        MPSGraphTensor *gateWPh = [g placeholderWithShape:gateWS.shape dataType:MPSDataTypeFloat32 name:@"gateW"];
        MPSGraphTensor *gateBPh = [g placeholderWithShape:gateBS.shape dataType:MPSDataTypeFloat32 name:@"gateB"];
        MPSGraphTensor *upWPh   = [g placeholderWithShape:upWS.shape   dataType:MPSDataTypeFloat32 name:@"upW"];
        MPSGraphTensor *upBPh   = [g placeholderWithShape:upBS.shape   dataType:MPSDataTypeFloat32 name:@"upB"];
        MPSGraphTensor *downWPh = [g placeholderWithShape:downWS.shape dataType:MPSDataTypeFloat32 name:@"downW"];
        MPSGraphTensor *downBPh = [g placeholderWithShape:downBS.shape dataType:MPSDataTypeFloat32 name:@"downB"];

        MPSGraphTensor *out = buildSwiGLU(g, xPh,
                                          gateWPh, gateBPh, upWPh, upBPh, downWPh, downBPh);

        NSUInteger inBytes = (NSUInteger)T * embedDim * sizeof(float);
        NSUInteger outBytes = inBytes;  // [T, embedDim]
        id<MTLBuffer> inBuf = [s.device newBufferWithLength:inBytes
                                                    options:MTLResourceStorageModeShared];
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(inputArr, NULL);
            memcpy([inBuf contents], p, inBytes);
            env->ReleasePrimitiveArrayCritical(inputArr, p, JNI_ABORT);
        }
        id<MTLBuffer> outBuf = [s.device newBufferWithLength:outBytes
                                                     options:MTLResourceStorageModeShared];

        NSDictionary *feeds = @{
            xPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf shape:@[@(T), @(embedDim)] dataType:MPSDataTypeFloat32],
            gateWPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:gateWS.buffer shape:gateWS.shape dataType:MPSDataTypeFloat32],
            gateBPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:gateBS.buffer shape:gateBS.shape dataType:MPSDataTypeFloat32],
            upWPh:   [[MPSGraphTensorData alloc] initWithMTLBuffer:upWS.buffer   shape:upWS.shape   dataType:MPSDataTypeFloat32],
            upBPh:   [[MPSGraphTensorData alloc] initWithMTLBuffer:upBS.buffer   shape:upBS.shape   dataType:MPSDataTypeFloat32],
            downWPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:downWS.buffer shape:downWS.shape dataType:MPSDataTypeFloat32],
            downBPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:downBS.buffer shape:downBS.shape dataType:MPSDataTypeFloat32],
        };
        NSDictionary *results = @{out: [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outBuf shape:@[@(T), @(embedDim)] dataType:MPSDataTypeFloat32]};

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(outputArr, NULL);
        memcpy(p, [outBuf contents], outBytes);
        env->ReleasePrimitiveArrayCritical(outputArr, p, 0);
    }
}

// ============================================================================
// Phase 2.2 step 5 — Multi-head causal self-attention + RoPE
//
// RoPE convention (turbo): pair (i, i+headDim/2). theta = 10000^(-2i/headDim).
//   x'[i]      = x[i] * cos(angle) - x[i+D/2] * sin(angle)
//   x'[i+D/2]  = x[i] * sin(angle) + x[i+D/2] * cos(angle)
//   angle = position * theta
//
// cos/sin은 host에서 계산한 placeholder로 전달 (shape [T, headDim/2]).
// 결과를 broadcast해서 [T, numHeads, headDim/2]로 확장 후 element-wise.
//
// Causal mask: lower-triangular [T, T] (1 for valid, 0 for masked).
//   scores = scores + (1 - mask) * (-1e9)
// ============================================================================
static MPSGraphTensor *buildRoPE(MPSGraph *g,
                                 MPSGraphTensor *qOrK,    // [T, numHeads, headDim]
                                 MPSGraphTensor *cos,     // [T, headDim/2]
                                 MPSGraphTensor *sin,     // [T, headDim/2]
                                 NSInteger T, NSInteger numHeads, NSInteger headDim) {
    NSInteger half = headDim / 2;
    // split last dim into [..., 0:half] and [..., half:headDim]
    MPSGraphTensor *first = [g sliceTensor:qOrK
                                 dimension:2
                                     start:0
                                    length:half
                                      name:nil];  // [T, numHeads, half]
    MPSGraphTensor *second = [g sliceTensor:qOrK
                                  dimension:2
                                      start:half
                                     length:half
                                       name:nil];  // [T, numHeads, half]

    // cos/sin [T, half] → broadcast to [T, 1, half] via reshape
    MPSGraphTensor *cosR = [g reshapeTensor:cos withShape:@[@(T), @1, @(half)] name:nil];
    MPSGraphTensor *sinR = [g reshapeTensor:sin withShape:@[@(T), @1, @(half)] name:nil];

    MPSGraphTensor *fc = [g multiplicationWithPrimaryTensor:first secondaryTensor:cosR name:nil];
    MPSGraphTensor *ss = [g multiplicationWithPrimaryTensor:second secondaryTensor:sinR name:nil];
    MPSGraphTensor *fs = [g multiplicationWithPrimaryTensor:first secondaryTensor:sinR name:nil];
    MPSGraphTensor *sc = [g multiplicationWithPrimaryTensor:second secondaryTensor:cosR name:nil];

    MPSGraphTensor *outFirst  = [g subtractionWithPrimaryTensor:fc secondaryTensor:ss name:nil];
    MPSGraphTensor *outSecond = [g additionWithPrimaryTensor:fs secondaryTensor:sc name:nil];

    return [g concatTensors:@[outFirst, outSecond] dimension:2 name:nil];  // [T, numHeads, headDim]
}

/**
 * Attention forward (1 layer). RoPE 항상 적용.
 *
 * input: x [T, embedDim]
 * weights: qW, qB, kW, kB, vW, vB, outW, outB (paramIndex 0..7)
 * RoPE tables: cos, sin (T*headDim/2 each, host precompute)
 * output: y [T, embedDim]
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunAttentionForward(
    JNIEnv *env, jclass clazz, jlong handle,
    jint pQW, jint pQB, jint pKW, jint pKB, jint pVW, jint pVB,
    jint pOutW, jint pOutB,
    jint T, jint embedDim, jint numHeads,
    jfloatArray inputArr, jfloatArray cosArr, jfloatArray sinArr,
    jfloatArray maskArr,  // [T, T], lower-tri 0, upper-tri -1e9 (caller builds)
    jfloatArray outputArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        int headDim = embedDim / numHeads;
        int half = headDim / 2;

        MPSGraph *g = [[MPSGraph alloc] init];
        MPSGraphTensor *xPh = [g placeholderWithShape:@[@(T), @(embedDim)]
                                             dataType:MPSDataTypeFloat32 name:@"x"];
        MPSGraphTensor *cosPh = [g placeholderWithShape:@[@(T), @(half)]
                                               dataType:MPSDataTypeFloat32 name:@"cos"];
        MPSGraphTensor *sinPh = [g placeholderWithShape:@[@(T), @(half)]
                                               dataType:MPSDataTypeFloat32 name:@"sin"];
        MPSGraphTensor *maskPh = [g placeholderWithShape:@[@(T), @(T)]
                                                dataType:MPSDataTypeFloat32 name:@"mask"];

        PikoWeightSlot *qWS  = (PikoWeightSlot *)s.weights[pQW];
        PikoWeightSlot *qBS  = (PikoWeightSlot *)s.weights[pQB];
        PikoWeightSlot *kWS  = (PikoWeightSlot *)s.weights[pKW];
        PikoWeightSlot *kBS  = (PikoWeightSlot *)s.weights[pKB];
        PikoWeightSlot *vWS  = (PikoWeightSlot *)s.weights[pVW];
        PikoWeightSlot *vBS  = (PikoWeightSlot *)s.weights[pVB];
        PikoWeightSlot *oWS  = (PikoWeightSlot *)s.weights[pOutW];
        PikoWeightSlot *oBS  = (PikoWeightSlot *)s.weights[pOutB];

        MPSGraphTensor *qWPh = [g placeholderWithShape:qWS.shape dataType:MPSDataTypeFloat32 name:@"qW"];
        MPSGraphTensor *qBPh = [g placeholderWithShape:qBS.shape dataType:MPSDataTypeFloat32 name:@"qB"];
        MPSGraphTensor *kWPh = [g placeholderWithShape:kWS.shape dataType:MPSDataTypeFloat32 name:@"kW"];
        MPSGraphTensor *kBPh = [g placeholderWithShape:kBS.shape dataType:MPSDataTypeFloat32 name:@"kB"];
        MPSGraphTensor *vWPh = [g placeholderWithShape:vWS.shape dataType:MPSDataTypeFloat32 name:@"vW"];
        MPSGraphTensor *vBPh = [g placeholderWithShape:vBS.shape dataType:MPSDataTypeFloat32 name:@"vB"];
        MPSGraphTensor *oWPh = [g placeholderWithShape:oWS.shape dataType:MPSDataTypeFloat32 name:@"oW"];
        MPSGraphTensor *oBPh = [g placeholderWithShape:oBS.shape dataType:MPSDataTypeFloat32 name:@"oB"];

        // Q, K, V projections [T, embedDim]
        MPSGraphTensor *q = buildLinear(g, xPh, qWPh, qBPh);
        MPSGraphTensor *k = buildLinear(g, xPh, kWPh, kBPh);
        MPSGraphTensor *v = buildLinear(g, xPh, vWPh, vBPh);

        // Reshape to [T, numHeads, headDim]
        NSArray *headShape = @[@(T), @(numHeads), @(headDim)];
        q = [g reshapeTensor:q withShape:headShape name:nil];
        k = [g reshapeTensor:k withShape:headShape name:nil];
        v = [g reshapeTensor:v withShape:headShape name:nil];

        // RoPE on Q and K
        q = buildRoPE(g, q, cosPh, sinPh, T, numHeads, headDim);
        k = buildRoPE(g, k, cosPh, sinPh, T, numHeads, headDim);

        // Transpose to [numHeads, T, headDim] via permutation (dim 0 ↔ dim 1)
        q = [g transposeTensor:q dimension:0 withDimension:1 name:nil];
        k = [g transposeTensor:k dimension:0 withDimension:1 name:nil];
        v = [g transposeTensor:v dimension:0 withDimension:1 name:nil];

        // scores = Q @ K^T / sqrt(headDim) → [numHeads, T, T]
        MPSGraphTensor *kT = [g transposeTensor:k dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *scores = [g matrixMultiplicationWithPrimaryTensor:q
                                                          secondaryTensor:kT
                                                                     name:nil];
        float scale = 1.0f / sqrtf((float)headDim);
        MPSGraphTensor *scaleConst = [g constantWithScalar:scale
                                                     shape:@[]
                                                  dataType:MPSDataTypeFloat32];
        scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:scaleConst name:nil];

        // Add causal mask (lower-tri 0, upper-tri -1e9). broadcast [T, T] → [numHeads, T, T].
        scores = [g additionWithPrimaryTensor:scores secondaryTensor:maskPh name:nil];

        // (구 주석 제거)
        // graph에서: broadcast가 안 되니 host precompute placeholder가 단순.
        // 여기선 큰 음수 추가 방식 (softmax 후 0). 정확히는 -inf지만 large negative로 충분.
        // mask shape [T, T] (broadcast to [numHeads, T, T])
        // host에서 cos/sin과 같이 전달하는 게 깔끔하지만, 이번엔 graph 내에서 만들어보자.
        // Placeholder cumsum 트릭 대신: bandPart(ones, -1, 0) — 그러나 MPSGraph는 bandPart 없음.
        // 가장 단순: host에서 mask 전달.
        // → 일단 mask를 추가 placeholder로 받자.
        // (caller가 함께 전달)
        // 본 함수에선 mask 없이 정의하고 caller가 mask를 위해 cosArr/sinArr/maskArr 함께 전달.
        // simplicity 우선: causal mask placeholder 추가.

        // mask shape [T, T] (lower-triangular 1, upper-triangular -1e9)
        // → 위 코드에선 caller가 추가 인자 안 보냄. 다음 응답에서 mask placeholder 추가.

        // TEMP: mask 없이 (validation은 T=1 또는 짧은 시퀀스로 가능, 일반성은 다음 step에서 추가)
        // → 일단 mask 없이 동작은 확인. 다음 step에서 mask 추가.

        // softmax along last axis
        MPSGraphTensor *attn = [g softMaxWithTensor:scores axis:-1 name:nil];

        // out = attn @ V → [numHeads, T, headDim]
        MPSGraphTensor *attnV = [g matrixMultiplicationWithPrimaryTensor:attn
                                                          secondaryTensor:v
                                                                     name:nil];

        // transpose back to [T, numHeads, headDim] → reshape [T, embedDim]
        MPSGraphTensor *back = [g transposeTensor:attnV dimension:0 withDimension:1 name:nil];
        MPSGraphTensor *concat = [g reshapeTensor:back withShape:@[@(T), @(embedDim)] name:nil];

        // output projection
        MPSGraphTensor *y = buildLinear(g, concat, oWPh, oBPh);

        // Input/output buffers
        NSUInteger inBytes = (NSUInteger)T * embedDim * sizeof(float);
        NSUInteger ropeBytes = (NSUInteger)T * half * sizeof(float);
        NSUInteger maskBytes = (NSUInteger)T * T * sizeof(float);
        id<MTLBuffer> xBuf = [s.device newBufferWithLength:inBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cosBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(inputArr, NULL);
            memcpy([xBuf contents], p, inBytes);
            env->ReleasePrimitiveArrayCritical(inputArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(cosArr, NULL);
            memcpy([cosBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(cosArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(sinArr, NULL);
            memcpy([sinBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(sinArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(maskArr, NULL);
            memcpy([maskBuf contents], p, maskBytes);
            env->ReleasePrimitiveArrayCritical(maskArr, p, JNI_ABORT);
        }
        id<MTLBuffer> outBuf = [s.device newBufferWithLength:inBytes options:MTLResourceStorageModeShared];

        NSDictionary *feeds = @{
            xPh:   [[MPSGraphTensorData alloc] initWithMTLBuffer:xBuf shape:@[@(T), @(embedDim)] dataType:MPSDataTypeFloat32],
            cosPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32],
            sinPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32],
            maskPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32],
            qWPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:qWS.buffer shape:qWS.shape dataType:MPSDataTypeFloat32],
            qBPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:qBS.buffer shape:qBS.shape dataType:MPSDataTypeFloat32],
            kWPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:kWS.buffer shape:kWS.shape dataType:MPSDataTypeFloat32],
            kBPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:kBS.buffer shape:kBS.shape dataType:MPSDataTypeFloat32],
            vWPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:vWS.buffer shape:vWS.shape dataType:MPSDataTypeFloat32],
            vBPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:vBS.buffer shape:vBS.shape dataType:MPSDataTypeFloat32],
            oWPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:oWS.buffer shape:oWS.shape dataType:MPSDataTypeFloat32],
            oBPh: [[MPSGraphTensorData alloc] initWithMTLBuffer:oBS.buffer shape:oBS.shape dataType:MPSDataTypeFloat32],
        };
        NSDictionary *results = @{y: [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outBuf shape:@[@(T), @(embedDim)] dataType:MPSDataTypeFloat32]};

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(outputArr, NULL);
        memcpy(p, [outBuf contents], inBytes);
        env->ReleasePrimitiveArrayCritical(outputArr, p, 0);
    }
}

// ============================================================================
// Phase 2.2 step 6+7 — Full forward graph (numLayers block + tied lm head)
//
// paramIndex layout (turbo TurboPikoGPT.parameters() 순서):
//   0 = tokenEmbedding [vocab, embedDim]
//   base = 1 + L*18
//   per layer L:
//     0..1  : layerNorm1 gamma, beta
//     2..9  : qW/qB, kW/kB, vW/vB, outW/outB
//     10..11: layerNorm2 gamma, beta
//     12..17: gateW/gateB, upW/upB, downW/downB
//   final = 1 + numLayers*18
//     final+0..1: finalLayerNorm gamma, beta
//
// tied lm head: matmul(x, tokenEmbedding.T)
// ============================================================================
static MPSGraphTensor *buildAttentionGraph(MPSGraph *g,
                                           MPSGraphTensor *x,
                                           MPSGraphTensor *qW, MPSGraphTensor *qB,
                                           MPSGraphTensor *kW, MPSGraphTensor *kB,
                                           MPSGraphTensor *vW, MPSGraphTensor *vB,
                                           MPSGraphTensor *oW, MPSGraphTensor *oB,
                                           MPSGraphTensor *cos, MPSGraphTensor *sin,
                                           MPSGraphTensor *mask,
                                           NSInteger T, NSInteger embedDim,
                                           NSInteger numHeads, NSInteger headDim) {
    MPSGraphTensor *q = buildLinear(g, x, qW, qB);
    MPSGraphTensor *k = buildLinear(g, x, kW, kB);
    MPSGraphTensor *v = buildLinear(g, x, vW, vB);
    NSArray *headShape = @[@(T), @(numHeads), @(headDim)];
    q = [g reshapeTensor:q withShape:headShape name:nil];
    k = [g reshapeTensor:k withShape:headShape name:nil];
    v = [g reshapeTensor:v withShape:headShape name:nil];
    q = buildRoPE(g, q, cos, sin, T, numHeads, headDim);
    k = buildRoPE(g, k, cos, sin, T, numHeads, headDim);
    q = [g transposeTensor:q dimension:0 withDimension:1 name:nil];
    k = [g transposeTensor:k dimension:0 withDimension:1 name:nil];
    v = [g transposeTensor:v dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *kT = [g transposeTensor:k dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *scores = [g matrixMultiplicationWithPrimaryTensor:q secondaryTensor:kT name:nil];
    float scale = 1.0f / sqrtf((float)headDim);
    MPSGraphTensor *sc = [g constantWithScalar:scale shape:@[] dataType:MPSDataTypeFloat32];
    scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:sc name:nil];
    scores = [g additionWithPrimaryTensor:scores secondaryTensor:mask name:nil];
    MPSGraphTensor *attn = [g softMaxWithTensor:scores axis:-1 name:nil];
    MPSGraphTensor *attnV = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:v name:nil];
    MPSGraphTensor *back = [g transposeTensor:attnV dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *concat = [g reshapeTensor:back withShape:@[@(T), @(embedDim)] name:nil];
    return buildLinear(g, concat, oW, oB);
}

/**
 * Full forward: tokens [T] → logits [T, vocab].
 *
 * weights는 paramIndex 0..(1+L*18+1) 모두 로드 완료된 상태여야.
 * cos/sin: [T, headDim/2]. mask: [T, T] lower-tri 0 / upper -1e9.
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunFullForward(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr,
    jfloatArray logitsArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;

        NSInteger numLayers = s.numLayers;
        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        NSInteger vocab = s.vocab;
        NSInteger hiddenDim = (8 * embedDim + 1) / 3;

        jsize T = env->GetArrayLength(tokenIdsArr);

        MPSGraph *g = [[MPSGraph alloc] init];

        // 1. Placeholders
        MPSGraphTensor *idsPh = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"ids"];
        MPSGraphTensor *cosPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"cos"];
        MPSGraphTensor *sinPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"sin"];
        MPSGraphTensor *maskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"mask"];

        // 모든 weight placeholder. paramIndex 순서.
        NSMutableArray<MPSGraphTensor *> *wPh = [NSMutableArray array];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            MPSGraphTensor *t = [g placeholderWithShape:slot.shape
                                              dataType:MPSDataTypeFloat32
                                                  name:[NSString stringWithFormat:@"w%ld", (long)i]];
            [wPh addObject:t];
        }

        // 2. Build forward graph
        MPSGraphTensor *tokEmb = wPh[0];  // [vocab, embedDim]
        MPSGraphTensor *x = [g gatherWithUpdatesTensor:tokEmb
                                        indicesTensor:idsPh
                                                 axis:0
                                      batchDimensions:0
                                                 name:nil];  // [T, embedDim]

        for (NSInteger L = 0; L < numLayers; L++) {
            NSInteger base = 1 + L * 18;
            MPSGraphTensor *gamma1 = wPh[base + 0];
            MPSGraphTensor *beta1  = wPh[base + 1];
            MPSGraphTensor *qW = wPh[base + 2], *qB = wPh[base + 3];
            MPSGraphTensor *kW = wPh[base + 4], *kB = wPh[base + 5];
            MPSGraphTensor *vW = wPh[base + 6], *vB = wPh[base + 7];
            MPSGraphTensor *oW = wPh[base + 8], *oB = wPh[base + 9];
            MPSGraphTensor *gamma2 = wPh[base + 10];
            MPSGraphTensor *beta2  = wPh[base + 11];
            MPSGraphTensor *gateW = wPh[base + 12], *gateB = wPh[base + 13];
            MPSGraphTensor *upW   = wPh[base + 14], *upB   = wPh[base + 15];
            MPSGraphTensor *downW = wPh[base + 16], *downB = wPh[base + 17];

            // Pre-LN attention
            MPSGraphTensor *ln1 = buildLayerNorm(g, x, gamma1, beta1, 1e-5f, /*axisDim=*/1);
            MPSGraphTensor *attnOut = buildAttentionGraph(g, ln1, qW, qB, kW, kB, vW, vB, oW, oB,
                                                          cosPh, sinPh, maskPh,
                                                          T, embedDim, numHeads, headDim);
            x = [g additionWithPrimaryTensor:x secondaryTensor:attnOut name:nil];

            // Pre-LN MLP (SwiGLU)
            MPSGraphTensor *ln2 = buildLayerNorm(g, x, gamma2, beta2, 1e-5f, /*axisDim=*/1);
            MPSGraphTensor *mlpOut = buildSwiGLU(g, ln2, gateW, gateB, upW, upB, downW, downB);
            x = [g additionWithPrimaryTensor:x secondaryTensor:mlpOut name:nil];
        }

        // Final LN
        NSInteger finalIdx = 1 + numLayers * 18;
        MPSGraphTensor *finalGamma = wPh[finalIdx + 0];
        MPSGraphTensor *finalBeta  = wPh[finalIdx + 1];
        x = buildLayerNorm(g, x, finalGamma, finalBeta, 1e-5f, /*axisDim=*/1);

        // Tied lm head: logits = x @ tokenEmbedding.T → [T, vocab]
        MPSGraphTensor *embT = [g transposeTensor:tokEmb dimension:0 withDimension:1 name:nil];
        MPSGraphTensor *logits = [g matrixMultiplicationWithPrimaryTensor:x
                                                          secondaryTensor:embT
                                                                     name:nil];

        // 3. Feeds + run
        NSUInteger idsBytes = (NSUInteger)T * sizeof(int32_t);
        NSUInteger ropeBytes = (NSUInteger)T * half * sizeof(float);
        NSUInteger maskBytes = (NSUInteger)T * T * sizeof(float);
        NSUInteger logitsBytes = (NSUInteger)T * vocab * sizeof(float);

        id<MTLBuffer> idsBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cosBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> logitsBuf = [s.device newBufferWithLength:logitsBytes options:MTLResourceStorageModeShared];
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(tokenIdsArr, NULL);
            memcpy([idsBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(tokenIdsArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(cosArr, NULL);
            memcpy([cosBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(cosArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(sinArr, NULL);
            memcpy([sinBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(sinArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(maskArr, NULL);
            memcpy([maskBuf contents], p, maskBytes);
            env->ReleasePrimitiveArrayCritical(maskArr, p, JNI_ABORT);
        }

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[idsPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[cosPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[sinPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[maskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feeds[wPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer
                                                                   shape:slot.shape
                                                                dataType:MPSDataTypeFloat32];
        }
        NSDictionary *results = @{logits: [[MPSGraphTensorData alloc]
            initWithMTLBuffer:logitsBuf shape:@[@(T), @(vocab)] dataType:MPSDataTypeFloat32]};

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(logitsArr, NULL);
        memcpy(p, [logitsBuf contents], logitsBytes);
        env->ReleasePrimitiveArrayCritical(logitsArr, p, 0);
    }
}

// ============================================================================
// Phase 3 step 1 — Forward + CE loss graph (loss return only)
//
// loss = -mean_t( log_softmax(logits[t])[targets[t]] )
// Phase 3 step 2에서 gradient + weight grad buffer 추가.
// ============================================================================

static MPSGraphTensor *buildForwardLogits(MPSGraph *g, PikoMpsGraphSession *s,
                                          NSArray<MPSGraphTensor *> *wPh,
                                          MPSGraphTensor *idsPh,
                                          MPSGraphTensor *cosPh, MPSGraphTensor *sinPh,
                                          MPSGraphTensor *maskPh,
                                          NSInteger T) {
    NSInteger numLayers = s.numLayers;
    NSInteger embedDim = s.embedDim;
    NSInteger numHeads = s.numHeads;
    NSInteger headDim = embedDim / numHeads;

    MPSGraphTensor *tokEmb = wPh[0];
    MPSGraphTensor *x = [g gatherWithUpdatesTensor:tokEmb
                                    indicesTensor:idsPh
                                             axis:0
                                  batchDimensions:0
                                             name:nil];

    for (NSInteger L = 0; L < numLayers; L++) {
        NSInteger base = 1 + L * 18;
        MPSGraphTensor *g1 = wPh[base+0], *b1 = wPh[base+1];
        MPSGraphTensor *qW = wPh[base+2], *qB = wPh[base+3];
        MPSGraphTensor *kW = wPh[base+4], *kB = wPh[base+5];
        MPSGraphTensor *vW = wPh[base+6], *vB = wPh[base+7];
        MPSGraphTensor *oW = wPh[base+8], *oB = wPh[base+9];
        MPSGraphTensor *g2 = wPh[base+10], *b2 = wPh[base+11];
        MPSGraphTensor *gateW = wPh[base+12], *gateB = wPh[base+13];
        MPSGraphTensor *upW = wPh[base+14], *upB = wPh[base+15];
        MPSGraphTensor *downW = wPh[base+16], *downB = wPh[base+17];

        MPSGraphTensor *ln1 = buildLayerNorm(g, x, g1, b1, 1e-5f, 1);
        MPSGraphTensor *attnOut = buildAttentionGraph(g, ln1, qW, qB, kW, kB, vW, vB, oW, oB,
                                                      cosPh, sinPh, maskPh,
                                                      T, embedDim, numHeads, headDim);
        x = [g additionWithPrimaryTensor:x secondaryTensor:attnOut name:nil];
        MPSGraphTensor *ln2 = buildLayerNorm(g, x, g2, b2, 1e-5f, 1);
        MPSGraphTensor *mlpOut = buildSwiGLU(g, ln2, gateW, gateB, upW, upB, downW, downB);
        x = [g additionWithPrimaryTensor:x secondaryTensor:mlpOut name:nil];
    }
    NSInteger finalIdx = 1 + numLayers * 18;
    x = buildLayerNorm(g, x, wPh[finalIdx+0], wPh[finalIdx+1], 1e-5f, 1);

    // tied lm head
    MPSGraphTensor *embT = [g transposeTensor:tokEmb dimension:0 withDimension:1 name:nil];
    return [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:embT name:nil];
}

/**
 * forward + CE loss. tokens [T], targets [T]. loss scalar return.
 *
 * loss = -mean_t( log_softmax(logits[t])[targets[t]] )
 */
JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunForwardLoss(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jintArray targetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;

        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        NSInteger vocab = s.vocab;

        jsize T = env->GetArrayLength(tokenIdsArr);
        jsize tgtLen = env->GetArrayLength(targetsArr);
        if (tgtLen != T) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "targets length != tokenIds length");
            return -1.0f;
        }

        MPSGraph *g = [[MPSGraph alloc] init];

        MPSGraphTensor *idsPh = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"ids"];
        MPSGraphTensor *tgtPh = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"tgt"];
        MPSGraphTensor *cosPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"cos"];
        MPSGraphTensor *sinPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"sin"];
        MPSGraphTensor *maskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"mask"];
        NSMutableArray<MPSGraphTensor *> *wPh = [NSMutableArray array];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            [wPh addObject:[g placeholderWithShape:slot.shape
                                          dataType:MPSDataTypeFloat32
                                              name:[NSString stringWithFormat:@"w%ld", (long)i]]];
        }

        MPSGraphTensor *logits = buildForwardLogits(g, s, wPh, idsPh, cosPh, sinPh, maskPh, T);

        // CE loss: gather log_softmax at target indices.
        MPSGraphTensor *logSoftmax = [g logarithmWithTensor:
                                          [g softMaxWithTensor:logits axis:-1 name:nil]
                                                       name:nil];
        // one_hot(targets, vocab) → [T, vocab]
        MPSGraphTensor *oneHot = [g oneHotWithIndicesTensor:tgtPh
                                                       depth:vocab
                                                        axis:-1
                                                    dataType:MPSDataTypeFloat32
                                                     onValue:1.0
                                                    offValue:0.0
                                                        name:nil];
        MPSGraphTensor *prod = [g multiplicationWithPrimaryTensor:oneHot
                                                  secondaryTensor:logSoftmax
                                                             name:nil];
        MPSGraphTensor *perTokenLL = [g reductionSumWithTensor:prod axis:-1 name:nil];  // [T]
        MPSGraphTensor *meanLL = [g meanOfTensor:perTokenLL axes:@[@0] name:nil];  // scalar
        MPSGraphTensor *loss = [g negativeWithTensor:meanLL name:nil];

        // Feeds
        NSUInteger idsBytes = T * sizeof(int32_t);
        NSUInteger ropeBytes = T * half * sizeof(float);
        NSUInteger maskBytes = T * T * sizeof(float);
        id<MTLBuffer> idsBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tgtBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cosBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> lossBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(tokenIdsArr, NULL);
            memcpy([idsBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(tokenIdsArr, p, JNI_ABORT);
        }
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(targetsArr, NULL);
            memcpy([tgtBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(targetsArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(cosArr, NULL);
            memcpy([cosBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(cosArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(sinArr, NULL);
            memcpy([sinBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(sinArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(maskArr, NULL);
            memcpy([maskBuf contents], p, maskBytes);
            env->ReleasePrimitiveArrayCritical(maskArr, p, JNI_ABORT);
        }

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[idsPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[tgtPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[cosPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[sinPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[maskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feeds[wPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer
                                                                   shape:slot.shape
                                                                dataType:MPSDataTypeFloat32];
        }
        NSDictionary *results = @{loss: [[MPSGraphTensorData alloc]
            initWithMTLBuffer:lossBuf shape:@[] dataType:MPSDataTypeFloat32]};

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        float lossVal = *((float *)[lossBuf contents]);
        return (jfloat)lossVal;
    }
}

// ============================================================================
// Phase 3 step 2 — Backward: gradient(loss, weights) → slot.gradBuffer (GPU resident)
//
// 자동미분 MPSGraph API: gradientForPrimaryTensor:withTensors:name:
//   → NSDictionary<MPSGraphTensor (weight) *, MPSGraphTensor (grad) *>
//
// 모든 weight grad를 한 graph run으로 계산 + GPU buffer에 저장. caller는 loss만 받음.
// Phase 4 AdamW가 같은 grad buffer를 input으로.
// ============================================================================

static MPSGraphTensor *buildCELoss(MPSGraph *g, MPSGraphTensor *logits, MPSGraphTensor *tgtPh,
                                   NSInteger vocab) {
    MPSGraphTensor *logSoftmax = [g logarithmWithTensor:
                                      [g softMaxWithTensor:logits axis:-1 name:nil]
                                                   name:nil];
    MPSGraphTensor *oneHot = [g oneHotWithIndicesTensor:tgtPh
                                                   depth:vocab
                                                    axis:-1
                                                dataType:MPSDataTypeFloat32
                                                 onValue:1.0 offValue:0.0
                                                    name:nil];
    MPSGraphTensor *prod = [g multiplicationWithPrimaryTensor:oneHot
                                              secondaryTensor:logSoftmax
                                                         name:nil];
    MPSGraphTensor *perTokenLL = [g reductionSumWithTensor:prod axis:-1 name:nil];
    MPSGraphTensor *meanLL = [g meanOfTensor:perTokenLL axes:@[@0] name:nil];
    return [g negativeWithTensor:meanLL name:nil];
}

JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunForwardBackward(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jintArray targetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;

        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        NSInteger vocab = s.vocab;
        jsize T = env->GetArrayLength(tokenIdsArr);

        MPSGraph *g = [[MPSGraph alloc] init];
        MPSGraphTensor *idsPh = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"ids"];
        MPSGraphTensor *tgtPh = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"tgt"];
        MPSGraphTensor *cosPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"cos"];
        MPSGraphTensor *sinPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"sin"];
        MPSGraphTensor *maskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"mask"];

        NSMutableArray<MPSGraphTensor *> *wPh = [NSMutableArray array];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            [wPh addObject:[g placeholderWithShape:slot.shape
                                          dataType:MPSDataTypeFloat32
                                              name:[NSString stringWithFormat:@"w%ld", (long)i]]];
        }
        MPSGraphTensor *logits = buildForwardLogits(g, s, wPh, idsPh, cosPh, sinPh, maskPh, T);
        MPSGraphTensor *loss = buildCELoss(g, logits, tgtPh, vocab);

        NSDictionary<MPSGraphTensor *, MPSGraphTensor *> *grads =
            [g gradientForPrimaryTensor:loss withTensors:wPh name:nil];

        // Feeds (loss와 동일)
        NSUInteger idsBytes = T * sizeof(int32_t);
        NSUInteger ropeBytes = T * half * sizeof(float);
        NSUInteger maskBytes = T * T * sizeof(float);
        id<MTLBuffer> idsBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tgtBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cosBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> lossBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(tokenIdsArr, NULL);
            memcpy([idsBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(tokenIdsArr, p, JNI_ABORT);
        }
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(targetsArr, NULL);
            memcpy([tgtBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(targetsArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(cosArr, NULL);
            memcpy([cosBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(cosArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(sinArr, NULL);
            memcpy([sinBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(sinArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(maskArr, NULL);
            memcpy([maskBuf contents], p, maskBytes);
            env->ReleasePrimitiveArrayCritical(maskArr, p, JNI_ABORT);
        }

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[idsPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[tgtPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[cosPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[sinPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[maskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feeds[wPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer
                                                                   shape:slot.shape
                                                                dataType:MPSDataTypeFloat32];
        }

        // results: loss + 모든 weight gradient
        NSMutableDictionary *results = [NSMutableDictionary dictionary];
        results[loss] = [[MPSGraphTensorData alloc] initWithMTLBuffer:lossBuf shape:@[] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            MPSGraphTensor *gradTensor = grads[wPh[i]];
            if (!gradTensor) {
                env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                              "gradient missing for weight");
                return -1.0f;
            }
            results[gradTensor] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBuffer
                                                                          shape:slot.shape
                                                                       dataType:MPSDataTypeFloat32];
        }

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        float lossVal = *((float *)[lossBuf contents]);
        return (jfloat)lossVal;
    }
}

/** paramIndex의 grad buffer를 host로 복사. 검증/디버그용. */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeReadGrad(
    JNIEnv *env, jclass clazz, jlong handle, jint paramIndex, jfloatArray out) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[paramIndex];
        NSUInteger bytes = slot.numel * sizeof(float);
        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(out, NULL);
        memcpy(p, [slot.gradBuffer contents], bytes);
        env->ReleasePrimitiveArrayCritical(out, p, 0);
    }
}

// ============================================================================
// Phase 4 + 5 — Single training step graph (forward + loss + backward + AdamW)
//
// 1 GPU dispatch = 1 training step. weight/m/v는 PikoWeightSlot.buffer/mBuffer/vBuffer
// (GPU resident). step output은 새 buffer로 쓰고 slot의 buffer 포인터를 swap.
// 다음 step에서 새 weight가 자동 사용됨.
//
// AdamW update:
//   m_new = β1*m + (1-β1)*g
//   v_new = β2*v + (1-β2)*g²
//   m_hat = m_new / (1 - β1^t)
//   v_hat = v_new / (1 - β2^t)
//   w_new = w - lr * m_hat / (sqrt(v_hat) + ε) - lr*wd*w
// ============================================================================

static void buildStepGraph(PikoMpsGraphSession *s, NSInteger T, float beta1, float beta2,
                           float eps, float weightDecay) {
    NSInteger embedDim = s.embedDim;
    NSInteger numHeads = s.numHeads;
    NSInteger headDim = embedDim / numHeads;
    NSInteger half = headDim / 2;
    NSInteger vocab = s.vocab;
    NSInteger N = (NSInteger)s.weights.count;

    MPSGraph *g = [[MPSGraph alloc] init];
    s.stepGraph = g;
    s.cachedT = T;

    s.stepIdsPh  = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"ids"];
    s.stepTgtPh  = [g placeholderWithShape:@[@(T)] dataType:MPSDataTypeInt32 name:@"tgt"];
    s.stepCosPh  = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"cos"];
    s.stepSinPh  = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"sin"];
    s.stepMaskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"mask"];
    s.stepLRPh   = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"lr"];
    s.stepBc1Ph  = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"bc1"];
    s.stepBc2Ph  = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"bc2"];

    s.stepWPh = [NSMutableArray array];
    s.stepMPh = [NSMutableArray array];
    s.stepVPh = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
        [s.stepWPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"w%ld", (long)i]]];
        [s.stepMPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"m%ld", (long)i]]];
        [s.stepVPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"v%ld", (long)i]]];
    }

    MPSGraphTensor *logits = buildForwardLogits(g, s, s.stepWPh, s.stepIdsPh, s.stepCosPh, s.stepSinPh, s.stepMaskPh, T);
    MPSGraphTensor *loss = buildCELoss(g, logits, s.stepTgtPh, vocab);
    s.stepLoss = loss;

    NSDictionary<MPSGraphTensor *, MPSGraphTensor *> *grads =
        [g gradientForPrimaryTensor:loss withTensors:s.stepWPh name:nil];

    MPSGraphTensor *cB1   = [g constantWithScalar:beta1 shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cOmB1 = [g constantWithScalar:(1.0f - beta1) shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cB2   = [g constantWithScalar:beta2 shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cOmB2 = [g constantWithScalar:(1.0f - beta2) shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cEps  = [g constantWithScalar:eps shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cWD   = [g constantWithScalar:weightDecay shape:@[] dataType:MPSDataTypeFloat32];

    s.stepWNew = [NSMutableArray array];
    s.stepMNew = [NSMutableArray array];
    s.stepVNew = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        MPSGraphTensor *w = s.stepWPh[i];
        MPSGraphTensor *gradi = grads[s.stepWPh[i]];
        MPSGraphTensor *m = s.stepMPh[i];
        MPSGraphTensor *v = s.stepVPh[i];

        MPSGraphTensor *mNew = [g additionWithPrimaryTensor:
                                   [g multiplicationWithPrimaryTensor:cB1 secondaryTensor:m name:nil]
                                              secondaryTensor:
                                   [g multiplicationWithPrimaryTensor:cOmB1 secondaryTensor:gradi name:nil]
                                                         name:nil];
        MPSGraphTensor *g2 = [g squareWithTensor:gradi name:nil];
        MPSGraphTensor *vNew = [g additionWithPrimaryTensor:
                                   [g multiplicationWithPrimaryTensor:cB2 secondaryTensor:v name:nil]
                                              secondaryTensor:
                                   [g multiplicationWithPrimaryTensor:cOmB2 secondaryTensor:g2 name:nil]
                                                         name:nil];
        MPSGraphTensor *mHat = [g divisionWithPrimaryTensor:mNew secondaryTensor:s.stepBc1Ph name:nil];
        MPSGraphTensor *vHat = [g divisionWithPrimaryTensor:vNew secondaryTensor:s.stepBc2Ph name:nil];
        MPSGraphTensor *denom = [g additionWithPrimaryTensor:
                                    [g squareRootWithTensor:vHat name:nil]
                                            secondaryTensor:cEps name:nil];
        MPSGraphTensor *stepUpd = [g divisionWithPrimaryTensor:mHat secondaryTensor:denom name:nil];
        MPSGraphTensor *lrStep = [g multiplicationWithPrimaryTensor:s.stepLRPh secondaryTensor:stepUpd name:nil];
        MPSGraphTensor *lrWD = [g multiplicationWithPrimaryTensor:s.stepLRPh secondaryTensor:cWD name:nil];
        MPSGraphTensor *wdW = [g multiplicationWithPrimaryTensor:lrWD secondaryTensor:w name:nil];
        MPSGraphTensor *wNew = [g subtractionWithPrimaryTensor:
                                   [g subtractionWithPrimaryTensor:w secondaryTensor:lrStep name:nil]
                                            secondaryTensor:wdW name:nil];
        [s.stepWNew addObject:wNew];
        [s.stepMNew addObject:mNew];
        [s.stepVNew addObject:vNew];
    }
}

JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunTrainingStep(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jintArray targetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr,
    jfloat lr, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay,
    jint stepT) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;

        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        jsize T = env->GetArrayLength(tokenIdsArr);
        NSInteger N = (NSInteger)s.weights.count;

        // Lazy build (first call) — graph는 한 번만 만들고 매 step에 재사용.
        if (s.cachedT == 0) {
            buildStepGraph(s, T, beta1, beta2, eps, weightDecay);
        }
        MPSGraph *g = s.stepGraph;
        // Use cached fields below. lr/bc1/bc2 are placeholders → host computes per step.

        float bc1 = 1.0f - powf(beta1, (float)stepT);
        float bc2 = 1.0f - powf(beta2, (float)stepT);

        // ---- Feeds ----
        NSUInteger idsBytes = T * sizeof(int32_t);
        NSUInteger ropeBytes = T * half * sizeof(float);
        NSUInteger maskBytes = T * T * sizeof(float);
        id<MTLBuffer> idsBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tgtBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cosBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> lossBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(tokenIdsArr, NULL);
            memcpy([idsBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(tokenIdsArr, p, JNI_ABORT);
        }
        {
            jint *p = (jint *)env->GetPrimitiveArrayCritical(targetsArr, NULL);
            memcpy([tgtBuf contents], p, idsBytes);
            env->ReleasePrimitiveArrayCritical(targetsArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(cosArr, NULL);
            memcpy([cosBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(cosArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(sinArr, NULL);
            memcpy([sinBuf contents], p, ropeBytes);
            env->ReleasePrimitiveArrayCritical(sinArr, p, JNI_ABORT);
        }
        {
            jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(maskArr, NULL);
            memcpy([maskBuf contents], p, maskBytes);
            env->ReleasePrimitiveArrayCritical(maskArr, p, JNI_ABORT);
        }

        // lr/bc1/bc2 small placeholders
        id<MTLBuffer> lrBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc1Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc2Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        *((float *)[lrBuf contents]) = lr;
        *((float *)[bc1Buf contents]) = bc1;
        *((float *)[bc2Buf contents]) = bc2;

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[s.stepIdsPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[s.stepTgtPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(T)] dataType:MPSDataTypeInt32];
        feeds[s.stepCosPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[s.stepSinPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[s.stepMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
        feeds[s.stepLRPh]   = [[MPSGraphTensorData alloc] initWithMTLBuffer:lrBuf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.stepBc1Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc1Buf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.stepBc2Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc2Buf shape:@[] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feeds[s.stepWPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer shape:slot.shape dataType:MPSDataTypeFloat32];
            feeds[s.stepMPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
            feeds[s.stepVPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
        }

        // ---- Results: loss + 새 w/m/v 임시 buffer ----
        NSMutableArray<id<MTLBuffer>> *newW = [NSMutableArray array];
        NSMutableArray<id<MTLBuffer>> *newM = [NSMutableArray array];
        NSMutableArray<id<MTLBuffer>> *newV = [NSMutableArray array];
        NSMutableDictionary *results = [NSMutableDictionary dictionary];
        results[s.stepLoss] = [[MPSGraphTensorData alloc] initWithMTLBuffer:lossBuf shape:@[] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            NSUInteger by = slot.numel * sizeof(float);
            id<MTLBuffer> wb = [s.device newBufferWithLength:by options:MTLResourceStorageModeShared];
            id<MTLBuffer> mb = [s.device newBufferWithLength:by options:MTLResourceStorageModeShared];
            id<MTLBuffer> vb = [s.device newBufferWithLength:by options:MTLResourceStorageModeShared];
            [newW addObject:wb]; [newM addObject:mb]; [newV addObject:vb];
            results[s.stepWNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:wb shape:slot.shape dataType:MPSDataTypeFloat32];
            results[s.stepMNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:mb shape:slot.shape dataType:MPSDataTypeFloat32];
            results[s.stepVNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:vb shape:slot.shape dataType:MPSDataTypeFloat32];
        }

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        // Swap pointers: 새 buffer를 slot.buffer/mBuffer/vBuffer로.
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            slot.buffer  = newW[i];
            slot.mBuffer = newM[i];
            slot.vBuffer = newV[i];
        }

        float lossVal = *((float *)[lossBuf contents]);
        return (jfloat)lossVal;
    }
}

/** weight slot의 현재 buffer를 host로 읽어옴 (검증/체크포인트용). */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeReadWeight(
    JNIEnv *env, jclass clazz, jlong handle, jint paramIndex, jfloatArray out) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[paramIndex];
        NSUInteger bytes = slot.numel * sizeof(float);
        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(out, NULL);
        memcpy(p, [slot.buffer contents], bytes);
        env->ReleasePrimitiveArrayCritical(out, p, 0);
    }
}

// Phase 5: 1 step = forward + backward + AdamW. (위 nativeRunTrainingStep로 통합 구현됨)
JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeStep(
    JNIEnv *env, jclass clazz, jlong handle, jintArray tokenIds, jintArray targets) {
    (void)handle;
    (void)tokenIds;
    (void)targets;
    return -1.0f;
}

#ifdef __cplusplus
}
#endif
