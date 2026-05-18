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
// P1.3 — Variable mode: stepGraph 안에서 weight/m/v를 graph variable로 관리. 외부 buffer와 분리.
@property (nonatomic, strong) MPSGraphTensor *stepWVar;
@property (nonatomic, strong) MPSGraphTensor *stepMVar;
@property (nonatomic, strong) MPSGraphTensor *stepVVar;
// P1.3 — ping-pong alt buffers. 매 step result에 재사용 후 swap. variable 패러다임의 단순 형태.
@property (nonatomic, strong) id<MTLBuffer> bufferAlt;
@property (nonatomic, strong) id<MTLBuffer> mBufferAlt;
@property (nonatomic, strong) id<MTLBuffer> vBufferAlt;
// P1.2 진정한 grad accumulation — 별도 accum buffer + alt (ping-pong)
@property (nonatomic, strong) id<MTLBuffer> gradBufferAlt;
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
@property (nonatomic, assign) NSInteger cachedB;  // P1.1 — batch dimension cache key
@property (nonatomic, strong) MPSGraphTensor *stepIdsPh;
@property (nonatomic, strong) MPSGraphTensor *stepTgtPh;
@property (nonatomic, strong) MPSGraphTensor *stepCosPh;
@property (nonatomic, strong) MPSGraphTensor *stepSinPh;
@property (nonatomic, strong) MPSGraphTensor *stepMaskPh;
@property (nonatomic, strong) MPSGraphTensor *stepLRPh;
@property (nonatomic, strong) MPSGraphTensor *stepBc1Ph;
@property (nonatomic, strong) MPSGraphTensor *stepBc2Ph;
@property (nonatomic, strong) MPSGraphTensor *stepClipPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepWPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepMPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepVPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepWNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepMNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *stepVNew;
@property (nonatomic, strong) NSMutableArray *stepAssignOps;  // P1.3 — variable mode: assign operations
@property (nonatomic, strong) MPSGraphTensor *stepLoss;
// P1.3 — variable mode read/write helper graphs. lazy build.
@property (nonatomic, strong) MPSGraph *readVarGraph;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *readVarWOut;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *readVarMOut;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *readVarVOut;

// P1.2 진정한 grad accumulation — accumGraph (forward+backward+grad accumulate)
@property (nonatomic, strong) MPSGraph *accumGraph;
@property (nonatomic, assign) NSInteger cachedAccumT;
@property (nonatomic, assign) NSInteger cachedAccumB;
@property (nonatomic, strong) MPSGraphTensor *accumIdsPh;
@property (nonatomic, strong) MPSGraphTensor *accumTgtPh;
@property (nonatomic, strong) MPSGraphTensor *accumCosPh;
@property (nonatomic, strong) MPSGraphTensor *accumSinPh;
@property (nonatomic, strong) MPSGraphTensor *accumMaskPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *accumWPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *accumGradOldPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *accumGradNew;
@property (nonatomic, strong) MPSGraphTensor *accumLoss;

// P1.2 진정한 grad accumulation — adamGraph (AdamW only, accumulated grad 입력)
@property (nonatomic, strong) MPSGraph *adamGraph;
@property (nonatomic, assign) BOOL adamBuilt;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamWPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamMPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamVPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamGradPh;
@property (nonatomic, strong) MPSGraphTensor *adamLRPh;
@property (nonatomic, strong) MPSGraphTensor *adamBc1Ph;
@property (nonatomic, strong) MPSGraphTensor *adamBc2Ph;
@property (nonatomic, strong) MPSGraphTensor *adamClipPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamWNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamMNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamVNew;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *adamGradReset;

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
@property (nonatomic, assign) BOOL useVariableForStep;  // P1.3
@property (nonatomic, assign) BOOL useFp16;             // P2.1 mixed precision forward
@property (nonatomic, assign) BOOL useDropout;          // P4 — turbo 동등
@property (nonatomic, assign) float dropoutProbability; // 0..1, useDropout=true 시점에만 의미
// PyTorch 표준 일치 — buildAccumGraph에서 loss를 /accumSteps로 나눠서 grad가 micro-mean이 되도록.
@property (nonatomic, assign) NSInteger accumSteps;
// P4 — 학습 graph용 dropout mask placeholder. [2*numLayers, B, T, embedDim].
@property (nonatomic, strong) MPSGraphTensor *stepDropoutMaskPh;
@property (nonatomic, strong) MPSGraphTensor *accumDropoutMaskPh;
// Phase 2 — forward-only (eval) graph cache. 매 호출 새 MPSGraph alloc 회피.
@property (nonatomic, strong) MPSGraph *forwardGraph;
@property (nonatomic, assign) NSInteger cachedForwardT;
@property (nonatomic, assign) NSInteger cachedForwardB;
@property (nonatomic, strong) MPSGraphTensor *forwardIdsPh;
@property (nonatomic, strong) MPSGraphTensor *forwardTgtPh;
@property (nonatomic, strong) MPSGraphTensor *forwardCosPh;
@property (nonatomic, strong) MPSGraphTensor *forwardSinPh;
@property (nonatomic, strong) MPSGraphTensor *forwardMaskPh;
@property (nonatomic, strong) NSMutableArray<MPSGraphTensor *> *forwardWPh;
@property (nonatomic, strong) MPSGraphTensor *forwardLoss;
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

// Forward declarations — helper functions used across JNI exports (file 순서 무관 호출 가능).
// dropoutMaskPh: shape [2*numLayers, B, T, embedDim] (useDropout=true 시점) 또는 nil.
static MPSGraphTensor *buildForwardLogits(MPSGraph *g, PikoMpsGraphSession *s,
                                          NSArray<MPSGraphTensor *> *wPh,
                                          MPSGraphTensor *idsPh,
                                          MPSGraphTensor *cosPh, MPSGraphTensor *sinPh,
                                          MPSGraphTensor *maskPh,
                                          MPSGraphTensor *dropoutMaskPh,
                                          NSInteger B, NSInteger T);
static MPSGraphTensor *buildCELoss(MPSGraph *g, MPSGraphTensor *logits, MPSGraphTensor *tgtPh,
                                   NSInteger vocab);
// P4 — slot layout helper. useRope=true: [tokEmb, layer0..L-1, finalLN_g, finalLN_b].
//      useRope=false: [tokEmb, posEmb, layer0..L-1, finalLN_g, finalLN_b].
//      Layer 1개당 SwiGLU=18 slot (LN1 2 + Attn 8 + LN2 2 + MLP 6), GELU=16 slot (MLP 4).
static inline NSInteger mlpSlotsPerLayer(BOOL useSwiglu) { return useSwiglu ? 6 : 4; }
static inline NSInteger slotsPerLayer(BOOL useSwiglu)    { return 12 + mlpSlotsPerLayer(useSwiglu); }
static inline NSInteger embeddingSlotCount(BOOL useRope) { return useRope ? 1 : 2; }
static inline NSInteger firstLayerBase(BOOL useRope)     { return embeddingSlotCount(useRope); }
static inline NSInteger finalLnGammaIdx(NSInteger numLayers, BOOL useRope, BOOL useSwiglu) {
    return embeddingSlotCount(useRope) + numLayers * slotsPerLayer(useSwiglu);
}
static inline NSInteger totalSlotCount(NSInteger numLayers, BOOL useRope, BOOL useSwiglu) {
    return finalLnGammaIdx(numLayers, useRope, useSwiglu) + 2;
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
    jboolean useSwiglu, jboolean useRope, jboolean tieWeights,
    jboolean useVariableForStep, jboolean useFp16,
    jboolean useDropout, jfloat dropoutProbability,
    jint gradientAccumulationSteps) {
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
        s.useVariableForStep = useVariableForStep == JNI_TRUE;
        s.useFp16 = useFp16 == JNI_TRUE;
        s.useDropout = useDropout == JNI_TRUE;
        s.dropoutProbability = (float)dropoutProbability;
        s.accumSteps = (NSInteger)gradientAccumulationSteps;
        if (s.accumSteps < 1) s.accumSteps = 1;

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
        // P1.3 — ping-pong alt buffers (매 step alloc 회피).
        id<MTLBuffer> bufAlt = [s.device newBufferWithLength:byteLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> mAlt   = [s.device newBufferWithLength:byteLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> vAlt   = [s.device newBufferWithLength:byteLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> gradAlt = [s.device newBufferWithLength:byteLen options:MTLResourceStorageModeShared];
        if (!buf || !grad || !m || !v || !bufAlt || !mAlt || !vAlt || !gradAlt) {
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
        memset([bufAlt contents], 0, byteLen);
        memset([mAlt contents], 0, byteLen);
        memset([vAlt contents], 0, byteLen);
        memset([gradAlt contents], 0, byteLen);

        while ((NSInteger)s.weights.count <= paramIndex) {
            [s.weights addObject:[NSNull null]];
        }
        PikoWeightSlot *slot = [[PikoWeightSlot alloc] init];
        slot.buffer = buf;
        slot.gradBuffer = grad;
        slot.mBuffer = m;
        slot.vBuffer = v;
        slot.bufferAlt = bufAlt;
        slot.mBufferAlt = mAlt;
        slot.vBufferAlt = vAlt;
        slot.gradBufferAlt = gradAlt;
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
    // P2.1 — fp16 모드 안정성: LN은 fp32로 처리. mean/variance fp16 정밀도 부족 회피.
    //   진입 시 fp32 cast → normalizationWithTensor (epsilon fp32 호환) → 결과를 input dtype으로 cast back.
    MPSDataType origDt = x.dataType;
    BOOL needsCast = (origDt != MPSDataTypeFloat32);
    MPSGraphTensor *xf = needsCast ? [g castTensor:x toType:MPSDataTypeFloat32 name:nil] : x;
    MPSGraphTensor *gf = needsCast ? [g castTensor:gamma toType:MPSDataTypeFloat32 name:nil] : gamma;
    MPSGraphTensor *bf = needsCast ? [g castTensor:beta toType:MPSDataTypeFloat32 name:nil] : beta;
    NSArray<NSNumber *> *axes = @[@(axisDim)];
    MPSGraphTensor *mean = [g meanOfTensor:xf axes:axes name:nil];
    MPSGraphTensor *diff = [g subtractionWithPrimaryTensor:xf secondaryTensor:mean name:nil];
    MPSGraphTensor *diffSq = [g squareWithTensor:diff name:nil];
    MPSGraphTensor *variance = [g meanOfTensor:diffSq axes:axes name:nil];
    MPSGraphTensor *out = [g normalizationWithTensor:xf
                                          meanTensor:mean
                                      varianceTensor:variance
                                         gammaTensor:gf
                                          betaTensor:bf
                                             epsilon:eps
                                                name:nil];
    if (needsCast) out = [g castTensor:out toType:origDt name:nil];
    return out;
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

// P4 — GELU tanh 근사 (turbo TurboGELU.kt와 동일 수식).
//   GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
static MPSGraphTensor *buildGELUActivation(MPSGraph *g, MPSGraphTensor *x) {
    MPSDataType dt = x.dataType;
    MPSGraphTensor *cA    = [g constantWithScalar:0.7978845608028654 shape:@[] dataType:dt];
    MPSGraphTensor *cK    = [g constantWithScalar:0.044715           shape:@[] dataType:dt];
    MPSGraphTensor *cHalf = [g constantWithScalar:0.5                shape:@[] dataType:dt];
    MPSGraphTensor *cOne  = [g constantWithScalar:1.0                shape:@[] dataType:dt];

    MPSGraphTensor *xSq    = [g squareWithTensor:x name:nil];
    MPSGraphTensor *xCube  = [g multiplicationWithPrimaryTensor:xSq secondaryTensor:x name:nil];
    MPSGraphTensor *kxCube = [g multiplicationWithPrimaryTensor:cK  secondaryTensor:xCube name:nil];
    MPSGraphTensor *inner  = [g additionWithPrimaryTensor:x        secondaryTensor:kxCube name:nil];
    MPSGraphTensor *innerA = [g multiplicationWithPrimaryTensor:cA secondaryTensor:inner name:nil];
    MPSGraphTensor *tanhV  = [g tanhWithTensor:innerA name:nil];
    MPSGraphTensor *onePlus = [g additionWithPrimaryTensor:cOne   secondaryTensor:tanhV name:nil];
    MPSGraphTensor *halfX   = [g multiplicationWithPrimaryTensor:cHalf secondaryTensor:x name:nil];
    return [g multiplicationWithPrimaryTensor:halfX secondaryTensor:onePlus name:nil];
}

// P4 — GELU MLP: fc(embedDim→hiddenDim) → GELU → proj(hiddenDim→embedDim).
// turbo의 TurboMLP.forwardGELU와 동치 (hiddenDim = 4·embedDim).
static MPSGraphTensor *buildGeluMLP(MPSGraph *g,
                                    MPSGraphTensor *x,
                                    MPSGraphTensor *fcW,   MPSGraphTensor *fcB,
                                    MPSGraphTensor *projW, MPSGraphTensor *projB) {
    MPSGraphTensor *h = buildLinear(g, x, fcW, fcB);
    MPSGraphTensor *a = buildGELUActivation(g, h);
    return buildLinear(g, a, projW, projB);
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
/**
 * P1.1 — 4D `[B, T, numHeads, headDim]` 입력을 받는 RoPE.
 * cos/sin은 [T, half] 그대로 받아 [1, T, 1, half]로 broadcast.
 * 단위 test의 3D path는 caller가 reshape하면 됨.
 */
static MPSGraphTensor *buildRoPE(MPSGraph *g,
                                 MPSGraphTensor *qOrK,    // [B, T, numHeads, headDim]
                                 MPSGraphTensor *cos,     // [T, headDim/2]
                                 MPSGraphTensor *sin,     // [T, headDim/2]
                                 NSInteger B, NSInteger T, NSInteger numHeads, NSInteger headDim) {
    (void)B;
    NSInteger half = headDim / 2;
    MPSGraphTensor *first = [g sliceTensor:qOrK
                                 dimension:3
                                     start:0
                                    length:half
                                      name:nil];
    MPSGraphTensor *second = [g sliceTensor:qOrK
                                  dimension:3
                                      start:half
                                     length:half
                                       name:nil];

    MPSGraphTensor *cosR = [g reshapeTensor:cos withShape:@[@1, @(T), @1, @(half)] name:nil];
    MPSGraphTensor *sinR = [g reshapeTensor:sin withShape:@[@1, @(T), @1, @(half)] name:nil];

    MPSGraphTensor *fc = [g multiplicationWithPrimaryTensor:first secondaryTensor:cosR name:nil];
    MPSGraphTensor *ss = [g multiplicationWithPrimaryTensor:second secondaryTensor:sinR name:nil];
    MPSGraphTensor *fs = [g multiplicationWithPrimaryTensor:first secondaryTensor:sinR name:nil];
    MPSGraphTensor *sc = [g multiplicationWithPrimaryTensor:second secondaryTensor:cosR name:nil];

    MPSGraphTensor *outFirst  = [g subtractionWithPrimaryTensor:fc secondaryTensor:ss name:nil];
    MPSGraphTensor *outSecond = [g additionWithPrimaryTensor:fs secondaryTensor:sc name:nil];

    return [g concatTensors:@[outFirst, outSecond] dimension:3 name:nil];
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

        // P1.1 — buildRoPE은 4D 전용. 3D 단위 test path에서 [1, T, H, headDim] reshape 어댑터.
        NSArray *headShape4D = @[@1, @(T), @(numHeads), @(headDim)];
        q = [g reshapeTensor:q withShape:headShape4D name:nil];
        k = [g reshapeTensor:k withShape:headShape4D name:nil];
        q = buildRoPE(g, q, cosPh, sinPh, 1, T, numHeads, headDim);
        k = buildRoPE(g, k, cosPh, sinPh, 1, T, numHeads, headDim);
        q = [g reshapeTensor:q withShape:headShape name:nil];
        k = [g reshapeTensor:k withShape:headShape name:nil];

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
/**
 * P1.1 — 4D `[B, T, embedDim]` 입력을 받는 multi-head causal self-attention.
 *   q,k,v reshape to [B, T, H, headDim]
 *   RoPE
 *   transpose 1↔2 → [B, H, T, headDim]
 *   scores = Q @ K^T / √headDim + mask
 *   attn = softmax(scores)
 *   out = attn @ V → transpose back → reshape [B, T, embedDim]
 *   output projection.
 * mask는 [T, T] 그대로 broadcast (브로드캐스트 시 [1, 1, T, T]로 확장).
 */
static MPSGraphTensor *buildAttentionGraph(MPSGraph *g,
                                           MPSGraphTensor *x,
                                           MPSGraphTensor *qW, MPSGraphTensor *qB,
                                           MPSGraphTensor *kW, MPSGraphTensor *kB,
                                           MPSGraphTensor *vW, MPSGraphTensor *vB,
                                           MPSGraphTensor *oW, MPSGraphTensor *oB,
                                           MPSGraphTensor *cos, MPSGraphTensor *sin,
                                           MPSGraphTensor *mask,
                                           NSInteger B, NSInteger T, NSInteger embedDim,
                                           NSInteger numHeads, NSInteger headDim) {
    MPSGraphTensor *q = buildLinear(g, x, qW, qB);
    MPSGraphTensor *k = buildLinear(g, x, kW, kB);
    MPSGraphTensor *v = buildLinear(g, x, vW, vB);
    NSArray *headShape = @[@(B), @(T), @(numHeads), @(headDim)];
    q = [g reshapeTensor:q withShape:headShape name:nil];
    k = [g reshapeTensor:k withShape:headShape name:nil];
    v = [g reshapeTensor:v withShape:headShape name:nil];
    q = buildRoPE(g, q, cos, sin, B, T, numHeads, headDim);
    k = buildRoPE(g, k, cos, sin, B, T, numHeads, headDim);
    // [B, T, H, headDim] → [B, H, T, headDim]
    q = [g transposeTensor:q dimension:1 withDimension:2 name:nil];
    k = [g transposeTensor:k dimension:1 withDimension:2 name:nil];
    v = [g transposeTensor:v dimension:1 withDimension:2 name:nil];
    // K^T: [B, H, T, headDim] → [B, H, headDim, T]
    MPSGraphTensor *kT = [g transposeTensor:k dimension:2 withDimension:3 name:nil];
    MPSGraphTensor *scores = [g matrixMultiplicationWithPrimaryTensor:q secondaryTensor:kT name:nil];
    float scale = 1.0f / sqrtf((float)headDim);
    // P2.1 — constant dtype을 input tensor와 맞춤 (fp16 mode에서 multiply 동일 dtype 강제).
    MPSGraphTensor *sc = [g constantWithScalar:scale shape:@[] dataType:scores.dataType];
    scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:sc name:nil];
    // mask [T, T] → broadcast to [1, 1, T, T]
    MPSGraphTensor *maskR = [g reshapeTensor:mask withShape:@[@1, @1, @(T), @(T)] name:nil];
    scores = [g additionWithPrimaryTensor:scores secondaryTensor:maskR name:nil];
    MPSGraphTensor *attn = [g softMaxWithTensor:scores axis:-1 name:nil];
    MPSGraphTensor *attnV = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:v name:nil];
    // [B, H, T, headDim] → [B, T, H, headDim] → [B, T, embedDim]
    MPSGraphTensor *back = [g transposeTensor:attnV dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *concat = [g reshapeTensor:back withShape:@[@(B), @(T), @(embedDim)] name:nil];
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

        // P1.1 — B=1 어댑터. idsPh를 [1, T]로 두고 helper들이 4D 처리.
        const NSInteger B = 1;
        MPSGraphTensor *idsPh = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"ids"];
        MPSGraphTensor *cosPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"cos"];
        MPSGraphTensor *sinPh = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"sin"];
        MPSGraphTensor *maskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"mask"];

        NSMutableArray<MPSGraphTensor *> *wPh = [NSMutableArray array];
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            MPSGraphTensor *t = [g placeholderWithShape:slot.shape
                                              dataType:MPSDataTypeFloat32
                                                  name:[NSString stringWithFormat:@"w%ld", (long)i]];
            [wPh addObject:t];
        }

        MPSGraphTensor *logits4D = buildForwardLogits(g, s, wPh, idsPh, cosPh, sinPh, maskPh, /*dropoutMaskPh=*/nil, B, T);
        // [1, T, vocab] → [T, vocab] (caller가 보는 flat shape는 그대로)
        MPSGraphTensor *logits = [g reshapeTensor:logits4D withShape:@[@(T), @(vocab)] name:nil];
        (void)numLayers; (void)hiddenDim;

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
        feeds[idsPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
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

/**
 * P1.1 — 4D `[B, T]` ids → 4D logits `[B, T, vocab]`.
 *
 * idsPh shape는 [B, T] (int32). gather + 모든 layer/LayerNorm axis는 마지막 차원(2).
 * cos/sin은 [T, half], mask는 [T, T] 그대로 broadcast.
 */
static MPSGraphTensor *buildForwardLogits(MPSGraph *g, PikoMpsGraphSession *s,
                                          NSArray<MPSGraphTensor *> *wPh,
                                          MPSGraphTensor *idsPh,
                                          MPSGraphTensor *cosPh, MPSGraphTensor *sinPh,
                                          MPSGraphTensor *maskPh,
                                          MPSGraphTensor *dropoutMaskPh,
                                          NSInteger B, NSInteger T) {
    NSInteger numLayers = s.numLayers;
    NSInteger embedDim = s.embedDim;
    NSInteger numHeads = s.numHeads;
    NSInteger headDim = embedDim / numHeads;
    BOOL useRope = s.useRope;
    BOOL useSwiglu = s.useSwiglu;
    NSInteger embStart = embeddingSlotCount(useRope);
    NSInteger layerStride = slotsPerLayer(useSwiglu);

    MPSGraphTensor *tokEmb = wPh[0];
    // idsPh [B, T] → x [B, T, embedDim]
    MPSGraphTensor *x = [g gatherWithUpdatesTensor:tokEmb
                                    indicesTensor:idsPh
                                             axis:0
                                  batchDimensions:0
                                             name:nil];
    // P4 — useRope=false면 learned positional embedding 더하기.
    //   posEmb : [blockSize, embedDim] → 처음 T rows slice → broadcast addition.
    if (!useRope) {
        MPSGraphTensor *posEmb = wPh[1];
        MPSGraphTensor *posSlice = [g sliceTensor:posEmb dimension:0 start:0 length:T name:nil];
        x = [g additionWithPrimaryTensor:x secondaryTensor:posSlice name:nil];
    }

    for (NSInteger L = 0; L < numLayers; L++) {
        NSInteger base = embStart + L * layerStride;
        MPSGraphTensor *g1 = wPh[base+0], *b1 = wPh[base+1];
        MPSGraphTensor *qW = wPh[base+2], *qB = wPh[base+3];
        MPSGraphTensor *kW = wPh[base+4], *kB = wPh[base+5];
        MPSGraphTensor *vW = wPh[base+6], *vB = wPh[base+7];
        MPSGraphTensor *oW = wPh[base+8], *oB = wPh[base+9];
        MPSGraphTensor *g2 = wPh[base+10], *b2 = wPh[base+11];

        MPSGraphTensor *ln1 = buildLayerNorm(g, x, g1, b1, 1e-5f, /*axisDim=*/2);
        MPSGraphTensor *attnOut = buildAttentionGraph(g, ln1, qW, qB, kW, kB, vW, vB, oW, oB,
                                                      cosPh, sinPh, maskPh,
                                                      B, T, embedDim, numHeads, headDim);
        // P4 — attn output dropout.
        if (dropoutMaskPh != nil) {
            MPSGraphTensor *m = [g sliceTensor:dropoutMaskPh dimension:0 start:(2*L) length:1 name:nil];
            m = [g reshapeTensor:m withShape:@[@(B), @(T), @(embedDim)] name:nil];
            attnOut = [g multiplicationWithPrimaryTensor:attnOut secondaryTensor:m name:nil];
        }
        x = [g additionWithPrimaryTensor:x secondaryTensor:attnOut name:nil];

        MPSGraphTensor *ln2 = buildLayerNorm(g, x, g2, b2, 1e-5f, /*axisDim=*/2);
        MPSGraphTensor *mlpOut;
        if (useSwiglu) {
            MPSGraphTensor *gateW = wPh[base+12], *gateB = wPh[base+13];
            MPSGraphTensor *upW   = wPh[base+14], *upB   = wPh[base+15];
            MPSGraphTensor *downW = wPh[base+16], *downB = wPh[base+17];
            mlpOut = buildSwiGLU(g, ln2, gateW, gateB, upW, upB, downW, downB);
        } else {
            MPSGraphTensor *fcW   = wPh[base+12], *fcB   = wPh[base+13];
            MPSGraphTensor *projW = wPh[base+14], *projB = wPh[base+15];
            mlpOut = buildGeluMLP(g, ln2, fcW, fcB, projW, projB);
        }
        // P4 — MLP output dropout.
        if (dropoutMaskPh != nil) {
            MPSGraphTensor *m = [g sliceTensor:dropoutMaskPh dimension:0 start:(2*L+1) length:1 name:nil];
            m = [g reshapeTensor:m withShape:@[@(B), @(T), @(embedDim)] name:nil];
            mlpOut = [g multiplicationWithPrimaryTensor:mlpOut secondaryTensor:m name:nil];
        }
        x = [g additionWithPrimaryTensor:x secondaryTensor:mlpOut name:nil];
    }
    NSInteger finalIdx = finalLnGammaIdx(numLayers, useRope, useSwiglu);
    x = buildLayerNorm(g, x, wPh[finalIdx+0], wPh[finalIdx+1], 1e-5f, /*axisDim=*/2);

    // tied lm head: [B, T, embedDim] @ [embedDim, vocab]
    MPSGraphTensor *embT = [g transposeTensor:tokEmb dimension:0 withDimension:1 name:nil];
    return [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:embT name:nil];
}

/**
 * forward + CE loss. tokens [B*T], targets [B*T]. loss scalar return.
 *
 * loss = -mean_b_t( log_softmax(logits[b,t])[targets[b,t]] )
 *
 * P1.1 — B>1 일반화. tokenIds.size = B*T로 flatten된 입력.
 */
JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunForwardLoss(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jintArray targetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr,
    jint batchSize) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;

        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        NSInteger vocab = s.vocab;

        jsize idsLen = env->GetArrayLength(tokenIdsArr);
        jsize tgtLen = env->GetArrayLength(targetsArr);
        if (tgtLen != idsLen) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "targets length != tokenIds length");
            return -1.0f;
        }
        const NSInteger B = (NSInteger)batchSize;
        if (B <= 0 || (idsLen % B) != 0) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "tokenIds.length must be divisible by batchSize");
            return -1.0f;
        }
        const NSInteger T = idsLen / B;

        // Phase 2 — forward graph cache: (B, T) 변경 시만 rebuild.
        if (s.forwardGraph == nil || s.cachedForwardT != T || s.cachedForwardB != B) {
            MPSGraph *fg = [[MPSGraph alloc] init];
            s.forwardGraph = fg;
            s.cachedForwardT = T;
            s.cachedForwardB = B;
            s.forwardIdsPh  = [fg placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"fw_ids"];
            s.forwardTgtPh  = [fg placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"fw_tgt"];
            s.forwardCosPh  = [fg placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"fw_cos"];
            s.forwardSinPh  = [fg placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"fw_sin"];
            s.forwardMaskPh = [fg placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"fw_mask"];
            s.forwardWPh = [NSMutableArray array];
            for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                [s.forwardWPh addObject:[fg placeholderWithShape:slot.shape
                                                       dataType:MPSDataTypeFloat32
                                                           name:[NSString stringWithFormat:@"fw_w%ld", (long)i]]];
            }
            MPSGraphTensor *fwLogits = buildForwardLogits(fg, s, s.forwardWPh, s.forwardIdsPh,
                                                          s.forwardCosPh, s.forwardSinPh, s.forwardMaskPh,
                                                          /*dropoutMaskPh=*/nil, B, T);
            s.forwardLoss = buildCELoss(fg, fwLogits, s.forwardTgtPh, vocab);
        }
        MPSGraph *g = s.forwardGraph;
        MPSGraphTensor *idsPh  = s.forwardIdsPh;
        MPSGraphTensor *tgtPh  = s.forwardTgtPh;
        MPSGraphTensor *cosPh  = s.forwardCosPh;
        MPSGraphTensor *sinPh  = s.forwardSinPh;
        MPSGraphTensor *maskPh = s.forwardMaskPh;
        NSArray<MPSGraphTensor *> *wPh = s.forwardWPh;
        MPSGraphTensor *loss = s.forwardLoss;

        // Feeds
        NSUInteger idsBytes = (NSUInteger)(B * T) * sizeof(int32_t);
        NSUInteger ropeBytes = (NSUInteger)(T * half) * sizeof(float);
        NSUInteger maskBytes = (NSUInteger)(T * T) * sizeof(float);
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
        feeds[idsPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
        feeds[tgtPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
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

// P1.1 — 4D `[B, T, V]` logits / `[B, T]` targets 처리. mean을 axes @[@0, @1]로.
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
    MPSGraphTensor *perTokenLL = [g reductionSumWithTensor:prod axis:-1 name:nil];  // [B, T]
    MPSGraphTensor *meanLL = [g meanOfTensor:perTokenLL axes:@[@0, @1] name:nil];   // scalar
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
        // P1.1 — B=1 어댑터.
        const NSInteger B = 1;
        MPSGraphTensor *idsPh = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"ids"];
        MPSGraphTensor *tgtPh = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"tgt"];
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
        MPSGraphTensor *logits = buildForwardLogits(g, s, wPh, idsPh, cosPh, sinPh, maskPh, /*dropoutMaskPh=*/nil, B, T);
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
        feeds[idsPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
        feeds[tgtPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
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

static void buildStepGraph(PikoMpsGraphSession *s, NSInteger B, NSInteger T,
                           float beta1, float beta2,
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
    s.cachedB = B;

    s.stepIdsPh  = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"ids"];
    s.stepTgtPh  = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"tgt"];
    s.stepCosPh  = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"cos"];
    s.stepSinPh  = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"sin"];
    s.stepMaskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"mask"];
    s.stepDropoutMaskPh = nil;
    if (s.useDropout) {
        s.stepDropoutMaskPh = [g placeholderWithShape:@[@(2*s.numLayers), @(B), @(T), @(embedDim)]
                                            dataType:MPSDataTypeFloat32 name:@"dropout_mask"];
    }
    s.stepLRPh   = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"lr"];
    s.stepBc1Ph  = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"bc1"];
    s.stepBc2Ph  = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"bc2"];
    // P0.4 — gradient clipping threshold. clip <= 0이면 caller가 1e30f 주입해 disable.
    s.stepClipPh = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"clip"];

    s.stepWPh = [NSMutableArray array];
    s.stepMPh = [NSMutableArray array];
    s.stepVPh = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
        if (s.useVariableForStep) {
            // P1.3 — variable: 초기값은 slot.buffer/mBuffer/vBuffer contents.
            //   weight는 nativeLoadWeights에서 채워져 있음. m/v는 0으로 초기화.
            NSUInteger byteLen = slot.numel * sizeof(float);
            // weight var: slot.buffer contents
            NSData *wData = [NSData dataWithBytes:[slot.buffer contents] length:byteLen];
            slot.stepWVar = [g variableWithData:wData shape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"wvar%ld", (long)i]];
            // m var: zeros (또는 mBuffer가 채워져 있다면 그 contents)
            void *mZeros = calloc(slot.numel, sizeof(float));
            NSData *mData = (slot.mBuffer != nil)
                ? [NSData dataWithBytes:[slot.mBuffer contents] length:byteLen]
                : [NSData dataWithBytes:mZeros length:byteLen];
            slot.stepMVar = [g variableWithData:mData shape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"mvar%ld", (long)i]];
            free(mZeros);
            void *vZeros = calloc(slot.numel, sizeof(float));
            NSData *vData = (slot.vBuffer != nil)
                ? [NSData dataWithBytes:[slot.vBuffer contents] length:byteLen]
                : [NSData dataWithBytes:vZeros length:byteLen];
            slot.stepVVar = [g variableWithData:vData shape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"vvar%ld", (long)i]];
            free(vZeros);
            // variable handle을 그래프 op로 사용하려면 readVariable 필요.
            MPSGraphTensor *wRead = [g readVariable:slot.stepWVar name:nil];
            MPSGraphTensor *mRead = [g readVariable:slot.stepMVar name:nil];
            MPSGraphTensor *vRead = [g readVariable:slot.stepVVar name:nil];
            [s.stepWPh addObject:wRead];
            [s.stepMPh addObject:mRead];
            [s.stepVPh addObject:vRead];
        } else {
            [s.stepWPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"w%ld", (long)i]]];
            [s.stepMPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"m%ld", (long)i]]];
            [s.stepVPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"v%ld", (long)i]]];
        }
    }

    // P2.1 — fp16 mixed precision: wPh/cosPh/sinPh/maskPh를 fp16 cast 후 forward 호출.
    // backward target은 fp32 wPh (autograd가 cast op 통해 chain rule). loss/grad/AdamW는 fp32.
    NSArray<MPSGraphTensor *> *forwardWPh = s.stepWPh;
    MPSGraphTensor *forwardCos = s.stepCosPh;
    MPSGraphTensor *forwardSin = s.stepSinPh;
    MPSGraphTensor *forwardMask = s.stepMaskPh;
    MPSGraphTensor *forwardDropoutMask = s.stepDropoutMaskPh;
    if (s.useFp16) {
        NSMutableArray<MPSGraphTensor *> *wCast = [NSMutableArray array];
        for (MPSGraphTensor *w in s.stepWPh) {
            [wCast addObject:[g castTensor:w toType:MPSDataTypeFloat16 name:nil]];
        }
        forwardWPh = wCast;
        forwardCos = [g castTensor:s.stepCosPh toType:MPSDataTypeFloat16 name:nil];
        forwardSin = [g castTensor:s.stepSinPh toType:MPSDataTypeFloat16 name:nil];
        forwardMask = [g castTensor:s.stepMaskPh toType:MPSDataTypeFloat16 name:nil];
        if (forwardDropoutMask != nil) {
            forwardDropoutMask = [g castTensor:s.stepDropoutMaskPh toType:MPSDataTypeFloat16 name:nil];
        }
    }
    MPSGraphTensor *logits = buildForwardLogits(g, s, forwardWPh, s.stepIdsPh, forwardCos, forwardSin, forwardMask, forwardDropoutMask, B, T);
    if (s.useFp16) {
        logits = [g castTensor:logits toType:MPSDataTypeFloat32 name:nil];
    }
    MPSGraphTensor *loss = buildCELoss(g, logits, s.stepTgtPh, vocab);
    s.stepLoss = loss;

    NSDictionary<MPSGraphTensor *, MPSGraphTensor *> *grads =
        [g gradientForPrimaryTensor:loss withTensors:s.stepWPh name:nil];

    // P0.4 — global gradient norm clipping.
    //   norm  = sqrt(Σ_i Σ_j g_i_j²)
    //   ratio = clip / max(norm, clip)   (norm ≤ clip이면 ratio = 1 → 변화 없음)
    //   g'_i = g_i * ratio
    // clip == 1e30f면 disabled (norm < clip 보장 → ratio = 1).
    MPSGraphTensor *sumSq = nil;
    for (NSInteger i = 0; i < N; i++) {
        MPSGraphTensor *gi = grads[s.stepWPh[i]];
        MPSGraphTensor *sq = [g squareWithTensor:gi name:nil];
        MPSGraphTensor *si = [g reductionSumWithTensor:sq axes:nil name:nil];
        sumSq = (sumSq == nil) ? si : [g additionWithPrimaryTensor:sumSq secondaryTensor:si name:nil];
    }
    MPSGraphTensor *gnorm = [g squareRootWithTensor:sumSq name:nil];
    MPSGraphTensor *maxNC = [g maximumWithPrimaryTensor:gnorm secondaryTensor:s.stepClipPh name:nil];
    MPSGraphTensor *clipRatio = [g divisionWithPrimaryTensor:s.stepClipPh secondaryTensor:maxNC name:nil];

    MPSGraphTensor *cB1   = [g constantWithScalar:beta1 shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cOmB1 = [g constantWithScalar:(1.0f - beta1) shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cB2   = [g constantWithScalar:beta2 shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cOmB2 = [g constantWithScalar:(1.0f - beta2) shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cEps  = [g constantWithScalar:eps shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cWD   = [g constantWithScalar:weightDecay shape:@[] dataType:MPSDataTypeFloat32];

    s.stepWNew = [NSMutableArray array];
    s.stepMNew = [NSMutableArray array];
    s.stepVNew = [NSMutableArray array];
    s.stepAssignOps = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        MPSGraphTensor *w = s.stepWPh[i];
        MPSGraphTensor *gradi = grads[s.stepWPh[i]];
        // P0.4 — clipped gradient
        gradi = [g multiplicationWithPrimaryTensor:gradi secondaryTensor:clipRatio name:nil];
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
        // P1.3 — variable mode: assignVariable로 in-place update.
        if (s.useVariableForStep) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            MPSGraphOperation *opW = [g assignVariable:slot.stepWVar withValueOfTensor:wNew name:nil];
            MPSGraphOperation *opM = [g assignVariable:slot.stepMVar withValueOfTensor:mNew name:nil];
            MPSGraphOperation *opV = [g assignVariable:slot.stepVVar withValueOfTensor:vNew name:nil];
            [s.stepAssignOps addObject:opW];
            [s.stepAssignOps addObject:opM];
            [s.stepAssignOps addObject:opV];
        }
    }
}

JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunTrainingStep(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jintArray targetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr,
    jfloat lr, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay,
    jfloat gradClip, jint stepT, jint batchSize,
    jfloatArray dropoutMaskArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;

        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        NSInteger B = batchSize > 0 ? (NSInteger)batchSize : 1;
        jsize total = env->GetArrayLength(tokenIdsArr);
        jsize T = (jsize)(total / B);
        if ((jsize)(B * T) != total) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "tokenIds length not divisible by batchSize");
            return -1.0f;
        }
        NSInteger N = (NSInteger)s.weights.count;

        // P1.1 — Lazy build (first call) + (B, T) cache key. (B, T) 다르면 rebuild.
        if (s.cachedT == 0 || s.cachedT != T || s.cachedB != B) {
            buildStepGraph(s, B, T, beta1, beta2, eps, weightDecay);
        }
        MPSGraph *g = s.stepGraph;
        // Use cached fields below. lr/bc1/bc2 are placeholders → host computes per step.

        float bc1 = 1.0f - powf(beta1, (float)stepT);
        float bc2 = 1.0f - powf(beta2, (float)stepT);

        // ---- Feeds ----
        NSUInteger idsBytes = B * T * sizeof(int32_t);
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

        // lr/bc1/bc2/clip small placeholders
        id<MTLBuffer> lrBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc1Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc2Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> clipBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        *((float *)[lrBuf contents]) = lr;
        *((float *)[bc1Buf contents]) = bc1;
        *((float *)[bc2Buf contents]) = bc2;
        // gradClip <= 0이면 disable (norm < 1e30f 보장 → ratio = 1).
        *((float *)[clipBuf contents]) = (gradClip > 0.0f) ? gradClip : 1.0e30f;

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[s.stepIdsPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
        feeds[s.stepTgtPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
        feeds[s.stepCosPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[s.stepSinPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[s.stepMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
        feeds[s.stepLRPh]   = [[MPSGraphTensorData alloc] initWithMTLBuffer:lrBuf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.stepBc1Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc1Buf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.stepBc2Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc2Buf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.stepClipPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:clipBuf shape:@[] dataType:MPSDataTypeFloat32];
        // P4 — useDropout=true 시 graph에 dropout placeholder가 있으므로 mask feed 필수.
        // dropoutMaskArr=NULL이면 mask=1 (identity) 강제 (eval-like 호출).
        if (s.useDropout && s.stepDropoutMaskPh != nil) {
            NSInteger numMaskEl = 2 * s.numLayers * B * T * embedDim;
            NSUInteger maskBytes = (NSUInteger)numMaskEl * sizeof(float);
            id<MTLBuffer> dpBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
            if (dropoutMaskArr != NULL) {
                jsize providedLen = env->GetArrayLength(dropoutMaskArr);
                if ((NSInteger)providedLen != numMaskEl) {
                    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                                  "dropoutMask length != 2*numLayers*B*T*embedDim");
                    return -1.0f;
                }
                jfloat *pp = (jfloat *)env->GetPrimitiveArrayCritical(dropoutMaskArr, NULL);
                memcpy([dpBuf contents], pp, maskBytes);
                env->ReleasePrimitiveArrayCritical(dropoutMaskArr, pp, JNI_ABORT);
            } else {
                float *dst = (float *)[dpBuf contents];
                for (NSInteger i = 0; i < numMaskEl; i++) dst[i] = 1.0f;
            }
            feeds[s.stepDropoutMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:dpBuf
                shape:@[@(2*s.numLayers), @(B), @(T), @(embedDim)] dataType:MPSDataTypeFloat32];
        }
        if (!s.useVariableForStep) {
            // placeholder mode: weight/m/v를 매 step feeds로 주입.
            for (NSInteger i = 0; i < N; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                feeds[s.stepWPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer shape:slot.shape dataType:MPSDataTypeFloat32];
                feeds[s.stepMPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
                feeds[s.stepVPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
            }
        }
        // P1.3 variable mode는 weight/m/v feeds 생략. variable 자체가 storage.

        // ---- Results ----
        NSMutableDictionary *results = [NSMutableDictionary dictionary];
        results[s.stepLoss] = [[MPSGraphTensorData alloc] initWithMTLBuffer:lossBuf shape:@[] dataType:MPSDataTypeFloat32];
        if (s.useVariableForStep) {
            // P1.3 variable mode: assignVariable이 in-place update를 graph 안에서 수행.
            // 추가로 wNew/mNew/vNew tensor를 slot.buffer/mBuffer/vBuffer에 sync —
            // readWeight/checkpoint가 latest 값을 보장받기 위해.
            for (NSInteger i = 0; i < N; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                results[s.stepWNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer  shape:slot.shape dataType:MPSDataTypeFloat32];
                results[s.stepMNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
                results[s.stepVNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
            }
        } else {
            // P1.3 placeholder mode — ping-pong: result는 alt buffer에 쓰고 swap.
            for (NSInteger i = 0; i < N; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                results[s.stepWNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.bufferAlt  shape:slot.shape dataType:MPSDataTypeFloat32];
                results[s.stepMNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBufferAlt shape:slot.shape dataType:MPSDataTypeFloat32];
                results[s.stepVNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBufferAlt shape:slot.shape dataType:MPSDataTypeFloat32];
            }
        }

        if (s.useVariableForStep) {
            // P1.3 variable mode: targetOperations로 assignVariable ops 실행 (variable의 in-place update).
            // 같은 run에서 wNew/mNew/vNew를 result로도 받아 slot.buffer에 sync.
            [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                     targetOperations:s.stepAssignOps resultsDictionary:results];
        } else {
            [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                     targetOperations:nil resultsDictionary:results];
        }

        // P1.3 placeholder mode — ping-pong swap: current ↔ alt.
        // variable mode는 slot.buffer가 이미 result로 갱신됨 (swap 불필요).
        if (!s.useVariableForStep) for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            id<MTLBuffer> tmpW = slot.buffer;  slot.buffer  = slot.bufferAlt;  slot.bufferAlt  = tmpW;
            id<MTLBuffer> tmpM = slot.mBuffer; slot.mBuffer = slot.mBufferAlt; slot.mBufferAlt = tmpM;
            id<MTLBuffer> tmpV = slot.vBuffer; slot.vBuffer = slot.vBufferAlt; slot.vBufferAlt = tmpV;
        }

        float lossVal = *((float *)[lossBuf contents]);
        return (jfloat)lossVal;
    }
}

// ============================================================================
// P1.2 — 진정한 grad accumulation: accumGraph + adamGraph 분리
//
// accumGraph: forward + loss + backward → grad_new = grad_old + computed_grad
//   매 micro-step 호출, slot.gradBuffer 누적.
// adamGraph: AdamW update + grad reset
//   N micro-step 후 한 번 호출, weight/m/v 갱신 + grad reset to 0.
// ============================================================================

static void buildAccumGraph(PikoMpsGraphSession *s, NSInteger B, NSInteger T) {
    NSInteger embedDim = s.embedDim;
    NSInteger numHeads = s.numHeads;
    NSInteger headDim = embedDim / numHeads;
    NSInteger half = headDim / 2;
    NSInteger vocab = s.vocab;
    NSInteger N = (NSInteger)s.weights.count;

    MPSGraph *g = [[MPSGraph alloc] init];
    s.accumGraph = g;
    s.cachedAccumT = T;
    s.cachedAccumB = B;

    s.accumIdsPh  = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"a_ids"];
    s.accumTgtPh  = [g placeholderWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32 name:@"a_tgt"];
    s.accumCosPh  = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"a_cos"];
    s.accumSinPh  = [g placeholderWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32 name:@"a_sin"];
    s.accumMaskPh = [g placeholderWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32 name:@"a_mask"];
    s.accumDropoutMaskPh = nil;
    if (s.useDropout) {
        s.accumDropoutMaskPh = [g placeholderWithShape:@[@(2*s.numLayers), @(B), @(T), @(embedDim)]
                                             dataType:MPSDataTypeFloat32 name:@"a_dropout_mask"];
    }

    s.accumWPh = [NSMutableArray array];
    s.accumGradOldPh = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
        [s.accumWPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"a_w%ld", (long)i]]];
        [s.accumGradOldPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"a_go%ld", (long)i]]];
    }

    MPSGraphTensor *logits = buildForwardLogits(g, s, s.accumWPh, s.accumIdsPh, s.accumCosPh, s.accumSinPh, s.accumMaskPh, s.accumDropoutMaskPh, B, T);
    MPSGraphTensor *loss = buildCELoss(g, logits, s.accumTgtPh, vocab);
    s.accumLoss = loss;

    // PyTorch 표준 일치: 매 micro의 backward grad를 1/accumSteps로 스케일.
    // accumSteps=1이면 invA=1.0 → 변경 없음.
    MPSGraphTensor *invA = [g constantWithScalar:(1.0f / (float)s.accumSteps) dataType:MPSDataTypeFloat32];
    MPSGraphTensor *lossScaled = [g multiplicationWithPrimaryTensor:loss secondaryTensor:invA name:@"loss_scaled_for_accum"];

    NSDictionary<MPSGraphTensor *, MPSGraphTensor *> *grads =
        [g gradientForPrimaryTensor:lossScaled withTensors:s.accumWPh name:nil];

    s.accumGradNew = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        MPSGraphTensor *gNew = [g additionWithPrimaryTensor:s.accumGradOldPh[i]
                                            secondaryTensor:grads[s.accumWPh[i]]
                                                       name:nil];
        [s.accumGradNew addObject:gNew];
    }
}

static void buildAdamGraph(PikoMpsGraphSession *s, float beta1, float beta2, float eps, float weightDecay) {
    NSInteger N = (NSInteger)s.weights.count;
    MPSGraph *g = [[MPSGraph alloc] init];
    s.adamGraph = g;
    s.adamBuilt = YES;

    s.adamLRPh   = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"ad_lr"];
    s.adamBc1Ph  = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"ad_bc1"];
    s.adamBc2Ph  = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"ad_bc2"];
    s.adamClipPh = [g placeholderWithShape:@[] dataType:MPSDataTypeFloat32 name:@"ad_clip"];

    s.adamWPh = [NSMutableArray array];
    s.adamMPh = [NSMutableArray array];
    s.adamVPh = [NSMutableArray array];
    s.adamGradPh = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
        [s.adamWPh   addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"ad_w%ld", (long)i]]];
        [s.adamMPh   addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"ad_m%ld", (long)i]]];
        [s.adamVPh   addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"ad_v%ld", (long)i]]];
        [s.adamGradPh addObject:[g placeholderWithShape:slot.shape dataType:MPSDataTypeFloat32 name:[NSString stringWithFormat:@"ad_g%ld", (long)i]]];
    }

    // global norm clipping (step graph와 동일 로직)
    MPSGraphTensor *sumSq = nil;
    for (NSInteger i = 0; i < N; i++) {
        MPSGraphTensor *sq = [g squareWithTensor:s.adamGradPh[i] name:nil];
        MPSGraphTensor *si = [g reductionSumWithTensor:sq axes:nil name:nil];
        sumSq = (sumSq == nil) ? si : [g additionWithPrimaryTensor:sumSq secondaryTensor:si name:nil];
    }
    MPSGraphTensor *gnorm = [g squareRootWithTensor:sumSq name:nil];
    MPSGraphTensor *maxNC = [g maximumWithPrimaryTensor:gnorm secondaryTensor:s.adamClipPh name:nil];
    MPSGraphTensor *clipRatio = [g divisionWithPrimaryTensor:s.adamClipPh secondaryTensor:maxNC name:nil];

    MPSGraphTensor *cB1   = [g constantWithScalar:beta1 shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cOmB1 = [g constantWithScalar:(1.0f - beta1) shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cB2   = [g constantWithScalar:beta2 shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cOmB2 = [g constantWithScalar:(1.0f - beta2) shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cEps  = [g constantWithScalar:eps shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *cWD   = [g constantWithScalar:weightDecay shape:@[] dataType:MPSDataTypeFloat32];
    MPSGraphTensor *zero  = [g constantWithScalar:0.0f shape:@[] dataType:MPSDataTypeFloat32];

    s.adamWNew = [NSMutableArray array];
    s.adamMNew = [NSMutableArray array];
    s.adamVNew = [NSMutableArray array];
    s.adamGradReset = [NSMutableArray array];
    for (NSInteger i = 0; i < N; i++) {
        MPSGraphTensor *w = s.adamWPh[i];
        MPSGraphTensor *m = s.adamMPh[i];
        MPSGraphTensor *v = s.adamVPh[i];
        MPSGraphTensor *grad = s.adamGradPh[i];
        // clipped grad
        grad = [g multiplicationWithPrimaryTensor:grad secondaryTensor:clipRatio name:nil];

        MPSGraphTensor *mNew = [g additionWithPrimaryTensor:
                                   [g multiplicationWithPrimaryTensor:cB1 secondaryTensor:m name:nil]
                                          secondaryTensor:
                                   [g multiplicationWithPrimaryTensor:cOmB1 secondaryTensor:grad name:nil]
                                                         name:nil];
        MPSGraphTensor *g2 = [g squareWithTensor:grad name:nil];
        MPSGraphTensor *vNew = [g additionWithPrimaryTensor:
                                   [g multiplicationWithPrimaryTensor:cB2 secondaryTensor:v name:nil]
                                          secondaryTensor:
                                   [g multiplicationWithPrimaryTensor:cOmB2 secondaryTensor:g2 name:nil]
                                                         name:nil];
        MPSGraphTensor *mHat = [g divisionWithPrimaryTensor:mNew secondaryTensor:s.adamBc1Ph name:nil];
        MPSGraphTensor *vHat = [g divisionWithPrimaryTensor:vNew secondaryTensor:s.adamBc2Ph name:nil];
        MPSGraphTensor *denom = [g additionWithPrimaryTensor:
                                    [g squareRootWithTensor:vHat name:nil]
                                            secondaryTensor:cEps name:nil];
        MPSGraphTensor *stepUpd = [g divisionWithPrimaryTensor:mHat secondaryTensor:denom name:nil];
        MPSGraphTensor *lrStep = [g multiplicationWithPrimaryTensor:s.adamLRPh secondaryTensor:stepUpd name:nil];
        MPSGraphTensor *lrWD = [g multiplicationWithPrimaryTensor:s.adamLRPh secondaryTensor:cWD name:nil];
        MPSGraphTensor *wdW = [g multiplicationWithPrimaryTensor:lrWD secondaryTensor:w name:nil];
        MPSGraphTensor *wNew = [g subtractionWithPrimaryTensor:
                                   [g subtractionWithPrimaryTensor:w secondaryTensor:lrStep name:nil]
                                            secondaryTensor:wdW name:nil];

        // grad reset = 0 (broadcast 가능하도록 zeros same shape)
        MPSGraphTensor *gradZero = [g multiplicationWithPrimaryTensor:grad secondaryTensor:zero name:nil];

        [s.adamWNew addObject:wNew];
        [s.adamMNew addObject:mNew];
        [s.adamVNew addObject:vNew];
        [s.adamGradReset addObject:gradZero];
    }
}

/**
 * P1.2 — 1 micro-step: forward + loss + backward + grad accumulate.
 * slot.gradBuffer ↔ gradBufferAlt swap (누적된 grad는 swap 후 slot.gradBuffer에 들어감).
 * 호출 전 nativeResetGradAccum으로 초기화 필요 (또는 직전 adam step이 reset 함).
 */
JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunAccumStep(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray tokenIdsArr, jintArray targetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr,
    jint batchSize, jfloatArray dropoutMaskArr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;

        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;
        NSInteger B = batchSize > 0 ? (NSInteger)batchSize : 1;
        jsize total = env->GetArrayLength(tokenIdsArr);
        jsize T = (jsize)(total / B);
        NSInteger N = (NSInteger)s.weights.count;

        if (s.cachedAccumT == 0 || s.cachedAccumT != T || s.cachedAccumB != B) {
            buildAccumGraph(s, B, T);
        }
        MPSGraph *g = s.accumGraph;

        NSUInteger idsBytes = B * T * sizeof(int32_t);
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
        feeds[s.accumIdsPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
        feeds[s.accumTgtPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBuf shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
        feeds[s.accumCosPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[s.accumSinPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
        feeds[s.accumMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
        // P4 — dropout mask feed (useDropout=true 시점만 graph에 placeholder 있음).
        if (s.useDropout && s.accumDropoutMaskPh != nil) {
            NSInteger numMaskEl = 2 * s.numLayers * B * T * embedDim;
            NSUInteger maskBytes2 = (NSUInteger)numMaskEl * sizeof(float);
            id<MTLBuffer> dpBuf = [s.device newBufferWithLength:maskBytes2 options:MTLResourceStorageModeShared];
            if (dropoutMaskArr != NULL) {
                jsize providedLen = env->GetArrayLength(dropoutMaskArr);
                if ((NSInteger)providedLen != numMaskEl) {
                    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                                  "dropoutMask length != 2*numLayers*B*T*embedDim");
                    return -1.0f;
                }
                jfloat *pp = (jfloat *)env->GetPrimitiveArrayCritical(dropoutMaskArr, NULL);
                memcpy([dpBuf contents], pp, maskBytes2);
                env->ReleasePrimitiveArrayCritical(dropoutMaskArr, pp, JNI_ABORT);
            } else {
                float *dst = (float *)[dpBuf contents];
                for (NSInteger i = 0; i < numMaskEl; i++) dst[i] = 1.0f;
            }
            feeds[s.accumDropoutMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:dpBuf
                shape:@[@(2*s.numLayers), @(B), @(T), @(embedDim)] dataType:MPSDataTypeFloat32];
        }
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feeds[s.accumWPh[i]]       = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer     shape:slot.shape dataType:MPSDataTypeFloat32];
            feeds[s.accumGradOldPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
        }

        NSMutableDictionary *results = [NSMutableDictionary dictionary];
        results[s.accumLoss] = [[MPSGraphTensorData alloc] initWithMTLBuffer:lossBuf shape:@[] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            results[s.accumGradNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBufferAlt shape:slot.shape dataType:MPSDataTypeFloat32];
        }

        [g runWithMTLCommandQueue:s.commandQueue feeds:feeds
                 targetOperations:nil resultsDictionary:results];

        // swap grad buffers
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            id<MTLBuffer> tmp = slot.gradBuffer;
            slot.gradBuffer = slot.gradBufferAlt;
            slot.gradBufferAlt = tmp;
        }

        return (jfloat)(*((float *)[lossBuf contents]));
    }
}

/**
 * P1.2 — AdamW 1 step (누적된 slot.gradBuffer 사용). w/m/v 갱신 + grad reset.
 */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeRunAdamStep(
    JNIEnv *env, jclass clazz, jlong handle,
    jfloat lr, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay,
    jfloat gradClip, jint stepT) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;

        if (!s.adamBuilt) {
            buildAdamGraph(s, beta1, beta2, eps, weightDecay);
        }
        NSInteger N = (NSInteger)s.weights.count;

        float bc1 = 1.0f - powf(beta1, (float)stepT);
        float bc2 = 1.0f - powf(beta2, (float)stepT);

        id<MTLBuffer> lrBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc1Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc2Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> clipBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        *((float *)[lrBuf contents]) = lr;
        *((float *)[bc1Buf contents]) = bc1;
        *((float *)[bc2Buf contents]) = bc2;
        *((float *)[clipBuf contents]) = (gradClip > 0.0f) ? gradClip : 1.0e30f;

        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[s.adamLRPh]   = [[MPSGraphTensorData alloc] initWithMTLBuffer:lrBuf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.adamBc1Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc1Buf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.adamBc2Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc2Buf shape:@[] dataType:MPSDataTypeFloat32];
        feeds[s.adamClipPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:clipBuf shape:@[] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feeds[s.adamWPh[i]]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer     shape:slot.shape dataType:MPSDataTypeFloat32];
            feeds[s.adamMPh[i]]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBuffer    shape:slot.shape dataType:MPSDataTypeFloat32];
            feeds[s.adamVPh[i]]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBuffer    shape:slot.shape dataType:MPSDataTypeFloat32];
            feeds[s.adamGradPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
        }

        NSMutableDictionary *results = [NSMutableDictionary dictionary];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            results[s.adamWNew[i]]      = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.bufferAlt     shape:slot.shape dataType:MPSDataTypeFloat32];
            results[s.adamMNew[i]]      = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBufferAlt    shape:slot.shape dataType:MPSDataTypeFloat32];
            results[s.adamVNew[i]]      = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBufferAlt    shape:slot.shape dataType:MPSDataTypeFloat32];
            results[s.adamGradReset[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBufferAlt shape:slot.shape dataType:MPSDataTypeFloat32];
        }

        [s.adamGraph runWithMTLCommandQueue:s.commandQueue feeds:feeds
                          targetOperations:nil resultsDictionary:results];

        // swap all
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            id<MTLBuffer> tw = slot.buffer;     slot.buffer = slot.bufferAlt;         slot.bufferAlt = tw;
            id<MTLBuffer> tm = slot.mBuffer;    slot.mBuffer = slot.mBufferAlt;       slot.mBufferAlt = tm;
            id<MTLBuffer> tv = slot.vBuffer;    slot.vBuffer = slot.vBufferAlt;       slot.vBufferAlt = tv;
            id<MTLBuffer> tg = slot.gradBuffer; slot.gradBuffer = slot.gradBufferAlt; slot.gradBufferAlt = tg;
        }
    }
}

/** P1.2 — 모든 slot.gradBuffer를 0으로 reset (학습 시작 또는 강제 reset용). */
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeResetGradAccum(
    JNIEnv *env, jclass clazz, jlong handle) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        for (NSInteger i = 0; i < (NSInteger)s.weights.count; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            NSUInteger by = slot.numel * sizeof(float);
            memset([slot.gradBuffer contents], 0, by);
        }
    }
}

/**
 * Phase 3 — fused step: accumSteps 회 forward+backward+grad accumulate + AdamW를
 * **단일 MTLCommandBuffer**에 encode + commit + wait. 매 iter 9번 분리된 GPU run을
 * 한 번의 commit으로 묶어 host-GPU sync 8회 제거 + GPU 작업 사이 host orchestration overhead 제거.
 *
 * 입력:
 *   allTokenIdsArr: [accumSteps * B * T] flatten
 *   allTargetsArr:  [accumSteps * B * T]
 *   cos/sin/mask:   매 micro 공통 (host precompute)
 *   allDropoutMaskArr: [accumSteps * 2*L*B*T*E] flatten (useDropout=true 시점만). NULL이면 mask=1.
 *
 * 반환: 마지막 micro-step의 loss (eval 가시성용).
 */
JNIEXPORT jfloat JNICALL
Java_mps_MpsGraphSession_nativeRunFusedStep(
    JNIEnv *env, jclass clazz, jlong handle,
    jintArray allTokenIdsArr, jintArray allTargetsArr,
    jfloatArray cosArr, jfloatArray sinArr, jfloatArray maskArr,
    jfloatArray allDropoutMaskArr,
    jfloat lr, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay,
    jfloat gradClip, jint stepT, jint batchSize, jint accumSteps) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return -1.0f;
        NSInteger N = (NSInteger)s.weights.count;
        NSInteger B = (NSInteger)batchSize;
        NSInteger A = (NSInteger)accumSteps;
        jsize totalLen = env->GetArrayLength(allTokenIdsArr);
        if (B <= 0 || A <= 0 || (totalLen % (A * B)) != 0) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "allTokenIds length not divisible by accumSteps*B");
            return -1.0f;
        }
        NSInteger T = totalLen / (A * B);
        NSInteger embedDim = s.embedDim;
        NSInteger numHeads = s.numHeads;
        NSInteger headDim = embedDim / numHeads;
        NSInteger half = headDim / 2;

        // graph build (cache)
        if (s.cachedAccumT == 0 || s.cachedAccumT != T || s.cachedAccumB != B) {
            buildAccumGraph(s, B, T);
        }
        if (!s.adamBuilt) {
            buildAdamGraph(s, beta1, beta2, eps, weightDecay);
        }

        // shared bufs (cos/sin/mask)
        NSUInteger idsBytes = (NSUInteger)B * T * sizeof(int32_t);
        NSUInteger ropeBytes = (NSUInteger)T * half * sizeof(float);
        NSUInteger maskBytes = (NSUInteger)T * T * sizeof(float);
        id<MTLBuffer> cosBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinBuf = [s.device newBufferWithLength:ropeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> maskBuf = [s.device newBufferWithLength:maskBytes options:MTLResourceStorageModeShared];
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

        // micro-step ids/tgt/dropoutMask/loss buffers
        NSInteger dpMaskNumel = 2 * s.numLayers * B * T * embedDim;
        NSUInteger dpMaskBytes = (NSUInteger)dpMaskNumel * sizeof(float);
        BOOL useDP = (s.useDropout && s.accumDropoutMaskPh != nil);
        NSMutableArray *idsBufs = [NSMutableArray arrayWithCapacity:A];
        NSMutableArray *tgtBufs = [NSMutableArray arrayWithCapacity:A];
        NSMutableArray *dpBufs  = [NSMutableArray arrayWithCapacity:A];
        NSMutableArray *lossBufs = [NSMutableArray arrayWithCapacity:A];

        jint *allTok = (jint *)env->GetPrimitiveArrayCritical(allTokenIdsArr, NULL);
        jint *allTgt = (jint *)env->GetPrimitiveArrayCritical(allTargetsArr, NULL);
        jfloat *allDp = NULL;
        if (useDP && allDropoutMaskArr != NULL) {
            jsize provided = env->GetArrayLength(allDropoutMaskArr);
            if ((NSInteger)provided != A * dpMaskNumel) {
                env->ReleasePrimitiveArrayCritical(allTokenIdsArr, allTok, JNI_ABORT);
                env->ReleasePrimitiveArrayCritical(allTargetsArr, allTgt, JNI_ABORT);
                env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                              "allDropoutMask length != accumSteps * 2*L*B*T*E");
                return -1.0f;
            }
            allDp = (jfloat *)env->GetPrimitiveArrayCritical(allDropoutMaskArr, NULL);
        }
        for (NSInteger m = 0; m < A; m++) {
            id<MTLBuffer> idsBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> tgtBuf = [s.device newBufferWithLength:idsBytes options:MTLResourceStorageModeShared];
            memcpy([idsBuf contents], allTok + m * B * T, idsBytes);
            memcpy([tgtBuf contents], allTgt + m * B * T, idsBytes);
            [idsBufs addObject:idsBuf];
            [tgtBufs addObject:tgtBuf];
            if (useDP) {
                id<MTLBuffer> dpBuf = [s.device newBufferWithLength:dpMaskBytes options:MTLResourceStorageModeShared];
                if (allDp != NULL) {
                    memcpy([dpBuf contents], allDp + m * dpMaskNumel, dpMaskBytes);
                } else {
                    float *dst = (float *)[dpBuf contents];
                    for (NSInteger i = 0; i < dpMaskNumel; i++) dst[i] = 1.0f;
                }
                [dpBufs addObject:dpBuf];
            } else {
                [dpBufs addObject:[NSNull null]];
            }
            id<MTLBuffer> lossBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
            [lossBufs addObject:lossBuf];
        }
        env->ReleasePrimitiveArrayCritical(allTokenIdsArr, allTok, JNI_ABORT);
        env->ReleasePrimitiveArrayCritical(allTargetsArr, allTgt, JNI_ABORT);
        if (allDp != NULL) env->ReleasePrimitiveArrayCritical(allDropoutMaskArr, allDp, JNI_ABORT);

        // adam scalars
        float bc1 = 1.0f - powf(beta1, (float)stepT);
        float bc2 = 1.0f - powf(beta2, (float)stepT);
        id<MTLBuffer> lrBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc1Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc2Buf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> clipBuf = [s.device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        *((float *)[lrBuf contents]) = lr;
        *((float *)[bc1Buf contents]) = bc1;
        *((float *)[bc2Buf contents]) = bc2;
        *((float *)[clipBuf contents]) = (gradClip > 0.0f) ? gradClip : 1.0e30f;

        // single commandBuffer for all 9 encodes.
        // `encodeToCommandBuffer:`는 `MPSCommandBuffer*`를 요구 (raw MTLCommandBuffer 넣으면
        // MPS가 내부 `[buf commandBuffer]` 셀렉터 호출하다 NSInvalidArgumentException).
        MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferFromCommandQueue:s.commandQueue];

        for (NSInteger m = 0; m < A; m++) {
            NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
            feeds[s.accumIdsPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:idsBufs[m] shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
            feeds[s.accumTgtPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:tgtBufs[m] shape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
            feeds[s.accumCosPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:cosBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
            feeds[s.accumSinPh]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:sinBuf shape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
            feeds[s.accumMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:maskBuf shape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
            if (useDP && dpBufs[m] != [NSNull null]) {
                feeds[s.accumDropoutMaskPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:dpBufs[m]
                    shape:@[@(2*s.numLayers), @(B), @(T), @(embedDim)] dataType:MPSDataTypeFloat32];
            }
            for (NSInteger i = 0; i < N; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                feeds[s.accumWPh[i]]       = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer     shape:slot.shape dataType:MPSDataTypeFloat32];
                feeds[s.accumGradOldPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
            }
            NSMutableDictionary *results = [NSMutableDictionary dictionary];
            results[s.accumLoss] = [[MPSGraphTensorData alloc] initWithMTLBuffer:lossBufs[m] shape:@[] dataType:MPSDataTypeFloat32];
            for (NSInteger i = 0; i < N; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                results[s.accumGradNew[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBufferAlt shape:slot.shape dataType:MPSDataTypeFloat32];
            }
            [s.accumGraph encodeToCommandBuffer:cb feeds:feeds targetOperations:nil resultsDictionary:results executionDescriptor:nil];
            // host-side ping-pong swap: 다음 encode가 새 slot.gradBuffer (=방금 result로 잡은 buffer) 참조하도록.
            for (NSInteger i = 0; i < N; i++) {
                PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
                id<MTLBuffer> tmp = slot.gradBuffer; slot.gradBuffer = slot.gradBufferAlt; slot.gradBufferAlt = tmp;
            }
        }

        // adam encode (uses fully accumulated slot.gradBuffer)
        NSMutableDictionary *adamFeeds = [NSMutableDictionary dictionary];
        adamFeeds[s.adamLRPh]   = [[MPSGraphTensorData alloc] initWithMTLBuffer:lrBuf shape:@[] dataType:MPSDataTypeFloat32];
        adamFeeds[s.adamBc1Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc1Buf shape:@[] dataType:MPSDataTypeFloat32];
        adamFeeds[s.adamBc2Ph]  = [[MPSGraphTensorData alloc] initWithMTLBuffer:bc2Buf shape:@[] dataType:MPSDataTypeFloat32];
        adamFeeds[s.adamClipPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:clipBuf shape:@[] dataType:MPSDataTypeFloat32];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            adamFeeds[s.adamWPh[i]]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.buffer     shape:slot.shape dataType:MPSDataTypeFloat32];
            adamFeeds[s.adamMPh[i]]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBuffer    shape:slot.shape dataType:MPSDataTypeFloat32];
            adamFeeds[s.adamVPh[i]]    = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBuffer    shape:slot.shape dataType:MPSDataTypeFloat32];
            adamFeeds[s.adamGradPh[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBuffer shape:slot.shape dataType:MPSDataTypeFloat32];
        }
        NSMutableDictionary *adamResults = [NSMutableDictionary dictionary];
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            adamResults[s.adamWNew[i]]      = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.bufferAlt     shape:slot.shape dataType:MPSDataTypeFloat32];
            adamResults[s.adamMNew[i]]      = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.mBufferAlt    shape:slot.shape dataType:MPSDataTypeFloat32];
            adamResults[s.adamVNew[i]]      = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.vBufferAlt    shape:slot.shape dataType:MPSDataTypeFloat32];
            adamResults[s.adamGradReset[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:slot.gradBufferAlt shape:slot.shape dataType:MPSDataTypeFloat32];
        }
        [s.adamGraph encodeToCommandBuffer:cb feeds:adamFeeds targetOperations:nil resultsDictionary:adamResults executionDescriptor:nil];

        [cb commit];
        [cb waitUntilCompleted];

        // adam swap (weight/m/v/grad 모두)
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            id<MTLBuffer> tw = slot.buffer;     slot.buffer = slot.bufferAlt;         slot.bufferAlt = tw;
            id<MTLBuffer> tm = slot.mBuffer;    slot.mBuffer = slot.mBufferAlt;       slot.mBufferAlt = tm;
            id<MTLBuffer> tv = slot.vBuffer;    slot.vBuffer = slot.vBufferAlt;       slot.vBufferAlt = tv;
            id<MTLBuffer> tg = slot.gradBuffer; slot.gradBuffer = slot.gradBufferAlt; slot.gradBufferAlt = tg;
        }

        return (jfloat)(*((float *)[(id<MTLBuffer>)lossBufs[A - 1] contents]));
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

// P0.1 — AdamW state (m, v) read/write. Checkpoint 저장/복원용.
JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeReadOptimizerM(
    JNIEnv *env, jclass clazz, jlong handle, jint paramIndex, jfloatArray out) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[paramIndex];
        NSUInteger bytes = slot.numel * sizeof(float);
        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(out, NULL);
        memcpy(p, [slot.mBuffer contents], bytes);
        env->ReleasePrimitiveArrayCritical(out, p, 0);
    }
}

JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeReadOptimizerV(
    JNIEnv *env, jclass clazz, jlong handle, jint paramIndex, jfloatArray out) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[paramIndex];
        NSUInteger bytes = slot.numel * sizeof(float);
        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(out, NULL);
        memcpy(p, [slot.vBuffer contents], bytes);
        env->ReleasePrimitiveArrayCritical(out, p, 0);
    }
}

JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeLoadOptimizerM(
    JNIEnv *env, jclass clazz, jlong handle, jint paramIndex, jfloatArray data) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[paramIndex];
        jsize len = env->GetArrayLength(data);
        if ((NSUInteger)len != slot.numel) {
            env->ThrowNew(
                env->FindClass("java/lang/IllegalArgumentException"),
                "optimizer m data length != slot.numel");
            return;
        }
        NSUInteger bytes = slot.numel * sizeof(float);
        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(data, NULL);
        memcpy([slot.mBuffer contents], p, bytes);
        env->ReleasePrimitiveArrayCritical(data, p, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_mps_MpsGraphSession_nativeLoadOptimizerV(
    JNIEnv *env, jclass clazz, jlong handle, jint paramIndex, jfloatArray data) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return;
        PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[paramIndex];
        jsize len = env->GetArrayLength(data);
        if ((NSUInteger)len != slot.numel) {
            env->ThrowNew(
                env->FindClass("java/lang/IllegalArgumentException"),
                "optimizer v data length != slot.numel");
            return;
        }
        NSUInteger bytes = slot.numel * sizeof(float);
        jfloat *p = (jfloat *)env->GetPrimitiveArrayCritical(data, NULL);
        memcpy([slot.vBuffer contents], p, bytes);
        env->ReleasePrimitiveArrayCritical(data, p, JNI_ABORT);
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

// ============================================================================
// P3.1 — MPSGraphExecutable serialize / deserialize.
//
// 디스크에 stepGraph를 compiled MPSGraphPackage로 직렬화.
//   - compile: graph + feeds (shapedType dict) + targetTensors → MPSGraphExecutable
//   - serialize: executable.serializeToMPSGraphPackageAtURL (macOS 14+)
//   - deserialize: MPSGraphExecutable initWithMPSGraphPackageAtURL
//
// 본 PoC: compile + serialize roundtrip 검증 (test 통과).
// run path는 graph.runWithMTLCommandQueue 그대로 유지 — Executable 호출 path는 후속 작업
// (inputsArray ordering이 광범위 refactor 필요해서 PoC scope에서 제외).
// ============================================================================

static MPSGraphExecutable *compileStepExecutable(PikoMpsGraphSession *s, NSInteger B, NSInteger T,
                                                 float beta1, float beta2, float eps, float weightDecay) {
    // stepGraph가 build되지 않았다면 먼저 build.
    if (s.cachedT == 0 || s.cachedT != T || s.cachedB != B) {
        buildStepGraph(s, B, T, beta1, beta2, eps, weightDecay);
    }
    MPSGraph *g = s.stepGraph;
    NSInteger embedDim = s.embedDim;
    NSInteger numHeads = s.numHeads;
    NSInteger headDim = embedDim / numHeads;
    NSInteger half = headDim / 2;
    NSInteger N = (NSInteger)s.weights.count;

    // feeds (shapedType dict)
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feedsDict = [NSMutableDictionary dictionary];
    feedsDict[s.stepIdsPh] = [[MPSGraphShapedType alloc] initWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
    feedsDict[s.stepTgtPh] = [[MPSGraphShapedType alloc] initWithShape:@[@(B), @(T)] dataType:MPSDataTypeInt32];
    feedsDict[s.stepCosPh] = [[MPSGraphShapedType alloc] initWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
    feedsDict[s.stepSinPh] = [[MPSGraphShapedType alloc] initWithShape:@[@(T), @(half)] dataType:MPSDataTypeFloat32];
    feedsDict[s.stepMaskPh] = [[MPSGraphShapedType alloc] initWithShape:@[@(T), @(T)] dataType:MPSDataTypeFloat32];
    feedsDict[s.stepLRPh] = [[MPSGraphShapedType alloc] initWithShape:@[] dataType:MPSDataTypeFloat32];
    feedsDict[s.stepBc1Ph] = [[MPSGraphShapedType alloc] initWithShape:@[] dataType:MPSDataTypeFloat32];
    feedsDict[s.stepBc2Ph] = [[MPSGraphShapedType alloc] initWithShape:@[] dataType:MPSDataTypeFloat32];
    feedsDict[s.stepClipPh] = [[MPSGraphShapedType alloc] initWithShape:@[] dataType:MPSDataTypeFloat32];
    if (s.useDropout && s.stepDropoutMaskPh != nil) {
        feedsDict[s.stepDropoutMaskPh] = [[MPSGraphShapedType alloc]
            initWithShape:@[@(2*s.numLayers), @(B), @(T), @(embedDim)] dataType:MPSDataTypeFloat32];
    }
    if (!s.useVariableForStep) {
        for (NSInteger i = 0; i < N; i++) {
            PikoWeightSlot *slot = (PikoWeightSlot *)s.weights[i];
            feedsDict[s.stepWPh[i]] = [[MPSGraphShapedType alloc] initWithShape:slot.shape dataType:MPSDataTypeFloat32];
            feedsDict[s.stepMPh[i]] = [[MPSGraphShapedType alloc] initWithShape:slot.shape dataType:MPSDataTypeFloat32];
            feedsDict[s.stepVPh[i]] = [[MPSGraphShapedType alloc] initWithShape:slot.shape dataType:MPSDataTypeFloat32];
        }
    }
    // variable mode는 weight가 graph 내부 storage이므로 feeds 안 들어감.

    // target: loss + stepWNew/stepMNew/stepVNew
    NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray array];
    [targets addObject:s.stepLoss];
    for (NSInteger i = 0; i < N; i++) {
        [targets addObject:s.stepWNew[i]];
        [targets addObject:s.stepMNew[i]];
        [targets addObject:s.stepVNew[i]];
    }

    MPSGraphDevice *gpuDev = [MPSGraphDevice deviceWithMTLDevice:s.device];
    MPSGraphCompilationDescriptor *cd = [[MPSGraphCompilationDescriptor alloc] init];
    NSArray<MPSGraphOperation *> *targetOps = s.useVariableForStep ? s.stepAssignOps : nil;
    MPSGraphExecutable *exe = [g compileWithDevice:gpuDev
                                              feeds:feedsDict
                                      targetTensors:targets
                                   targetOperations:targetOps
                              compilationDescriptor:cd];
    return exe;
}

JNIEXPORT jboolean JNICALL
Java_mps_MpsGraphSession_nativeCompileStepAndSerialize(
    JNIEnv *env, jclass clazz, jlong handle, jstring pathStr,
    jint B, jint T, jfloat beta1, jfloat beta2, jfloat eps, jfloat weightDecay) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return JNI_FALSE;
        const char *cpath = env->GetStringUTFChars(pathStr, NULL);
        NSString *nspath = [NSString stringWithUTF8String:cpath];
        env->ReleaseStringUTFChars(pathStr, cpath);
        NSURL *url = [NSURL fileURLWithPath:nspath isDirectory:YES];

        MPSGraphExecutable *exe = compileStepExecutable(s, (NSInteger)B, (NSInteger)T,
                                                        beta1, beta2, eps, weightDecay);
        if (!exe) return JNI_FALSE;

        MPSGraphExecutableSerializationDescriptor *sd = [[MPSGraphExecutableSerializationDescriptor alloc] init];
        @try {
            [exe serializeToMPSGraphPackageAtURL:url descriptor:sd];
        } @catch (NSException *e) {
            NSLog(@"serializeToMPSGraphPackageAtURL exception: %@", e);
            return JNI_FALSE;
        }
        // 파일이 실제 생성되었는지 빠른 확인.
        NSFileManager *fm = [NSFileManager defaultManager];
        BOOL isDir = NO;
        return [fm fileExistsAtPath:nspath isDirectory:&isDir] ? JNI_TRUE : JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_mps_MpsGraphSession_nativeLoadStepExecutable(
    JNIEnv *env, jclass clazz, jlong handle, jstring pathStr) {
    @autoreleasepool {
        PikoMpsGraphSession *s = sessionFromHandle(handle);
        if (!s) return JNI_FALSE;
        const char *cpath = env->GetStringUTFChars(pathStr, NULL);
        NSString *nspath = [NSString stringWithUTF8String:cpath];
        env->ReleaseStringUTFChars(pathStr, cpath);
        NSURL *url = [NSURL fileURLWithPath:nspath isDirectory:YES];

        NSFileManager *fm = [NSFileManager defaultManager];
        BOOL isDir = NO;
        if (![fm fileExistsAtPath:nspath isDirectory:&isDir]) return JNI_FALSE;

        MPSGraphCompilationDescriptor *cd = [[MPSGraphCompilationDescriptor alloc] init];
        @try {
            MPSGraphExecutable *exe = [[MPSGraphExecutable alloc] initWithMPSGraphPackageAtURL:url
                                                                          compilationDescriptor:cd];
            if (!exe) {
                NSLog(@"initWithMPSGraphPackageAtURL returned nil");
                return JNI_FALSE;
            }
            // feedTensors/targetTensors는 deserialize 직후 nil일 수 있음 (specialize 전).
            // executable 객체 자체가 존재하면 성공.
            return JNI_TRUE;
        } @catch (NSException *e) {
            NSLog(@"initWithMPSGraphPackageAtURL exception: %@", e);
            return JNI_FALSE;
        }
    }
}

#ifdef __cplusplus
}
#endif
