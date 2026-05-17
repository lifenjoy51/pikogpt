package mps.jni

/**
 * libpikogpt_metal.dylib의 Metal MatMul JNI 진입점.
 *
 * 실제 native 함수는 `src/main/objc/MetalMatMulBridge.m`에 정의.
 * 호출자(`mps.ops.mpsMatmul`)는 인자 검증을 끝낸 뒤에만 호출해야 한다 — 여기선 가정 검사 없음.
 *
 * `@JvmStatic`로 static native symbol(`Java_mps_jni_MetalMatMulBridge_xxx`)을 노출.
 */
object MetalMatMulBridge {
    @JvmStatic external fun nativeInit(): Boolean

    @JvmStatic external fun nativeMatmul(
        a: FloatArray, b: FloatArray,
        m: Int, k: Int, n: Int,
        c: FloatArray,
    )

    @JvmStatic external fun nativeMatmulBackwardA(
        b: FloatArray, gy: FloatArray,
        m: Int, k: Int, n: Int,
        dA: FloatArray,
    )

    @JvmStatic external fun nativeMatmulBackwardB(
        a: FloatArray, gy: FloatArray,
        m: Int, k: Int, n: Int,
        dB: FloatArray,
    )

    // F: fp16 mixed precision. forward only (backward는 fp32 그대로 — mixed precision pattern).
    // 입력 fp32 → GPU fp16 buffer로 cast 후 MPSMatrixMultiplication<half>, 결과 fp16 → fp32 복귀.
    // 학습 안정성 사용자 수동 검증 필요 — 기본 비활성, [mps.MpsBackend.enableFp16] 호출시 활성.
    @JvmStatic external fun nativeMatmulFp16(
        a: FloatArray, b: FloatArray,
        m: Int, k: Int, n: Int,
        c: FloatArray,
    )
}
