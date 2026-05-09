# Turbo 백엔드 다음 개선 계획

현재 baseline: **vec 대비 3.7× (1M)·3.2× (5M)** 가속 (worker=cpuCount×2/3).
대부분 영역이 SIMD 적용 완료 또는 ROI 검토 후 폐기 결정. 남은 작은 항목만 정리.

## 완료 (참고)

git history 상 이미 적용된 최적화:
- `turboMatmul` forward + 양 backward SIMD (Phase 2)
- fused `TurboAdamW` SIMD step
- `TurboKVCache` + sampler `forwardIncremental` (추론 5~10×)
- `TurboTransformerBlock.useGradientCheckpointing` 토글
- `turboForkJoinAll` / `turboForkJoinIndices` (coroutines 의존성 제거)
- `TurboLinear.backward` weight grad SIMD
- **Attention forward/backward SIMD化** (commit `d31ede5`) — Q·K^T dot, attn·V scaled add, dV/dQ/dK 모두
- **LayerNorm / RMSNorm SIMD化** (commit `3994a67`)
- **Worker grad merge + CrossEntropy SIMD化** (commit `57e16a0`)

## 남은 작업

### Pointwise SIMD化 (작은 ROI)
- **Softmax** (`ops/TurboSoftmax.kt`) — max reduction + exp + invSum mul
- **GELU** (`ops/TurboGELU.kt`) — `0.5 * x * (1 + tanh(...))`
- **SiLU** (`ops/TurboSiLU.kt`) — `x * sigmoid(x)`

`exp/tanh/sigmoid`는 Vector API가 직접 지원 안 해서 polynomial approx 또는 scalar fallback 필요. **예상 효과 +5~15%**, 작업량 2~3일.

### JIT / 마이크로 튜닝
- `TurboNorm` sealed interface call site monomorphic 검증
- `TurboSelfAttention` 내부 dropout 분기 inline
- primitive type 통일 (Float boxing 회피)
- `-XX:+UseG1GC`, `-Djava.util.concurrent.ForkJoinPool.common.parallelism=N` 명시

**예상 효과**: +5~10%, 작업량 1일.

## 폐기 결정 (진행 안 함)

ROI/복잡도 검토 결과:

| 항목 | 사유 |
|---|---|
| **bf16 weight storage + AMP trainer** | CPU + JDK Vector API 환경에서 native bf16 SIMD 미지원. unpack overhead로 속도 이득 모호하고, 1~10M 모델에선 메모리 압박 없음 |
| **model 복제 폐기 (cache 외부화)** | 모든 layer/op 시그니처 변경 필요한 매우 큰 리팩토링. 현재 모델 사이즈에서 메모리 부담 없음 |
| **Attention head별 batched matmul / 추가 inner loop 최적화** | forward/backward 모두 이미 SIMD化 완료, 추가 이득 작음 |
| **Embedding scatter 최적화** | vocab 작아 효과 미미 |
| **Flash Attention v2** | 현재 컨텍스트 길이 (T<128)에서 효과 미미 |
| **GPU 가속 (Vulkan/Metal/TornadoVM)** | Apple Silicon에서 학습 곡선 큼, ROI 낮음 |
| **추가 알고리즘 옵션** | RMSNorm/GQA/qk-norm/fused QKV/z-loss 이미 모두 보유 |

## 검증 게이트

1. 기존 turbo 테스트 100% 통과 (동등성)
2. 1M (stage2) + 5M (bench5m) 학습 elapsed 비교 — 회귀 없음
3. final loss vec과 ±0.05 이내

## 측정 명령

```bash
# 마이크로벤치
./gradlew runTurboBench

# 1M 학습 비교 (3000 iter, ~10분)
# pretrain ckpt 경로는 사용자 환경에 맞게 (예: 새 경로 형식 model/stage1/main/v0001).
./gradlew runCcmcV2ProStage2TrainTurbo --args="model/stage1/main/v0001"

# 5M 학습 비교 (500 iter, ~5분)
./gradlew runBench5MTurbo
```

## 현재 baseline

| 모델 | iter | vec | turbo | speedup |
|---|--:|--:|--:|--:|
| 1M | 3000 | 36m 10s | 9m 45s | **3.73×** |
| 5M | 500 | 14m 11s | 4m 26s | **3.20×** |
| 10M | 250 | 16m 18s | 4m 19s | **3.78×** |
