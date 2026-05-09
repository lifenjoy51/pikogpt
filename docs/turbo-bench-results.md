# Turbo 백엔드 벤치마크 결과 (2026-05-07)

> 역사 기록: vec 백엔드 폐기 직전 마지막 vec ↔ turbo 비교 측정. 이후 vec은 삭제됨.

JDK 21 Vector API 기반 turbo 백엔드의 vec 백엔드 대비 학습 속도 비교 측정. 12-core
Apple Silicon, NEON SIMD lane 4.

## 결론 요약

| 모델 | 데이터 | iter | vec | turbo | speedup | 비고 |
|---|---|---:|---:|---:|---:|---|
| **stage2 1M** | ccmc-v2-pro/stage2 | 3000 | 36m 10s | **9m 45s** | **3.73×** | finetune from stage1 ckpt |
| **bench 5M** | ccmc-v2-pro/stage2 | 500 | 14m 11s | **4m 26s** | **3.20×** | scratch |
| **bench 10M** | ccmc-v2-pro/stage2 | 250 | 16m 18s | **4m 19s** | **3.78×** | scratch (16L × 256 × 8h) |

50 iter warmup 시점 가속비:
- 1M: 3.61×, 5M: 3.62×, 10M: **4.32×** — 큰 모델일수록 SIMD 효과 분명

두 모델 모두 turbo가 **3.2~3.7× 가속**. final loss는 vec와 거의 동등 (random init 차이
범위, 알고리즘 동등성은 동등성 테스트에서 검증됨).

## Phase 별 적용된 최적화

### Phase 0~1: 골격 + 알고리즘 옵션
- vec와 알고리즘 동등 (LayerNorm/GELU/SwiGLU/RoPE/tied 등)
- 추가 옵션: RMSNorm / GQA / qk-norm / fused QKV / z-loss

### Phase 2: SIMD MatMul + fused AdamW
- `turboMatmul` forward + 양 backward를 Java Vector API (`FloatVector.SPECIES_PREFERRED`)로 SIMD 가속
- `TurboAdamW` step을 단일 SIMD 패스 (m/v/decay/p 한 lane register pass)
- 마이크로벤치: vec 대비 3.6~4.4× (큰 사이즈), 작은 사이즈에서도 2.5~4.4×

### Phase 3.0: KV cache + sampler 통합
- `TurboKVCache` per-layer ring buffer
- `forwardIncremental` — 토큰당 비용 O(t)→O(1)
- 추론 5~10× 가속 (단순 모드: numKvHeads=numHeads, !useFusedQkv, !useQkNorm)

### Phase 4.1
- `TurboTransformerBlock.useGradientCheckpointing` 토글 (dropout=0 require)

### Phase 5.0/5.1
- `turboForkJoinAll` / `turboForkJoinIndices` helper
- `trainStepParallel` / `evaluateBatchesParallel`을 ForkJoinPool로 전환 (coroutines 의존성 제거)

## Worker 수 튜닝 — 결정적 발견

초기 default `worker=4`로 측정 시 **1M 모델에서 turbo가 vec보다 9% 더 느렸음** (39m 22s
vs 36m 10s). 원인 분석 결과:

- vec의 coroutine `Dispatchers.Default`는 `ForkJoinPool.commonPool()` 기반 work-stealing →
  4 launch지만 사실상 더 많은 thread 활용 (CPU 958%, ~9.6 core)
- turbo의 `turboForkJoinAll`은 정확히 4 task 점유 → CPU 392%, ~3.9 core만 사용
- 12-core 머신에서 worker=4는 자원 미활용

`defaultCap`을 **`cpuCount - 1`**로 변경 후:
- worker=11 사용
- 1M 모델: 39m 22s → **9m 45s (3.73× 가속)**
- 5M 모델: (worker=4)로 측정 시 ~6m → (worker=11) **4m 26s (3.20×)**

## 학습 hot path 분석

마이크로벤치 SIMD 4× 가속이 학습 시간에 직접 반영되지 않은 이유:

| 코드 경로 | SIMD 적용 | 비중 |
|---|---|---|
| `turboMatmul` forward (linear projection) | ✅ | 작음 (작은 shape) |
| `turboMatmulBackward` (linear input grad) | ✅ | 작음 |
| `TurboLinear.backward` weight grad | ✅ (Phase 5+ 적용) | **큼** |
| Attention nested loop (Q·K^T, attn·V) | ❌ vec과 동일 | **큼** |
| AdamW step | ✅ fused | 작음 |
| Worker grad accumulation | ❌ nested for | 중간 |
| Embedding scatter / dropout / LayerNorm | ❌ | 중간 |

`TurboLinear.backward` weight grad SIMD化 (n outer × o middle × i inner SIMD) 적용 후
worker=11과 결합해 3.7× 가속 달성.

## 모델 사이즈별 결과 표

### stage2 (1M params: 8 layer × 96 emb × 3 head, block 32)

| iter | vec (w4) elapsed | turbo (w4) elapsed | turbo (w11) elapsed | speedup vs vec |
|---:|---:|---:|---:|---:|
| 100 | 83s | 82s | 23s | 3.61× |
| 500 | 392s | (중단) | 104s | 3.77× |
| 1000 | 760s | (중단) | 203s | 3.74× |
| 1500 | 1115s | (중단) | 294s | 3.79× |
| 2000 | 1458s | (중단) | 385s | 3.79× |
| 2500 | 1812s | (중단) | 484s | 3.74× |
| 3000 | 2167s | 2362s (느림) | **581s** | **3.73×** |

### bench5m (5.7M params: 12 layer × 192 emb × 6 head, block 32)

| iter | vec (w4) elapsed | turbo (w11) elapsed | speedup |
|---:|---:|---:|---:|
| 50 | 76s | 21s | 3.62× |
| 100 | 174s | 57s | 3.05× |
| 200 | 347s | 112s | 3.10× |
| 300 | 523s | 167s | 3.13× |
| 400 | 699s | 221s | 3.16× |
| 500 | 851s | **266s** | **3.20×** |

## 자원 사용률

worker=11 활성 시:
- turbo: CPU ~900~1000% (9-10 core 사용)
- load average: ~10
- 12-core 머신을 거의 풀 활용

## 폐기 항목

ROI/복잡도 검토 결과 진행하지 않기로 결정:
- **bf16 dtype + AMP trainer** — CPU + JDK Vector API 환경에서 native bf16 SIMD 미지원. unpack overhead로 속도 이득 모호하고, 1~10M 모델에선 메모리 압박도 없음
- **model 복제 폐기 (cache 외부화)** — 모든 layer/op 시그니처 변경 필요한 매우 큰 리팩토링. 현재 모델 사이즈에서 메모리 부담 없음
- **Flash Attention v2** — 현재 컨텍스트 길이(T<128)에서 효과 미미
- **Embedding scatter 최적화** — vocab 작아 효과 미미
- **Attention head별 batched matmul / 추가 inner loop 최적화** — 이미 forward/backward 모두 SIMD化 완료, 추가 이득 작음

## 검증

모든 sub-phase에서 동등성 회귀 테스트 통과:
- `TurboMatMulEquivalenceTest`, `TurboOpsEquivalenceTest`, `TurboFullPipelineTest`
- `TurboRMSNormTest`, `TurboZLossTest`, `TurboQkNormTest`, `TurboFusedQkvTest`, `TurboGqaTest`
- `TurboAdamWEquivalenceTest`, `TurboKVCacheTest`

마이크로벤치: `./gradlew runTurboBench` (Apple Silicon NEON lane 4 결과)

```
shape                    turbo (us)     vec (us)    speedup
[1, 768, 768]                  52         188        3.62x
[64, 768, 768]               2842       12122        4.27x
[64, 768, 3072]             11379       46950        4.13x
[64, 3072, 768]             11599       47589        4.10x
[128, 128, 128]               161         667        4.15x
[256, 256, 256]              1344        5243        3.90x
[32, 96, 96]                   27         100        3.69x   (stage2 QKV)
[32, 96, 256]                  58         254        4.36x   (SwiGLU expand)
[32, 256, 96]                  68         264        3.88x   (SwiGLU contract)
[32, 96, 2000]                476        1941        4.08x   (tied lm_head)
[32, 32, 32]                    5          13        2.51x   (attn head Q·K^T)
```

## 사용 가이드

```bash
# 기본 학습 (worker=cpuCount-1 자동)
./gradlew runCcmcV2ProStage2TrainTurbo --args="model/stage1/vec/1087936/v0053"
./gradlew runBench5MTurbo

# worker 수 override
TURBO_MAX_WORKERS=8 ./gradlew runBench5MTurbo

# 마이크로벤치
./gradlew runTurboBench
```
