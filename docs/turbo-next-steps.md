# Turbo 백엔드 다음 개선 계획

현재 baseline: **vec 대비 3.7× (1M)·3.2× (5M)** 가속 (worker=cpuCount-1). 추가 가속 가능
영역을 ROI 순으로 정리.

## 우선순위

| # | 항목 | 예상 가속 | 작업량 | 위험 | 단독 가치 |
|--:|---|--:|--:|--:|--:|
| 1 | **Attention SIMD化 (Q·K^T, attn·V)** | +30~60% | 중 | 낮 | ★★★ |
| 2 | **Pointwise ops SIMD化 (LN/RMS/Softmax/GELU/SiLU)** | +10~20% | 중 | 낮 | ★★ |
| 3 | **Worker grad merge SIMD化** | +5~10% | 작 | 낮 | ★★ |
| 4 | **Embedding scatter 최적화 (token grad)** | +5% | 작 | 낮 | ★ |
| 5 | **bf16 weight storage + forward unpack** | 메모리 -25% | 중 | 중 | ★★ |
| 6 | **model 복제 폐기 + thread-local pool** | scaling +20%, 메모리 -75% | 매우 큼 | 중 | ★★★ |
| 7 | **Attention head별 batched matmul (turboMatmul 재사용)** | +20~40% | 중 | 낮 | ★★★ (1과 결합) |
| 8 | **JIT 친화적 hot loop (monomorphic, primitive)** | +5~10% | 작 | 낮 | ★ |

총 예상 누적 효과 (Phase A+B+C): **5M 모델 vec 대비 ~5~6× 가속, 1M ~5×**

---

## Phase A — Attention 최적화 (가장 큰 임팩트)

### A1. Attention head별 batched matmul (`#7`)
- 현재: head 안에서 nested for loop (`for j in 0..i: for d in 0..headDim: ...`)
- 변경: head별로 Q[T, headDim] @ K[T, headDim]^T → 작은 turboMatmul 호출 + causal mask
- score는 `[T, T]` matrix 한 번에 SIMD 가속
- output `attn @ V`도 동일 패턴
- **위치**: `TurboSelfAttention.forward` / `backward` 핵심 nested loop
- **검증**: 기존 동등성 테스트 통과 + 마이크로벤치 attention shape

### A2. Inline SIMD score/output (`#1`)
- A1 적용 후 잔여 — head dim 작은 경우 (headDim 32) turboMatmul overhead 클 수 있음
- 직접 inner loop SIMD: Q row · K row dot product → `TurboSimdMath.dot` 사용
- A1과 비교 측정 후 더 빠른 쪽 채택

**예상 효과**: 1M/5M 모두 +30~60% (3.7× → 5×, 3.2× → 4.5×)
**작업량**: 1-2일

---

## Phase B — Pointwise ops + 보조 핫패스 (`#2~4`)

### B1. RMSNorm/LayerNorm SIMD化
- 행별 reduction (sum, sumSq) — SIMD horizontal sum (`reduceLanes`)
- mean/var 계산 + 정규화 inner loop SIMD
- backward의 `meanDxHat`, `meanDxHatXHat` reduce도 SIMD

### B2. Softmax SIMD化
- max reduction + exp (lanewise EXP 또는 polynomial approx) + invSum scaled mul
- backward Jacobian의 dot(softmax, gy) reduce SIMD

### B3. GELU/SiLU SIMD化
- tanh/sigmoid는 Vector API에서 직접 지원 안 함 → polynomial approx 또는 scalar fallback
- 효과 작지만 hot loop이라 측정 가치

### B4. Worker grad merge SIMD化
- `turboAccumulateGrads`의 `tg[j] += sg[j]` 단순 SIMD scaled add (이미 helper 있음)

### B5. Embedding scatter
- random access지만 같은 token id 묶어 처리 가능 (sort + group)
- 작은 vocab은 효과 미미

**예상 효과**: +10~20% (5× → 5.5~6×)
**작업량**: 2-3일

---

## Phase C — bf16 + scaling (`#5, #6`)

### C1. bf16 weight storage (`#5`)
- 이미 `TurboBf16` 변환 helper 작성됨
- master weight fp32 + forward 직전 bf16 unpack → weight 저장 메모리 50%
- 단 unpack overhead로 속도 가치 미미. **메모리 가치만**
- 큰 모델 (50M+) 도입 시 활성화

### C2. model 복제 폐기 (`#6`)
- 현재 worker 11 = model 11 인스턴스 (1M × 11 = 11M params 메모리, 5M × 11 = 55M)
- 단일 model + thread-local activation cache + thread-local grad delta
- 모든 forward/backward의 cache 외부화 (큰 리팩토링)
- **메모리 -75%**, scaling overhead 제거 → grad merge 비용 감소
- **작업량**: 매우 큼 (모든 layer/op 시그니처 변경)

**예상 효과**: scaling +20~30%, 메모리 큰 절감
**작업량**: 5-7일 (Phase C2)

---

## Phase D — JIT/마이크로 (`#8`)

### D1. Hot loop 점검
- `TurboNorm` sealed interface call site monomorphic 검증
- `TurboSelfAttention` 내부 dropout 분기 inline
- primitive type 통일 (Float boxing 회피)

### D2. JIT compile 옵션 튜닝
- `-XX:+UseG1GC`, `-XX:+UnlockDiagnosticVMOptions -XX:+PrintInlining` 분석
- ForkJoinPool size 명시 (`-Djava.util.concurrent.ForkJoinPool.common.parallelism=11`)

**예상 효과**: +5~10%
**작업량**: 1일

---

## 진행 순서 (권장)

```
Phase A (1-2일) → 측정 → Phase B (2-3일) → 측정
                                 ↓
                  큰 모델 도입 시 → Phase C (5-7일)
                                 ↓
                            Phase D 마무리
```

**다음 즉시 진행**: **Phase A (attention 최적화)** — 가장 큰 single 임팩트, 작업량 적당, 위험 낮음.

## 진행 안 할 것 (이전 검토 결과)

- ❌ Flash Attention v2 — 1M~5M 모델은 T 작아 효과 미미
- ❌ AMP trainer (4.2/4.3) — 작은 모델 메모리 충분
- ❌ GPU 가속 (Vulkan/Metal/TornadoVM) — Apple Silicon에서 학습 곡선 큼, ROI 낮음
- ❌ 추가 알고리즘 옵션 — 이미 RMSNorm/GQA/qk-norm/fused QKV/z-loss 다 있음

## 검증 게이트 (각 Phase)

1. 기존 turbo 테스트 100% 통과 (동등성)
2. 새 마이크로벤치 측정 (각 op 단위)
3. 1M (stage2) + 5M (bench5m) 학습 elapsed 비교 — 회귀 없음
4. final loss vec과 ±0.05 이내

## 측정 명령

```bash
# 마이크로벤치
./gradlew runTurboBench

# 1M 학습 비교 (3000 iter, ~10분)
./gradlew runCcmcV2ProStage2TrainTurbo --args="model/stage1/vec/1087936/v0053"

# 5M 학습 비교 (500 iter, ~5분)
./gradlew runBench5MTurbo
```

## 현재까지 누적 결과 (baseline)

| 모델 | iter | vec | turbo (현재) | speedup |
|---|--:|--:|--:|--:|
| 1M | 3000 | 36m 10s | 9m 45s | **3.73×** |
| 5M | 500 | 14m 11s | 4m 26s | **3.20×** |
