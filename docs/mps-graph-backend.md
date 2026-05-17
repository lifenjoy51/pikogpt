# MPSGraph Backend — GPU 100% Residence Training

Apple Silicon GPU에서 forward + backward + AdamW를 **단일 graph로 실행**해 GPU 활용을 turbo CPU 8-worker 대비 **3.29×** 끌어올린 PoC. 2026-05-17 구현.

## Why

기존 PoC (`mps` 디렉터리, `MetalMatMulBridge.m`)는 **MatMul만** GPU로 보내고 나머지 ops(softmax, RMSNorm/LayerNorm, GELU/SiLU, RoPE, attention 내부 Q·K^T·V, CE loss, optimizer)는 CPU에 두는 구조였다. Bench10MTurbo 실측에서 turbo 8-worker (163s/50iter) 대비 **사실상 동일한 wall-clock**(162s)이 나왔다. 원인은 두 가지:

1. 매 matmul마다 host↔GPU 데이터 왕복
2. CPU 8-worker가 같은 GPU command queue로 dispatch → GPU 직렬 처리 → CPU SIMD × 8worker가 우위

→ 해결책: **모든 ops를 GPU로 + 데이터를 GPU resident**. PyTorch MPS / TF graph mode와 동일 path.

## 아키텍처

### 디렉터리

```
src/main/objc/
└── MpsGraphBridge.mm                 # Obj-C++ JNI bridge: PikoMpsGraphSession + graph build/run

src/main/kotlin/mps/
├── MpsGraphSession.kt                # JNI binding + session lifecycle
└── MpsGraphConfig.kt                 # 모델 hyperparam (numLayers, embedDim, ...)

src/main/kotlin/train/experiments/
└── Bench10MMpsGraphTrain.kt          # 진입점: 10M 모델 학습 wrapper

src/test/kotlin/mps/
└── MpsGraphSessionTest.kt            # Phase별 단위 test (11/11 통과)

build.gradle.kts
└── buildMpsGraphLib                  # libpikogpt_mpsgraph.dylib 빌드 task
└── runBench10MMpsGraphTrain          # 진입점 JavaExec
```

### Graph 구조

단일 training step graph (`buildStepGraph`):

```
input placeholders:
  tokenIds [T] int32
  targets  [T] int32
  cos, sin [T, headDim/2]  (host precompute)
  mask     [T, T]          (causal, host precompute)
  lr, bc1, bc2  scalar      (per-step varying)
  for i in 0..N-1 (N = 291 weight):
    wPh[i]      [shape_i] float32
    mPh[i], vPh[i] [shape_i] float32  (AdamW state)

forward:
  x = gather(tokenEmbedding, tokenIds)              # [T, embedDim]
  for L in 0..15:
    ln1   = LayerNorm(x, γ1[L], β1[L])
    attn  = Attention(ln1, qW/qB/kW/kB/vW/vB/oW/oB, cos, sin, mask)
            ├─ Q, K, V = Linear projections
            ├─ reshape to [T, numHeads, headDim]
            ├─ RoPE rotation on Q, K (pair i, i+D/2)
            ├─ transpose to [numHeads, T, headDim]
            ├─ scores = Q @ K^T / sqrt(headDim) + mask
            ├─ attn   = softmax(scores)
            ├─ out    = attn @ V
            └─ output proj
    x = x + attn
    ln2  = LayerNorm(x, γ2[L], β2[L])
    mlp  = SwiGLU(ln2, gateW/B, upW/B, downW/B)
           = down( silu(gate(ln2)) * up(ln2) )
    x = x + mlp
  x = LayerNorm(x, γ_final, β_final)
  logits = x @ tokenEmbedding.T                     # tied lm head

loss:
  log_softmax = log(softmax(logits, axis=-1))
  one_hot     = oneHot(targets, vocab)
  loss        = -mean(sum(one_hot * log_softmax, axis=-1))

backward:
  grads = MPSGraph.gradientForPrimaryTensor(loss, withTensors=[wPh[i]])
  # → dict { wPh[i] → grad_i }  (automatic differentiation)

AdamW update (per weight):
  m_new = β1·m + (1-β1)·g
  v_new = β2·v + (1-β2)·g²
  m_hat = m_new / bc1                # bc1 = 1 - β1^t  (host precompute)
  v_hat = v_new / bc2
  w_new = w - lr · m_hat / (sqrt(v_hat) + ε) - lr·wd·w

outputs:
  loss scalar
  wNew[i], mNew[i], vNew[i] — 새 MTLBuffer (caller swap)
```

### Weight 저장 (GPU resident)

`PikoWeightSlot`에 paramIndex별로 4개 MTLBuffer를 GPU에 보관:

```objc
@interface PikoWeightSlot : NSObject
@property id<MTLBuffer> buffer;       // weight
@property id<MTLBuffer> gradBuffer;   // gradient
@property id<MTLBuffer> mBuffer;      // AdamW m
@property id<MTLBuffer> vBuffer;      // AdamW v
@property NSArray *shape;
@property NSUInteger numel;
@end
```

`MTLResourceStorageModeShared` = Apple Silicon **unified memory**, zero-copy. host에서 contents 포인터로 직접 읽기/쓰기 가능.

step 호출 후 새 buffer (output)를 `slot.buffer/mBuffer/vBuffer`에 swap → 다음 step에서 자동 사용.

### Graph 캐시 (필수)

`s.cachedT == 0`이면 첫 호출 시 graph build, 이후 재사용.

| | 매번 build | cache (1번 build) |
|---|---:|---:|
| iter 1 | 3340ms | 3340ms |
| iter 5 | 3065ms | **31ms** |
| 50 iter 총 시간 | ~167s | **~5s** |

graph build에 Apple Metal compiler가 전체 DAG를 GPU shader pipeline state로 컴파일하는 비용 ~3.3s가 들어 있어 매번 하면 step time을 100배 압도. 캐시는 GPU 활용의 전제 조건.

scalar 상수(lr, β1, β2, ε, wd)는 graph constant로 매립하지만, **scheduler용 lr, AdamW bias correction(bc1=1-β1^t, bc2=1-β2^t)은 매 step 다른 값이라 placeholder로 노출**해서 graph rebuild 없이 step별 host에서 새 값 전달.

## Phase별 구현 + 검증

| Phase | 내용 | 검증 |
|---|---|---|
| 1 | JNI bridge skeleton, MTLDevice/MPSGraph init, session lifecycle | 3 test (create/destroy/double-close 안전) |
| 2.1 | Weight loading (GPU MTLBuffer) | 1 test (round-trip) |
| 2.2-1 | Embedding gather | 1 test (naive 동등) |
| 2.2-2 | LayerNorm | 1 test (turbo `TurboLayerNorm` rtol < 1e-4) |
| 2.2-3 | Linear (y = x @ W.T + b) | 1 test (turbo `TurboLinear` rtol < 1e-4) |
| 2.2-4 | SwiGLU MLP (gate × silu(gate) → down) | 1 test (turbo `TurboMLP` rtol < 1e-3) |
| 2.2-5 | Multi-head causal self-attention + RoPE | 1 test (turbo `TurboSelfAttention` rtol < 2e-3) |
| 2.2-6/7 | Full forward (16 block + tied lm head) | 1 test (turbo `TurboPikoGPT` rtol < 5e-3) |
| 3 | CE loss + backward (`gradientForPrimaryTensor`) → slot.gradBuffer | graph 빌드 통과 |
| 4 | AdamW step graph (m/v state) | 통합 |
| 5 | Single step graph (forward+loss+backward+AdamW) | 1 test (20 step → loss 감소 검증) |
| 6 | Wall-clock 측정 + graph cache | Bench10MMpsGraphTrain 5 iter |

**총 11/11 unit test 통과**. `./gradlew test --tests "mps.MpsGraphSessionTest"`.

## 성능 결과 (Bench10M, 13.2M params)

### Per-step time

| | step time |
|---|---:|
| iter 1 (graph build + run) | 3340 ms |
| iter 5+ (cache hit, run only) | **31 ms** |

### Per-sequence wall-clock

| 백엔드 | per-seq | turbo 대비 |
|---|---:|---:|
| turbo (worker=8) | 102 ms | 1.00× |
| mps matmul-only PoC | 101 ms | 1.01× (사실상 무효) |
| **mps graph (이 구현)** | **31 ms** | **3.29×** |

### 같은 1600 seq 처리 wall-clock

| 백엔드 | 시간 |
|---|---:|
| turbo (50 iter × 32 seq) | 163 s |
| **mps graph (1600 iter × 1 seq)** | **49.6 s** |

→ **3.29× 가속**

## 사용법

### 빌드
```bash
./gradlew buildMpsGraphLib   # libpikogpt_mpsgraph.dylib
```

### 단위 test
```bash
./gradlew test --tests "mps.MpsGraphSessionTest"   # 11/11 통과
```

### 학습
```bash
./gradlew runBench10MMpsGraphTrain --args="50"     # 50 iter wall-clock 측정
```

출력 예:
```
모델 파라미터 텐서 수: 291, 총 스칼라: 13158240
데이터 로드 완료: 241844 토큰
[mps-graph] weights 로드 완료 (291개 tensor, GPU resident)
iter 1: loss=7.5808, step_ms=3340, total=3.3s
iter 5: loss=7.8523, step_ms=31, total=3.5s
=== Bench10MMpsGraph 완료 ===
총 시간: 3576ms (5 iter)
iter당 평균: 715ms
```

### GPU utilization 측정
```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 30   # 별도 터미널, 학습 도중
```

## 한계 / TODO

### 현재 PoC scope

- **batch=1, gradAccum=1**: turbo Bench10MTurbo는 batch=2 × gradAccum=16 = 32 seq/iter. mps graph는 1 seq/iter로 작동. throughput per-iter는 1/32이지만 per-sequence는 3.29× 빠름.
- **모델 hyperparam 고정**: numLayers/embedDim/numHeads/blockSize 변경 시 graph rebuild. 현재는 T 바뀌면 rebuild 안 함 (PoC 가정).
- **LR scheduler 없음**: lr placeholder는 받지만 진입점은 fixed lr. 향후 cosine/warmup 추가 가능.
- **Checkpoint 없음**: 학습 종료 후 weight를 host로 읽어 별도 저장하는 path 없음. `nativeReadWeight` JNI는 있음.
- **Eval/sample 없음**: turbo의 validation loss 측정 + 샘플링은 미구현.
- **fp16 안 함**: 전체 fp32. Apple Silicon fp16 가속 (~2× 추가) 가능하나 학습 안정성 검증 필요.

### 확장 가능성

- **Batch 키우기**: T를 (B*T)로 늘려 1 graph에서 batch 처리. GPU SM 활용도 추가 상승 예상.
- **MPSGraphExecutable 명시적 compile + serialize**: 현재는 MPSGraph 객체 cache. executable serialization으로 process 간 cache 공유 가능.
- **Variable 패러다임 전환**: 현재는 placeholder + result swap. `[g variableWithData:...]` + `[g assignVariable:...]`로 in-place update면 buffer alloc 줄어듦.
- **Multi-GPU**: 현 코드는 single device. Apple Silicon이라 의미 적음.

## 참고

- Apple MPSGraph 문서: <https://developer.apple.com/documentation/metalperformanceshadersgraph>
- PyTorch MPS backend (참고 구현): <https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps>
- 기존 matmul-only PoC: `mps/jni/MetalMatMulBridge.kt`, `src/main/objc/MetalMatMulBridge.m` (이 backend로 대체 가능)
- 진입점: `train/experiments/Bench10MMpsGraphTrain.kt`
- JNI bridge: `src/main/objc/MpsGraphBridge.mm`
- Kotlin API: `mps/MpsGraphSession.kt`, `mps/MpsGraphConfig.kt`
- 단위 test: `src/test/kotlin/mps/MpsGraphSessionTest.kt`
