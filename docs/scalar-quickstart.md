# Scalar 백엔드 quickstart

PikoGPT 스칼라 백엔드로 **데이터 준비 → 학습 → 샘플링**까지 한 번 끝까지 돌려보는 가이드.
알파벳 a-z 텍스트로 작은 모델을 학습해 음절·단어 패턴을 보는 데모입니다.

코드 자체를 읽으며 이해하고 싶다면 [educational-walkthrough.md](educational-walkthrough.md)도 함께 보세요.
이 문서는 *손으로 돌려보는* 가이드, walkthrough는 *코드를 읽는* 가이드입니다.

---

## 전제

- JDK 21
- 약 10분의 시간 (학습)
- macOS / Linux (`./gradlew`)

이 가이드의 모든 명령어는 PikoGPT 저장소 루트(`pikogpt/`)에서 실행합니다.

---

## Step 1 — 데이터 준비 (~1초)

```bash
./gradlew runAlphabetPrep
```

**무슨 일이 일어나는가**:
1. `data/alphabet/az.txt` (영문 알파벳 패턴 텍스트, ~6KB)를 읽음
2. `CharBPE`로 char-level 토큰화 (vocab 33: a-z + 공백 + 줄바꿈 + 특수)
3. 90:10으로 train/val 분할
4. `data/alphabet/{train.bin, val.bin, meta.json}`로 저장

**산출물 확인**:
```bash
ls -la data/alphabet/
# az.txt, train.bin, val.bin, meta.json
```

`meta.json`을 열어보면 `indexToString`/`stringToIndex` 매핑과 `vocabularySize`가 들어 있습니다.
학습은 train.bin/val.bin의 정수 토큰 시퀀스만 사용합니다.

**예상 출력 일부**:
```
Total tokens: 6,338
Train tokens: 5,704
Val tokens:   634
Vocab size:   33
```

---

## Step 2 — 학습 (~10분)

```bash
./gradlew runMiniTrainer
```

**모델/학습 설정** (`train/MiniTrainerMain.kt`):
- 모델: `layers=1`, `heads=2`, `embeddingDimension=8`, `blockSize=16` (~수천 파라미터)
- 학습: `maxIters=1000`, `batchSize=16`, `learningRate=1e-3`
- 평가: 매 100 iter (`evalIntervalRatio=0.1`)
- 체크포인트: best 갱신 또는 매 평가마다 (`alwaysSaveCheckpoint=true`)

**진행 로그 읽기**:

```
=== PikoGPT 훈련 시작 ===
베이스라인 손실: 3.4965 (0% 진행률 기준)         ← ln(33), 랜덤 추측 수준의 loss
모델 파라미터 수: NN
스텝 0: 훈련 X.XX (X.X%) ░░░░░ | 검증 X.XX (X.X%) ░░░░░ | ...
반복 50: 손실 X.XX (X.X%), elapsed Ns.
...
체크포인트 저장 완료 (best, loss=2.XXXX): .../v0001/checkpoint.json
스텝 100: 훈련 ... | 검증 ... | ...
```

각 항목 의미:
- **베이스라인 손실** = `ln(vocabSize)`. 모델이 학습 안 된 상태의 cross-entropy. 이보다 낮아져야 학습이 의미 있음.
- **진행률 %** = `(baseline - 현재 loss) / baseline × 100`. 100%면 완벽 예측 (도달 불가).
- **▓░░░░** 진행률 바.
- **grad-norm** = optimizer step 직전 gradient의 L2 norm. 발산하면 너무 큼.

**산출물**:
```
model/
└── alphabet/
    └── main/
        ├── v0001/   ← 첫 best 갱신
        │   ├── checkpoint.json
        │   ├── meta.json
        │   └── model_weights.bin
        ├── v0002/   ← 두 번째 갱신
        ├── v0003/
        └── ...
```

매 best 갱신 또는 평가마다 새 v 디렉토리가 생성됩니다 (turbo 백엔드와 동일 schema).

---

## Step 3 — 샘플링 (~수 초)

```bash
./gradlew runSampler
# → 자동으로 model/alphabet/main 아래 가장 큰 v 번호 디렉토리 사용

# 또는 명시 ckpt:
./gradlew runSampler --args="model/alphabet/main/v0005"
```

`SamplerMain.kt`가 4개 짧은 프롬프트(`"a"`, `"the"`, `"the cat"`, `"abc"`)에 대해 각 2 샘플씩 생성.

**기대 출력 (학습 충분히 한 경우)**:
- 알파벳 문자 시퀀스가 살아 있어야 함 (vocab 밖 문자가 나오지 않음)
- 자주 나오는 단어 단편이 부분적으로 재현 (`the`, `a `, `cat`, `dog` 등)
- 완전한 문장은 어려움 (모델이 너무 작음)

예시 출력 (실제 학습에 따라 다름):
```
=== prompt: 'the' ===
  [1] the cat ran on the green
  [2] the boy
=== prompt: 'a' ===
  [1] a small ...
```

**해석**:
- 학습 직후라면 거의 랜덤. 1000 iter면 단어 시작 패턴 정도 보임.
- 모델이 너무 작아 의미 있는 문장 생성은 어려움. **목표는 파이프라인이 도는 것을 확인하고 chain rule + transformer가 실제로 학습 신호를 만드는 것을 눈으로 보는 것**.

---

## 디버깅 팁

### 학습이 발산 (loss가 NaN/Inf)
- `learningRate`를 절반으로 (예: 1e-3 → 5e-4)
- `gradClip` 그대로 1.0 유지 (이미 최대 norm 1.0으로 클리핑 중)
- ScalarTrainer는 NaN 감지 시 즉시 학습 중단 (`error()`)하므로 console에 메시지 표시됨

### loss가 줄지 않음 (베이스라인에서 정체)
- `maxIters`를 늘려보기 (1000 → 2000)
- `batchSize`를 줄이기 (메모리 여유 시) — 작은 배치로 더 자주 업데이트
- 모델 사이즈 키우기: `embeddingDimension=16`, `numberOfLayers=2`

### OOM (메모리 부족)
- `batchSize`/`blockSize` 줄이기
- `gradle.properties`의 `-Xmx2g` → `-Xmx4g`로

### 샘플링에서 ckpt 못 찾음
```
java.lang.IllegalArgumentException: model/alphabet/main 디렉토리가 없습니다.
```
- `./gradlew runMiniTrainer`를 먼저 끝까지 돌리세요. v0001 이상이 만들어져야 합니다.
- 또는 다른 ckpt 경로를 명시: `--args="<your-ckpt-dir>"`

### 생성된 텍스트가 이상함 (반복, 의미 없음)
- 학습이 충분치 않음. `maxIters` 늘리기.
- `samplingTemperature` 낮추기 (0.8 → 0.5) — 결정론적 샘플링
- `repetitionPenalty` 1.1~1.3 적용 (`SampleConfig`에서)

---

## 다음 단계

- **코드 읽기**: [educational-walkthrough.md](educational-walkthrough.md) — `Value.kt`부터 `ScalarTrainer.kt`까지 10단계 가이드.
- **더 큰 모델/긴 학습**: `MiniTrainerMain`의 설정을 늘려보기.
   - `embeddingDimension=16`, `numberOfLayers=2`로 키우면 학습 시간 ~3-5×.
   - `blockSize=32`로 늘리면 더 긴 컨텍스트 학습.
- **실전 학습 (turbo 백엔드)**: 스칼라는 ~1000× 느립니다. 의미 있는 실험은 turbo로:
   - `./gradlew runTinyHelenTrainTurbo` — TinyHelen leaner 코퍼스로 1M 모델 학습.
   - `./gradlew runChatTurbo` — 학습된 ckpt와 인터랙티브 대화.
- **resume 학습**: ckpt에서 이어 학습하기 — `TrainConfig`의 `initFrom = "resume"` + `modelCheckpointDir`.

---

## 참고: 스칼라 vs Turbo

| | Scalar | Turbo |
|---|---|---|
| 데이터 단위 | `Value` 객체 (스칼라 1개당 1개) | `TurboTensor` (FloatArray) |
| autograd | 동적 그래프 + topo sort | 명시적 layer-별 backward |
| 속도 | 71K param iter당 ~16초 | 1M param iter당 ~2.6초 |
| modern features | LayerNorm + GELU | + RMSNorm, SwiGLU, RoPE, GQA, qk-norm, KV cache |
| 목적 | **autodiff + transformer 수식 학습용** | 실전 성능 |

스칼라는 chain rule이 코드에 그대로 보이는 reference. 한 번 돌려보고 코드를 읽어보면, turbo 백엔드의 SIMD 코드를 읽을 때도 같은 식이 그 안에 있다는 것을 알아볼 수 있습니다.
