# TinyHelen 학습 테스트 런북

`TinyHelen` 데이터셋으로 PikoGPT를 end-to-end 학습·샘플링한 실험 기록. 재현 가능한 순서와 실제로 관측된 결과를 함께 남긴다.

## 목적

- 스칼라 autodiff 기반 PikoGPT가 실제 자연어 코퍼스에서 학습 신호를 받는지 확인.
- overnight 규모 (≈10시간) 안에 끝나는 설정의 현실적 한계 측정.
- 이후 최적화(벡터화, LR 스케줄 개선 등)의 **baseline**.

## 데이터 출처

- Repo: https://github.com/EmpathYang/TinyHelen
- 사용한 서브셋: `data/leaner/10M/train/` (simplification 거친 "leaner" 버전)
- 선택한 파일 (narrative/natural-language 편향):
  - `book0000.jsonl` (≈1.4 MB)
  - `textbook0000.jsonl` (≈3.1 MB)
  - `wiki0000.jsonl` (≈1.0 MB)
  - `conversation0000.jsonl` (≈0.78 MB)
- 제외: `code`, `math`, `web` — 노이즈/syntax-heavy. tiny LM이 배우기 어렵고 신호-노이즈비가 나쁨.
- validation/test 폴더는 사용하지 않음. `StoriesBPEPrep`가 한 파일에서 90:10 자동 분할.

## 재현 절차

### 1. TinyHelen 클론 + 텍스트 추출

```bash
git clone --depth 1 https://github.com/EmpathYang/TinyHelen.git /tmp/TinyHelen
mkdir -p data/tinyhelen
for f in book textbook wiki conversation; do
  # 각 jsonl line(= 한 문서)을 <|bos|> … <|eos|>로 감싸 경계 정보를 보존.
  jq -r '"<|bos|>" + .text + "<|eos|>"' "/tmp/TinyHelen/data/leaner/10M/train/${f}0000.jsonl"
done > data/tinyhelen/stories.txt
```

결과: `data/tinyhelen/stories.txt` 약 5.1 MB, 45,753 줄.

### 2. BPE 전처리

`StoriesBpePrep.main`은 경로를 CLI 인자로 받는다 (기본값 `data/simple`).

```bash
./gradlew runStoriesBpe --args="data/tinyhelen"
```

산출물 (`data/tinyhelen/` 아래):
- `train.bin`, `val.bin` — big-endian 4-byte int 토큰 시퀀스
- `meta.json` — vocab + stoi/itos
- `unique_words.txt` — 진단용

규모: `maxVocabSize = 1000`, 총 1,834,850 토큰 (train 1,651,365 / val 183,485). 전처리 자체는 ≈5분.

### 3. 학습

엔트리: `src/main/kotlin/train/TinyHelenTrain.kt`

```bash
./gradlew runTinyHelenTrain
```

전체 설정 (코드에서 인용):

```kotlin
TrainConfig(
    dataPath = "data/tinyhelen",
    modelDir = "model",
    gradientAccumulationSteps = 4,   // effective batch = 8
    batchSize = 2,
    blockSize = 48,
    numberOfLayers = 2,
    numberOfHeads = 3,               // 24/3 = 8 dim/head
    embeddingDimension = 24,
    dropout = 0.1f,
    bias = true,
    learningRate = 3e-4f,
    weightDecay = 0.05f,
    gradClip = 1.0f,
    beta1 = 0.9f,
    beta2 = 0.95f,
    maxIters = 1500,
    warmupRatio = 0.03f,             // 45 iter warmup
    learningRateDecayRatio = 1.0f,   // cosine decay 전구간
    minimumLearningRate = 3e-5f,
    evalIntervalRatio = 0.05f,       // eval 매 75 iter
    evalIters = 4,                   // no-grad 도입 후 복원 (1 → 4, val loss 노이즈 감소)
    logInterval = 10,
    alwaysSaveCheckpoint = true,
)
```

> **중요**: `evalIters=1`은 이전 attempt에서 OOM을 피하려고 강제로 내렸던 값. 이후 `GradContext.noGrad`
> (아래 "no-grad 도입 기록" 참조)를 도입해 eval 중 그래프 구축을 생략하므로 4로 복원해도 안전하고,
> val loss 추정이 4배치 평균으로 훨씬 덜 흔들린다.

모델 파라미터: **71,256** (embed 25,536 + 2×block 14,448 + final LN 48 + lm_head 24,000).

### 4. 샘플링

엔트리: `src/main/kotlin/sample/TinyHelenSample.kt` — 최근 수정된 체크포인트를 자동으로 고름.

```bash
./gradlew runTinyHelenSample
# 혹은 특정 체크포인트 직접 지정
./gradlew runTinyHelenSample --args="model/63648/62"
```

프롬프트 목록 (하드코딩됨): `Once upon a time`, `The little girl`, `In the morning`, `He opened the book`, `She said,`.  
샘플 구성: `numberOfSamples=2`, `maximumNewTokens=80`, `samplingTemperature=0.8`, `topKFilteringSize=40`.

### 5. overnight 래퍼

`scripts/run-overnight.sh` — train → sample 연속 실행, 로그는 `run-logs/overnight.log`에 tee, 종료 시 `run-logs/overnight.done`에 `train=<exit> sample=<exit>` 기록.

```bash
nohup bash scripts/run-overnight.sh >/dev/null 2>&1 & disown
```

### 6. 벡터 백엔드로 전환 (≈1M 파라미터)

`vec/` 백엔드(커밋 `25538c3`)가 완성된 뒤, 같은 데이터셋에 **4층 / embd 128 / 4 heads / FFN 512 = ≈1.05M 파라미터** 구성을 `TinyHelenTrainVec`로 구동 가능.

```bash
./gradlew runTinyHelenTrainVec              # 1500 iter 전체 (≈40분)
./gradlew runTinyHelenTrainVec --args="50"  # 짧은 스모크

# 학습 후
./gradlew runTinyHelenSampleVec             # model/vec/ 최신 체크포인트 자동 선택
```

벡터 백엔드는 `model/vec/{params}/{loss*10}/` 경로에 독립적으로 저장 — 스칼라 체크포인트와 간섭 없음.

**측정 (20 iter 스모크, 1M 파라미터)**:

| 구간 | 벡터 백엔드 (1M) | 스칼라 백엔드 (71K) |
|---|---|---|
| iter당 | **≈1.7초** | 16초 |
| 첫 학습 감지 (avg < 6.8) | **iter 5 부근** | iter 150 부근 |
| 20 iter loss | 6.65 | 6.91 (사실상 제자리) |
| 1500 iter 예상 소요 | **≈42분** | ≈7시간 |

파라미터당 기준으로 **약 130× 빠름** (15× 큰 모델을 9× 빠르게 학습).

## 관측 결과 (2026-04-23 ~ 04-24 기준)

### 타이밍

- 학습: 약 6시간 52분 (iter 0 → 1500), **≈16초/iter**
- 샘플링: ≈5분
- 전체: 약 7시간

### Loss 궤적 (검증 평균)

| iter | train | val | 평균 |
|---|---|---|---|
| 0 | 6.91 | 6.91 | 6.909 (baseline = ln 1000) |
| 150 | 6.72 | 6.69 | 6.706 ← 첫 학습 감지 |
| 300 | 6.40 | 6.62 | 6.508 |
| 525 | 6.21 | 6.31 | 6.260 |
| 750 | 6.15 | 6.46 | 6.303 |
| 900 | 6.50 | 6.60 | 6.546 ← 튀어오름 |
| 1200 | 6.14 | 6.32 | **6.231 ← 최저** |
| 1500 | 6.25 | 6.37 | 6.312 |

곡선은 전반적으로 내려가지만 진폭이 크고 iter 1200 이후 반등. 배치=2의 그래디언트 노이즈 + cosine decay 하한 근처의 작은 LR에서 표류하는 조합.

### 저장된 체크포인트 (`model/63648/`)

| 디렉토리 | iter | best val loss |
|---|---|---|
| 67 | 150 | 6.7065 |
| 66 | 225 | 6.6147 |
| 65 | 300 | 6.5077 |
| **62** | **1200** | **6.2309 ← best** |

디렉토리 이름 규약: `model/<총파라미터>/<(bestLoss*10).toInt()>/`.

### 샘플 품질 (best checkpoint, temp 0.8, top-k 40)

```
"Once upon a time" →
cc . ll. cof to ed , y to to a thand  thand isis e a , nes are are s at esee to...

"She said," →
lmea about s e an e a rto an  fe of y sare are bf. lto , e ese the of of s ct...
```

- 구문적으로는 **gibberish**.
- 학습된 신호: 빈도 높은 기능어·접미사 토큰 (`the`, `and`, `to`, `of`, `is`, `are`, `ing`, `ed`)의 확률이 상승. 공백/구두점 패턴 포착.
- BPE subword 조각은 올라오지만 **단어 단위로 묶지 못함** — 실질 어휘 학습이 아니라 토큰 빈도 학습 수준.

Perplexity ≈ e^6.23 ≈ **509** (vocab 1000 대비 랜덤의 약 2배 좋음).

## OOM 1차 실패 (attempt 1)

첫 시도는 iter 0 평가 구간에서 `java.lang.OutOfMemoryError: Java heap space`로 죽음 (`run-logs/overnight-attempt1-oom.log`).

원인: `Trainer.estimateLoss`가 train/val 양쪽에서 `evalIters`개를 `async`로 병렬 실행 → 스칼라 autodiff 그래프 수백만 `Value` 객체가 동시에 존재. 초기 설정 (`evalIters=4`, `numberOfLayers=3`, `blockSize=64`)에서 8GB 힙 초과.

수정:
- `evalIters` 4 → 1 (병렬도 = train 1 + val 1 = 2)
- `numberOfLayers` 3 → 2
- `blockSize` 64 → 48
- `maxIters` 2000 → 1500 (시간 여유)

이후 재실행에서 정상 완료.

## no-grad 도입 기록 (2026-04-24)

`evalIters=1`로 강제되던 근본 원인(병렬 forward × grad 그래프 = 메모리 폭발)을 **구조적으로 해결**:

- `Value.kt`에 `GradContext` object 추가. PyTorch `torch.no_grad()`와 동일한 ThreadLocal 기반 on/off 플래그.
- Value의 6개 graph-building 연산자 (`plus`/`times`/`div` on Value-Value, `pow`, `relu`, `exp`)에 `if (GradContext.enabled)` 가드 삽입. 블록 안에서는 `_parentNodes`/`backwardFunction` 할당을 건너뛰고 스칼라만 계산.
- `Trainer.estimateLoss`는 이제 `GradContext.noGrad { ... }` 안에서 **순차 실행** (`async`/`awaitAll` 제거). 코드도 더 짧아지고 OOM 근본 원인 사라짐.
- `Sampler.generateTokenSequence`도 같은 블록으로 감쌈 (샘플링도 gradient 불필요).

### 벤치 (동일 config, evalIters 제외)

| 구간 | 이전 (evalIters=1, grad eval) | 이후 (evalIters=4, no-grad eval) |
|---|---|---|
| iter 0 elapsed | 23s | 21s |
| iter당 (학습, grad-on) | 15.9s | 16.2s (사실상 동일) |
| eval 배치 수 | 2 (train 1 + val 1) | **8 (train 4 + val 4)** |
| eval 피크 메모리 | 8GB 접근 | 크게 감소 (별도 측정 안 함) |

**핵심**: 학습 iter 자체는 안 빨라짐 (훈련은 여전히 grad 필요). 얻은 것은:
1. **eval 단위 비용 4-5× 하락** → batch 수 4배 늘려도 시간 동일
2. **val loss 추정이 4배치 평균으로 안정** → 학습 곡선 읽기 쉬움
3. **메모리 여유** → 향후 `evalIters`를 더 키울 수도 있음

### 참고 커밋
- `b5fac34` feat: GradContext.noGrad + sequential eval

## 알려진 한계

1. **스칼라 autodiff**: 71K 파라미터 모델에서 iter당 16초. 현대 vectorized 프레임워크였다면 10-100배 빠름. **2K iter 이상의 실험은 항상 overnight**.
2. **배치=2 노이즈**: 그래디언트 분산이 커서 loss 곡선이 매끈하지 않음. 벡터화 이후 batch를 키워야 안정 수렴.
3. **LR 스케줄 후반 플랫**: cosine이 `minimumLearningRate=3e-5`로 너무 일찍 내려가서 후반부에 유의미한 업데이트가 거의 없음. iter 1200 이후 loss 반등이 그 증거.
4. **평가 신뢰도**: ~~`evalIters=1`은 128 토큰 분량의 단일 배치. val 6.32 vs 6.37 같은 차이는 잡음 가능성 큼.~~ **해소됨** — no-grad 도입으로 `evalIters=4` 복원 (위 "no-grad 도입 기록" 참조). 향후 오버나잇 재실행 시 곡선 노이즈 감소 기대.
5. **데이터 규모 대비 under-trained**: 1.65M 토큰을 1500 iter × 효과배치 8 × block 48 = 576K 토큰 노출. 1회도 다 보지 못함 (1 epoch ≈ 3400 iter 필요).

## 다음 실험 아이디어 (선행 조건 포함)

- **벡터화** (필수): `Value` 그래프를 행렬 연산 단위로 재작성. 이게 유일한 진짜 레버.
- 벡터화 이후: batch 16+, block 128, layers 4, embd 128 규모로 실제 언어 학습 관측 가능할 것.
- **Early stopping**: patience 기반으로 iter 1200 전에 종료. 지금 구조에서 바로 적용 가능한 개선.
- **Scheduler 조정**: `learningRateDecayRatio` 0.7-0.8로 줄이고 후반은 `minimumLearningRate` 유지 구간을 없애기.
- **평가 품질**: Trainer `estimateLoss`에서 병렬도를 `Semaphore`로 묶고 `evalIters`를 4-8로 복구 → OOM 없이 val 추정 분산 감소.

## 참조 파일

- 코드: `src/main/kotlin/train/TinyHelenTrain.kt`, `src/main/kotlin/sample/TinyHelenSample.kt`
- 래퍼: `scripts/run-overnight.sh`
- Gradle 태스크: `build.gradle.kts` (`runStoriesBpe`, `runTinyHelenTrain`, `runTinyHelenSample`)
- 로그: `run-logs/overnight.log`, `run-logs/bpe-prep.log`, `run-logs/overnight-attempt1-oom.log`
- 베스트 체크포인트: `model/63648/62/` (gitignore 제외, 필요 시 별도 보관)
- Trainer 본체: `src/main/kotlin/train/Trainer.kt`
- 데이터 형식 참조: `src/main/kotlin/train/DataLoader.kt` (big-endian 4-byte int)
