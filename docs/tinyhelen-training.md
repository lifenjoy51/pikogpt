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
  jq -r '.text' "/tmp/TinyHelen/data/leaner/10M/train/${f}0000.jsonl"
  echo ""
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
    evalIters = 1,                   // OOM 회피
    logInterval = 10,
    alwaysSaveCheckpoint = true,
)
```

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

## 알려진 한계

1. **스칼라 autodiff**: 71K 파라미터 모델에서 iter당 16초. 현대 vectorized 프레임워크였다면 10-100배 빠름. **2K iter 이상의 실험은 항상 overnight**.
2. **배치=2 노이즈**: 그래디언트 분산이 커서 loss 곡선이 매끈하지 않음. 벡터화 이후 batch를 키워야 안정 수렴.
3. **LR 스케줄 후반 플랫**: cosine이 `minimumLearningRate=3e-5`로 너무 일찍 내려가서 후반부에 유의미한 업데이트가 거의 없음. iter 1200 이후 loss 반등이 그 증거.
4. **평가 신뢰도**: `evalIters=1`은 128 토큰 분량의 단일 배치. val 6.32 vs 6.37 같은 차이는 잡음 가능성 큼.
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
