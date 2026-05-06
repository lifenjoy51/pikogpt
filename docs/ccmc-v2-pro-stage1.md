# CCMC v2-pro Stage 1 학습 기록

**기간**: 2026-05-04 ~ 2026-05-06 (누적 학습 ~7시간 45분)
**대상**: CCMC v2-pro 데이터셋 Stage 1 (binding) — vec 백엔드 ~1M 모델

## Context

`/Users/joey51/works/llm-playground/data/processed/ccmc_v2_pro/pikogpt/` 의 새 코퍼스로 vec backend 학습. lemma 1,826개의 5축(sensory + category + multi_role + contrast + qa) 묶음 중 Stage 1은 binding 4축, Stage 2는 qa.

## 데이터 형식

### 원본 → 변환 후

원본(외부): `<|bos|>\nAbraham has a big tent.\nThis is Abraham's sheep.\n...\n<|eos|>`
- 한 줄 = 1 record = 한 lemma의 sentence 묶음
- literal `\n` (백슬래시+n 두 글자)이 sentence 구분자

문제: BPE가 literal `\n`을 보고 cross-sentence 합성 토큰 75개 (`.\nA`, `.\nT` 등) 생성 → vocab 낭비 + 의미 불명료.

해결: literal `\n` → `<|sep|>` special token으로 치환. 합성 0건, 단일 토큰 1개(id=4).

### 최종 record 구조

- Stage 1: `<|bos|><|sep|>Sentence 1.<|sep|>Sentence 2.<|sep|>...<|eos|>`
- Stage 2: `<|bos|>Question?<|turn|>Answer.<|turn|>...<|eos|>` (literal `\n` 없음)

### 통계

| split | records | 토큰 | 출처 |
|---|---|---|---|
| stage1 train.bin | 1,824 | 602,655 | sensory + category + multi_role + contrast |
| stage1 val.bin | 1,824 | 66,668 | |
| stage2 train.bin | 1,795 | 220,343 | qa pairs |
| stage2 val.bin | 1,795 | 46,486 | |

## 코드 변경

### 1. SEP_TOKEN 정식 5번째 special token 등록

- `src/main/kotlin/data/SimpleBPE.kt` — `SEP_TOKEN = "<|sep|>"` companion 상수 + default specialTokens 리스트에 추가 (id=4)
- `src/main/kotlin/data/MetaInfo.kt` — default specialTokens 5개로 갱신
- `src/main/kotlin/data/BpePrep.kt` — analysis용 토큰 리스트도 5개로

### 2. Record-aware DataLoader

`src/main/kotlin/train/DataLoader.kt`에 `RecordAwareDataLoader` 클래스 추가.
- `<|bos|>` token id로 record 경계 사전 추출
- 두 단계 sampling: (a) random record 선택 → (b) record 안에서 random offset
- 한 슬라이스가 항상 한 record 내부에 머물러 cross-record 노이즈 제거

### 3. TrainConfig 확장

`src/main/kotlin/train/TrainConfig.kt`에 두 필드 추가:
- `samplePrompts: List<String>?` — ckpt 자동 샘플링 prompt 커스터마이즈
- `recordAwareSampling: Boolean` — record-aware loader 활성

### 4. VecTrainer 분기

`src/main/kotlin/vec/VecTrainer.kt`:
- `valLoader` 타입 `DataLoader` → `BatchSource`로 일반화
- `recordAwareSampling=true`이면 train + val 모두 RecordAwareDataLoader 사용 (val도 같은 분포로 평가해 fair comparison)
- 자동 샘플링 prompt가 `config.samplePrompts`에서 옴 (null 시 기본 5개)

### 5. SamplePromptsFromFile temperature 0

`src/main/kotlin/sample/SamplePromptsFromFile.kt:45`: temp 0.8 → 0.0 (greedy, deterministic 비교)

### 6. 진입점 + Gradle 태스크

- `src/main/kotlin/train/experiments/CcmcV2ProStage1TrainVec.kt`
- `src/main/kotlin/train/experiments/CcmcV2ProStage2TrainVec.kt`
- `build.gradle.kts:243-255`: `runCcmcV2ProStage1TrainVec`, `runCcmcV2ProStage2TrainVec`

## 모델 / 학습 설정 (Stage 1)

```
numberOfLayers = 8
numberOfHeads = 3            (head_dim = 32)
embeddingDimension = 96
blockSize = 64
mlpActivation = "swiglu"
positionEncoding = "rope"
tieWeights = true
bias = true
dropout = 0.05f
```
실측 paramCount = **1,087,936** (~1.09M)

```
batchSize = 2, gradientAccumulationSteps = 32   # eff batch 4096 tok/step
learningRate = 3e-4, minimumLearningRate = 3e-5
warmupRatio = 0.03, learningRateDecayRatio = 0.95, decayLr = true
weightDecay = 0.05, labelSmoothing = 0.05
beta1 = 0.9, beta2 = 0.95, gradClip = 1.0
maxIters = 10000             # 최초 5000 → 의미 연결 약해 5000 추가
evalIntervalRatio = 0.02     # 100 iter마다 eval/ckpt/sampling
evalIters = 100
alwaysSaveCheckpoint = true  # 매 ckpt 보존
recordAwareSampling = true
samplePrompts = 10개 (Cat is / Water is / Run means / ... 등 다양한 POS)
```

## 데이터 prep 절차 (재현)

```bash
mkdir -p data/ccmc-v2-pro/{shared,stage1,stage2}
cp /Users/joey51/works/llm-playground/data/processed/ccmc_v2_pro/pikogpt/stage1/{train,val}.txt data/ccmc-v2-pro/stage1/
cp /Users/joey51/works/llm-playground/data/processed/ccmc_v2_pro/pikogpt/stage2/{train,val}.txt data/ccmc-v2-pro/stage2/

# literal \n → <|sep|> (Stage 1만 영향, Stage 2는 0건)
python3 -c "
for sub in ('stage1/train.txt','stage1/val.txt','stage2/train.txt','stage2/val.txt'):
    p='data/ccmc-v2-pro/'+sub
    t=open(p).read()
    open(p,'w').write(t.replace('\\\\n','<|sep|>'))
"

# 합본 (BPE vocab 학습용)
cat data/ccmc-v2-pro/stage1/train.txt data/ccmc-v2-pro/stage2/train.txt > data/ccmc-v2-pro/shared/train.txt
cat data/ccmc-v2-pro/stage1/val.txt   data/ccmc-v2-pro/stage2/val.txt   > data/ccmc-v2-pro/shared/val.txt

# 공유 vocab 학습 (vocab 2000, cased, skip-bin: meta.json만)
./gradlew runBpe --args="data/ccmc-v2-pro/shared 2000 cased skip-bin"
# → vocab 2000, merges 1884, special 5개 (eos/unk/bos/turn/sep)

# 각 stage encode
./gradlew runEncodeWithExistingMeta --args="data/ccmc-v2-pro/shared/meta.json data/ccmc-v2-pro/stage1"
./gradlew runEncodeWithExistingMeta --args="data/ccmc-v2-pro/shared/meta.json data/ccmc-v2-pro/stage2"
```

## 학습 실행

```bash
# 첫 5000 iter (scratch)
./gradlew runCcmcV2ProStage1TrainVec
# → best 4600 (val 2.486), final 5000 (val 2.80)

# 의미 연결 강화 위해 5000 iter 추가 (resume + maxIters override)
./gradlew runCcmcV2ProStage1TrainVec --args="resume 10000"
# → best 8800 (val 2.1895), final 10000 (val 2.67, train 1.88)
```

## 결과

### val loss 추이

| iter | val | 비고 |
|---|---|---|
| 0 | 7.62 | baseline ≈ ln(2000) |
| 100 | 6.47 | |
| 500 | 4.46 | |
| 1000 | 3.83 | |
| 2500 | 3.03 | |
| 3600 | 2.94 | |
| 4200 | 2.76 | |
| 4600 | **2.486** | 첫 best (5000 iter 학습 종료 직전) |
| 5000 | 2.80 | cosine LR plateau |
| 5000 (resume) | 2.474 | maxIters 재계산 후 LR 회복 |
| 7400 | 2.59 | |
| **8800** | **2.1895** | **최종 best (v0072)** |
| 10000 | 2.67 | final (val ↑, train 1.88로 과적합 영역) |

### Best 체크포인트

- **`model/stage1/vec/1087936/v0072/`** (iter 8800, val 2.1895)
- 다음 단계 Stage 2 warm-start에 사용

### 샘플 quality 진화 (greedy T=0)

prompt: `<|bos|><|sep|>Cat is`

| iter | 샘플 |
|---|---|
| 100 | "..The lo can make a toy ares.." (random tokens) |
| 800 | "a big big staces. An angan was very soft and warm.." (sentence 구조 잡힘) |
| 1900 | "a big picture of the sun. We put a car where it is a small book.." (자연스러운 sentence) |
| 2500 | "a revisrogogen as answers are to a new prast.." (analogy 패턴) |
| 4200 | "a thing you can see with it. Cats eat sweet food.." (lemma 직접 등장) |
| 7400 | "a small fruit, but some apple is not true.. Cats can be fast, but a dog has no war." (contrast) |
| 10000 | "a thing you can feel and bumpy. Cats look at the big building.." (자연스러움 + lemma) |

prompt: `<|bos|><|sep|>Happy means` (의미 매핑이 가장 명확하게 발현된 case)

| iter | 샘플 |
|---|---|
| 4200 | "you go to the other. The rabbit's body is big.." (binding 약함) |
| 10000 | "**Happy is good, but now is a sad. Running helps you feel happy.**" (good↔sad contrast + running 연관) |

### Record-aware vs Uniform sampling 비교

같은 iter 1000 시점:
- **Uniform** (val 2.95): "X is Y", contrast, 일반 sentence — analogy 드물게 등장
- **Record-aware** (val 3.83): analogy "X is to Y as W is to Z" 패턴이 압도적으로 자주 발현

val loss는 record-aware가 더 높지만 (val 데이터의 record 외부 token 예측이 약하게 측정), 실제 샘플은 record 내부 응집(특히 CCMC contrast 축의 analogy)을 더 강하게 학습.

### 학습 효율

- iter당 ~3.0초 (1.087M 모델, 8L 깊이, 데이터 병렬 worker 4)
- 첫 5000 iter: 3시간 46분
- 추가 5000 iter (resume): 3시간 58분

## 자동 샘플링 prompt (Stage 1)

매 ckpt마다 stdout에 출력되어 진행 추적 가능:

```
<|bos|><|sep|>Cat is
<|bos|><|sep|>Water is
<|bos|><|sep|>Tree is
<|bos|><|sep|>Run means
<|bos|><|sep|>Eat means
<|bos|><|sep|>Happy means
<|bos|><|sep|>Big means
<|bos|><|sep|>Above means
<|bos|><|sep|>Quickly means
<|bos|><|sep|>And means
```

POS 분포: noun×3 (cat/water/tree), verb×2 (run/eat), adj×2 (happy/big), prep (above), adv (quickly), conj (and)

## 디렉터리 정리

학습 도중 여러 시도가 누적되어 v0001~v0033까지 다른 실험의 ckpt가 섞였음. 정리:
- `model/stage1/vec/1087936/_archive/` — 이전 시도들 33개 ckpt 보존 (uniform sampling 학습 결과 등)
- `model/stage1/vec/1087936/v0001 ~ v0078` — 현재 record-aware 학습 시퀀스 (78개)

## Stage 2 다음 단계

```bash
./gradlew runCcmcV2ProStage2TrainVec --args="model/stage1/vec/1087936/v0072"
```

설정: pretrain_weights warm-start + Stage 1 replay 0.25, LR 1e-4, maxIters 3000.

> Note: 현재 `CcmcV2ProStage2TrainVec.kt`에는 `recordAwareSampling=true` 미적용 상태. 일관성 위해 Stage 2도 활성화하려면 진입점에 추가 후 학습.

## 관찰 / 후기

- **literal `\n` → `<|sep|>` 효과 명확**: 합성 토큰 75개 → 0개. vocab 효율 ↑.
- **record-aware sampling 효과**: val loss는 측정 분포 차이로 약간 높지만 sample quality에선 CCMC 특유의 analogy 패턴을 더 강하게 학습. lemma-grouped 데이터엔 적합.
- **5000 → 10000 iter 추가**: 후반 5000 iter는 작은 데이터셋(0.6M tok)에 ~33 epoch 추가라 과적합 영역. train loss는 1.88까지 떨어지지만 val 2.67로 상승. 사용자 의도(과적합이라도 의미 강화)대로 진행.
- **의미 매핑은 prompt별 차이**: Happy("good but sad" contrast)는 매우 명확, Cat/Water는 lemma 등장 정도, Run은 iter 7400 peak 후 약화 (train fit이 의미 표현보다 sentence 패턴에 집중).
- **best는 iter 8800** — 그 이후는 val 상승 plateau. Stage 2 warm-start는 v0072 사용.
