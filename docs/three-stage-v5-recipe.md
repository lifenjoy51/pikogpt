# three-stage v5 chain — 옵션 C (2.93x scale) 실험 보고

## 요약

v4 (864K params) chain에서 wiki 단계 의미 매핑 깜빡임 (Run "go/move" 일부 ckpt만 정합) 한계 진단 → **architecture를 ~2.93x 확대한 v5 chain**으로 재학습. 데이터·LR·iter는 v4와 동일하게 두고 capacity 효과만 분리 측정.

- **paramCount**: 2,546,352 (v4의 2.95x)
- **architecture 변경 (옵션 C)**: emb 96→144, layers 6→9, heads 3→6 (head_dim 24 유지), dropout 0.05→0.10
- **학습 시간 합계**: dict 18h 10m + wiki 15h 55m + conv 20h 29m ≈ **54.5h**
- **chain best ckpts**: dict v0049 (val 1.666) → wiki v0038 (val 3.080) → conv v0047 (val 2.709)
- **의미 정합 peak**: wiki v0021 40% (biology cluster narrative), conv v0042 13%, conv v0047 (다양 prompt) 24%

핵심 결론은 **§7 결과/한계** 절 참고.

---

## 1. 동기

v4 chain (864K params, dict→wiki→conv) 종료 후 sampling 분석:
- **dict 단계**: train 1.40 / val 1.93 (gap 0.53) — 형식 학습은 되지만 단어→정의 매핑 학습 실패
- **wiki 단계**: 의미 정합 깜빡임 — Run "go/move" 매핑이 일부 ckpt에서만 등장
- **결론**: capacity 부족 신호. MLP 용량(emb²)과 합성 깊이를 동시에 늘려 단어→hypernym 매핑을 안정 인코딩

옵션 C 선택 이유:
- emb 144 = head_dim 24 × 6 heads (head_dim 유지로 attention dynamics 보존)
- layers 9 = 6×1.5 (compositional depth ↑)
- dropout 0.05→0.10 (params 2.93x인데 dict 토큰은 1.107M로 동일 → overfit 보강)

---

## 2. 데이터 (v4와 동일, `data/three-stage-v4/` 재사용)

`data/three-stage-v4/` 그대로 사용. shared/meta.json (vocab=2000) 기반 BPE 단일 인코딩.

| stage | dataPath | tokens (train) | replay |
|---|---|---|---|
| 1. dict | data/three-stage-v4/dict | 1,107,220 | — |
| 2. wiki | data/three-stage-v4/wiki | 7,485,510 | dict 0.30 |
| 3. conv | data/three-stage-v4/conv | 2,933,638 | dict 0.15 + wiki 0.15 |

dict 데이터 형식: WordNet abstract gloss 기반 (`X means [verb]` ~29%, `An X is Y` ~36%, `is a kind of` ~11%, Similar/Opposite ~25%).

---

## 3. 학습 entry points (신규 3개)

### `src/main/kotlin/train/experiments/ThreeStageDictTrainV5Vec.kt`
```
emb 144 / layers 9 / heads 6 / dropout 0.10
batch 2 × blockSize 64 × gradAccum 32
LR 3e-4 / warmup 0.03 / cosine 0.95 / minLR 3e-5
labelSmoothing 0.05 / weightDecay 0.05 / gradClip 1.0
maxIters 22000 / earlyStopPatience 10 / evalIntervalRatio 0.02
mlpActivation swiglu / positionEncoding rope / tieWeights true
initFrom scratch
```

### `src/main/kotlin/train/experiments/ThreeStageWikiTrainV5Vec.kt`
- pretrain_weights = Stage 1 dict best (v0049)
- replayDataPath = dict, replayRatio 0.30
- maxIters 20000, warmup 0.015, LR 1e-4 (보수)

### `src/main/kotlin/train/experiments/ThreeStageConvTrainV5Vec.kt`
- pretrain_weights = Stage 2 wiki best (v0038)
- replayDataPath = dict (0.15), replayDataPath2 = wiki (0.15)
- maxIters 24000, warmup 0.05, LR 1e-4

build.gradle.kts에 `runThreeStageDictTrainV5Turbo`, `runThreeStageWikiTrainV5Turbo`, `runThreeStageConvTrainV5Turbo` 3개 task 추가.

---

## 4. 실행

`/tmp/v5-train/run.sh`로 chain 자동화:
```
VEC_MAX_WORKERS=10
1) runThreeStageDictTrainV5Turbo → dict best 추출
2) runThreeStageWikiTrainV5Turbo --args=<dict best> → wiki best 추출
3) runThreeStageConvTrainV5Turbo --args=<wiki best>
```

## 5. 학습 결과

### Stage 1 — dict (scratch, 22000 iter)

| iter | train | val | val avg | 비고 |
|---|---|---|---|---|
| 17160 | 1.43 | 1.91 | 1.670 | v0040 best (until v0049) |
| 18920 | 1.40 | 1.98 | 1.689 | counter 4/10 |
| 21120 | 1.40 | 1.93 | **1.666** | **v0049 best (final)** |
| 22000 | 1.43 | 1.97 | 1.701 | maxIters 종료, BUILD 18h 10m |

- best갱신 추이: ... → v0040 → v0049 (counter reset)
- 형식 학습 매우 빠름 (v0010부터 BPE 분해 안정)
- **의미 정합**: Run "cause to move" 7+ ckpt 안정 — 유일한 motion verb cluster
- Apple "Exle/Exic" 환각 prefix 영구 형성 (v0042~v0050)

### Stage 2 — wiki (pretrain v0049, 20000 iter)

| iter | train | val | val avg | 비고 |
|---|---|---|---|---|
| 0 | 4.14 | 5.21 | 4.672 | wiki 도메인 baseline |
| 8000 | 2.84 | 3.46 | **3.150** | v0021 — **biology cluster 폭발** (semantic peak) |
| 14800 | 2.79 | 3.37 | **3.080** | **v0038 best (final)** |
| 18800 | — | — | — | Early stop (10 ckpt 갱신 없음), BUILD 15h 55m |

- best 갱신 추이: v0019 (3.175) → v0021 → v0023 → v0026 → v0031 → v0033 → **v0038**
- val 단조 ↓, 약 -1.6 in 18800 iter
- **wiki context가 dict 환각 정화** — Apple "Exle" 영구 환각이 v0003에서 사라짐
- **wiki v0021 biology cluster** (Tree → "phylogenesis, cell, pigment, food, natural defenses") = chain 전체 semantic peak (40%)
- 그러나 후속 ckpt에서 cluster 사라짐 — 일시 attractor

### Stage 3 — conv (pretrain v0038, 24000 iter)

| iter | train | val | val avg | 비고 |
|---|---|---|---|---|
| 0 | 4.76 | 5.88 | 5.320 | conv 도메인 baseline |
| 2880 | 3.01 | 3.17 | 3.093 | v0007 |
| 5280 | 2.89 | 3.00 | 2.946 | v0012 |
| 8160 | 2.87 | 2.88 | 2.874 | v0018 best (이 시점) |
| 12000 | 2.76 | 2.86 | 2.814 | v0026 |
| 15840 | 2.72 | 2.78 | **2.750** | v0034 |
| 19680 | 2.71 | 2.76 | **2.734** | v0042 (semantic peak) |
| 22080 | 2.67 | 2.75 | **2.709** | **v0047 best (final)** |
| 24000 | — | — | — | maxIters 종료, BUILD 20h 29m |

- best 갱신 추이: ... → v0034 → v0042 → v0044 → **v0047**
- val 단조 ↓, 약 -2.6 in 22080 iter (가장 큰 감소율)
- **wiki narrative + conv 대화체 결합** 학습 — definition format ("It means that...") 안정화
- 의미 점수 conv 후반에 가시적 진전 (Tree → seeds/squirrel/birds/forest, Apple → red/yummy/sweet/applesauce)

---

## 6. 의미 정합 평가 방법론

### 6.1 dict-style (`# Word\n` prompt) — 부적합

```
# Apple\n → Apple\nOpposite: applicant.
# Cat\n → Cat means stooge into a cat.
```

- conv 학습된 모델이 dict 형식 prompt에 "X means Y" 패턴으로 응답 시도
- 의미 매핑 거의 없음 (점수 0~5%)
- **conv ckpt 평가에는 부적합** — 학습 안 한 prefix

### 6.2 대화체 v1 (10 prompts) — `prompts/conv-semantic.txt`

```
What is an apple?
Tell me about cats.
Why are trees so tall?
What animal says meow?
What grows on trees?
What color is an apple?
...
```

- conv 형식 정합 (sampler가 자동으로 `<|turn|>` append → single-turn 응답)
- v0042 peak: 13% (yummy/sweet apple, applesauce, high tree, "It means that")

### 6.3 대화체 v2 (15 prompts, 다양화) — `prompts/conv-semantic-v2.txt`

```
Name three fruits.
Name three animals you know.
Where do fish live?
A banana is yellow. An apple is what color?
A dog has four legs. How many legs does a cat have?
What is the opposite of big?
What do bees make?
Is ice hot or cold?
...
```

- 직접 commonsense 질의 (사실/속성/예시/비교/반대말)
- v0047 final: **24%** (fish→water, hungry→pizza, moon→dark 정확 hit)
- **이전 평가가 narrow했음을 발견** — 모델이 의미를 더 알고 있음

---

## 7. 결과 / 한계 / 핵심 인사이트

### 의미 점수 종합

| ckpt | iter | val avg | 평가셋 | 점수 | 핵심 hit |
|---|---|---|---|---|---|
| dict v0049 | 21120 | 1.666 | dict-style | 20% | Run "cause to move" |
| **wiki v0021** (peak) | 8000 | 3.150 | dict-style | **40% 🌟** | biology cluster (Tree → phylogenesis/cell/pigment) |
| wiki v0038 (val best) | 14800 | 3.080 | (미평가) | — | — |
| conv v0042 (peak) | 19680 | 2.734 | 대화 v1 | 13% | yummy/sweet apple, applesauce, high tree |
| **conv v0047** (val best) | 22080 | **2.709** | 대화 v1 | 10% | RED apple, plants, animals |
| **conv v0047** | 22080 | **2.709** | **대화 v2** | **24% 🎯** | **fish→water, hungry→pizza, moon→dark** |

### 핵심 인사이트

1. **wiki narrative가 conv "X is Y" 매핑보다 의미적으로 풍부할 수 있음** — wiki v0021 biology cluster (40%)는 conv 어떤 ckpt보다 강함. 단, 일시 attractor — 안정화 못함.

2. **val loss ↓와 의미 점수 ↑는 약상관** — conv 단계에서 val 3.09 → 2.71로 -0.38 감소했지만 의미 점수는 5-13% 진동. **형식 학습이 우선되고 의미는 부차적으로 emerge**.

3. **평가 방법론이 모델 능력을 underestimate** — narrow한 prompt로 평가 시 5-10%, 다양한 commonsense 질의로 평가 시 24%. **prompt diversity가 평가의 핵심**.

4. **Apple "Exle" 영구 환각 → wiki에서 정화** — dict 학습 후반에 형성된 잘못된 attractor가 wiki 도메인 노출로 사라짐. Multi-stage curriculum의 catastrophic forgetting 완화 효과를 어느 정도 입증.

5. **conv 데이터 한계** — 부모-아이 대화체로 명시적 정의 ("X is Y") 거의 없음 → 대화 형식만 학습. 진짜 의미 학습은 wiki/dict replay에 의존.

### 데이터 한계 (v6에서 개선)

- **dict**: WordNet abstract gloss ("involve in a function...") — "X is Y" 패턴 11%만, 단어당 entry 1-3개 → distributional vector narrow
- **wiki**: simplewiki vital articles (~4M words) — apple이 다양한 context로 등장하지만 명시적 정의 부재
- **conv**: 부모-아이 대화 — 의미보다 대화 형식 위주

토큰당 노출량 분석:
- dict: 1.107M tokens × 22000 iter × 4096 batch = 토큰당 **81 epoch** — 노출 부족 아님
- 한계는 **데이터 형식과 다양성** (paraphrase 부족, 명시 매핑 비율 낮음)

---

## 8. v4 vs v5 비교

| 지표 | v4 (864K) | v5 (2.55M) | 비고 |
|---|---|---|---|
| paramCount | 864K | 2.55M | 2.95x |
| dict best val avg | ~1.7 | 1.666 | 비슷 |
| wiki best val avg | ~3.3 | 3.080 | -7% |
| conv best val avg | ~3.0 | 2.709 | -10% |
| dict 형식 학습 | ~iter 18920 안정 | ~iter 1320 안정 | **5-10x 빠름** |
| wiki biology cluster | (없음) | **v0021 40%** | **신규** |
| Apple "Exle" 환각 | 일부 ckpt | 영구 (dict v0042~v0050) | overfit ↑ |

**capacity 효과**:
- 형식 학습 5-10x 빠름 ✓
- val loss 7-10% ↓ ✓
- 의미 정합은 wiki에서 한 ckpt 한정 강함 (v0021), but 안정 못함
- **데이터 한계가 capacity 한계보다 강함**

---

## 9. 영향받은 파일

신규:
- `src/main/kotlin/train/experiments/ThreeStageDictTrainV5Vec.kt`
- `src/main/kotlin/train/experiments/ThreeStageWikiTrainV5Vec.kt`
- `src/main/kotlin/train/experiments/ThreeStageConvTrainV5Vec.kt`
- `prompts/conv-semantic.txt` (10 대화 prompts)
- `prompts/conv-semantic-v2.txt` (15 다양 commonsense prompts)
- `docs/three-stage-v5-recipe.md` (본 문서)

갱신:
- `build.gradle.kts` — 3개 신규 task

신규 (모델, gitignored):
- `model/dict/vec/2546352/v0001~v0051/`
- `model/wiki/vec/2546352/v0001~v0048/`
- `model/conv/vec/2546352/v0001~v0051/`

영향 없음:
- 코어 학습 코드 (`VecTrainer.kt`, `TrainConfig.kt`) 수정 불필요
- `data/three-stage-v4/` 재사용 (수정 없음)

---

## 10. 다음 단계 권장 (v6 chain)

### Plan A — 합성 데이터 (Gemma rewrite, 1순위 추천)

dict 데이터 한계 직접 해결:
- 표제어 5K × 5-10 paraphrase = 25-50K docs
- 형식 강제: "X is Y" 50%+, 예문 30%, Q→A 20%
- Simple/short 어휘 (5세 친화적)
- 기대 의미 점수: **20-40% → 60-80%**

```
An apple is a sweet red fruit.
Apples grow on apple trees.
You can eat an apple raw or cooked.
My favorite fruit is the apple.
Apples are round and crunchy.
```

### Plan B — Instruction format 추가

- "What is X?" / "X is Y" pair 합성
- 동일 entity에 다양한 query 변형
- Q→A format 명시 학습 신호 강함

### Plan C — TinyHelen 도메인 도입

- 이미 검증된 narrative kid corpus (300K-1M tokens)
- dict 대체 또는 추가 stage로
- 같은 entity (cat, dog, apple) 다양한 context 반복

### Plan D — wiki "X is Y" 패턴 발췌

- simple wiki에서 명시 정의 sentence만 추출
- 자연스러우나 양 적음 (~50K sentences 추정)

---

## 11. 사용/재현

```bash
# v5 chain 전체 실행 (54h 소요)
./gradlew runThreeStageDictTrainV5Turbo
./gradlew runThreeStageWikiTrainV5Turbo --args="model/dict/vec/2546352/v0049"
./gradlew runThreeStageConvTrainV5Turbo --args="model/wiki/vec/2546352/v0038"

# 의미 평가
./gradlew runSamplePromptsFromFile --args="model/conv/vec/2546352/v0047 prompts/conv-semantic-v2.txt"

# 인터랙티브 대화
./gradlew runChatTurbo --args="model/conv/vec/2546352/v0047"
```

---

## 12. 참고

- v4 recipe: `docs/three-stage-v4-recipe.md`
- 코어 학습 루프: `src/main/kotlin/vec/VecTrainer.kt`
- TripleDataLoader (multi-replay): `src/main/kotlin/train/DataLoader.kt`
- 샘플링: `src/main/kotlin/sample/SamplePromptsFromFile.kt`
- 학습 로그: `/tmp/v5-train/{dict,wiki,conv}.log` (gitignored)
