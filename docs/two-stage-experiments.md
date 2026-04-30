# Two-stage 학습 실험 — Topic Relevance 강화

dialogues-only 베스트 모델의 약점(질문 토픽과 무관한 답변)을 해결하기 위해
사실 풍부한 코퍼스로 BASE pretrain → dialogue로 IT finetune하는 두 단계 학습 실험 기록.

| Model | val loss | 비고 |
|---|---:|---|
| baseline (dialogues-a510) | 2.69 | `model/dialogues-a510/vec/768000/v0013` |
| v1 BASE (TinyHelen wiki+textbook) | 2.97 | `model/base/vec/768000/v0012` |
| v1 IT (BASE → dialogues + 20% replay) | **2.84** | `model/it/vec/768000/v0013` — topic 어휘 일부 발현 |
| v2 (계획) | — | 데이터/BPE/인프라 준비 완료, 학습 미실행 |

---

## v1 — 첫 시도 (768K params, vocab 1000)

### 데이터
- BASE: `TinyHelen leaner/100M`의 `wiki + textbook` (~8.48M BPE tokens, 28 MB plain).
- IT: `data/dialogues-a510` (~17.9M tokens). symlink 재사용.
- 정제 규칙(`scripts/build-base-corpus.sh`): jq로 `.text` 추출 → literal `\n` + actual newline 모두 공백 치환 → 1라인 = 1doc.

### 모델
- `embeddingDimension=96, numberOfLayers=6, numberOfHeads=3, blockSize=64`.
- SwiGLU + RoPE + tied weights → **768,000 params**.

### 인프라 추가
- `vec/VecAdamW.kt`: `resetState()` (timeStep/m/v 0 reset).
- `train/TrainConfig.kt`: `pretrainCheckpointDir`, `replayDataPath`, `replayRatio`.
- `vec/VecTrainer.kt`: `initFrom="pretrain_weights"` 분기 + `MixedDataLoader` 사용 분기.
- `train/DataLoader.kt`: `BatchSource` interface + `MixedDataLoader` (시퀀스 단위 Bernoulli replay).
- `data/EncodeWithExistingMeta.kt`: 공유 vocab으로 분리 .bin 인코딩.
- `train/experiments/TwoStageBaseTrainVec.kt`, `TwoStageITTrainVec.kt`: 진입점.
- `build.gradle.kts`: 3개 task 추가.

### 결과
- 시간: BASE 5h21m + IT 4h17m = ~10시간 (vocab 1000 BPE는 SimpleBPE의 unique-word 압축으로 80분 → 향후 ~5분).
- 정성(`topic-relevance-prompts.txt` 8개):
  - **개선**: "What is the sun?" → "the sun needs to do" / "Why does it rain?" → "the water is all through" / "Where does water come from?" → "drink the water...stay hot in the sunshine" / "How do plants grow?" → "the dirt is a long thing" — sun/water/dirt 어휘가 dialogue 형식으로 발현.
  - **부족**: sky/grass/moon은 여전히 무관. 표현 다양성/정확성이 BASE 단계에서 부족.
- 정량: IT val 2.84 vs baseline 2.69 — BASE+dialogue mix로 단순 dialogue fitting 손해 0.15.

### 진단
표현 다양성/정확성은 BASE에서 형성됨. v1 BASE 8.48M tokens / 768K 모델은 양·다양성 모두 부족. v2에서 데이터 6× + vocab 2× 확장.

---

## v2 — 데이터/Vocab 확장 (864K params, vocab 2000) — 인프라 완료, 학습 미실행

### 데이터 (~57M total, 6× v1)
- BASE: `wiki + textbook + book + web(4 shards)` (~50M tokens, 203 MB plain). conversation/code/math 제외.
- IT: dialogues-a510 정제본 (~16M tokens, 60 MB).
- vocab 2000 (special 4 + 학습 1996, 단일 char 54개, merges 1942).

### 데이터 파이프라인 (3단계)

#### 1. raw 정제 — `scripts/build_base_v2_corpus.py`
TinyHelen jsonl 7개 (`wiki, textbook, book, web0000-0003`)를 streaming으로 읽어 `<|bos|>{cleaned}<|eos|>` 한 줄에 1 doc.
- jq 버전 (`build-base-v2-corpus.sh`) 도 있지만 web 큰 shard에서 macOS jetsam SIGKILL 발생 → Python streaming으로 대체.
- `clean()`: literal `\n` + actual newline + tab → 공백 1개 → multi-space squeeze.

#### 2. 정제 강화 — `scripts/clean_v2_data.py`
두 단계:
1. **strict (smart quote / 외국어 / 이모지 제거)**: smart quotes → ASCII quotes, em/en-dash → hyphen, ellipsis → 3 dots, 그 외 ASCII 외 chars(`ñ`, `é`, `🎶` 등) 제거.
2. **char-freq 동적 임계값**: 합본에서 `_`(underscore) 빈도 측정 → 임계값 = `_freq + 1`. `_` 자체와 그보다 자주 등장하지 않은 char(`/`, `$`, `%`, `&`, `` ` ``, `@`, `]`, `[`, `=`, `+`, `}`, `~`, `{`, `^`, `\`) 모두 공백으로 치환. 항상 보존: 알파벳, 숫자, 공백/newline, special token wrap chars (`<`, `|`, `>`), 자주 등장 punctuation.
   - 측정 시점 기준 임계값 6647(`_` freq), 16개 char 제거 → vocab base 96 → 80.
   - 동적 계산이라 다른 데이터셋에도 자동 적응.

#### 3. BPE 학습 + 인코딩
- `./gradlew runStoriesBpe --args="<dir> <vocab> [skip-bin]"` (skip-bin은 합본 .bin 작성 건너뛰고 meta.json만 — 큰 코퍼스 OOM 회피).
- `./gradlew runEncodeWithExistingMeta --args="<meta.json> <target_dir>"`로 base/it 분리 인코딩.

### SimpleBPE 최적화 (`data/SimpleBPE.kt`)

| 단계 | 최적화 | 측정 (vocab 2000, 264 MB 합본) |
|---|---|---|
| Phase 1: train (countPairs/applyMerge) | unique-word 압축 + char→String 캐시 + Kotlin coroutines 병렬 (CPU 8 worker) | **140초** (이전 v1 vocab 1000 합본 88 MB가 80분 → 단일 thread → 병렬 후 ~30× 가속) |
| Phase 2: encode (apply merges + ID 변환) | **word-level cache** — 같은 word는 1번만 merge 처리 후 IntArray 캐싱. Zipf 분포로 cache hit ratio ~1/180 | base-v2 train 203MB → **39초** (cache 적용 전 3시간 20분 진행해도 안 끝남) |

### 모델 아키텍처
v1과 동일 (`embeddingDimension=96, L=6, heads=3, SwiGLU, RoPE, tied`). vocab 1000→2000으로 token embed +96K → **864,000 params**.

### 학습 entry points (미실행)
- `TwoStageBaseV2TrainVec`: `dataPath="data/two-stage-v2/base-v2"`, `maxIters=18000` (1.5 epoch), `evalIters=200`.
- `TwoStageITV2TrainVec`: `dataPath="data/two-stage-v2/it-v2"`, `replayRatio=0.2`, `maxIters=8000`, `pretrainCheckpointDir=args[0]`.
- ckpt: `model/base-v2/vec/864000/`, `model/it-v2/vec/864000/`.

### 미완료
v2 BASE pretrain은 시작 후 ~3분에 사용자 요청으로 중단. ckpt 저장 안 됨.
재시작 시 데이터/BPE/모델 그대로 사용 가능.

---

## 재시작 절차 (v2)

```bash
# 0. 데이터 (이미 빌드됨, 다시 안 해도 됨)
#    data/two-stage-v2/base-v2/{train,val}.{txt,bin}
#    data/two-stage-v2/it-v2/{train,val}.{txt,bin}
#    data/two-stage-v2/shared/meta.json (vocab 2000)

# 만약 raw에서 다시 빌드한다면:
git clone --depth 1 https://github.com/EmpathYang/TinyHelen.git /tmp/TinyHelen   # 휘발 시
python3 scripts/build_base_v2_corpus.py
cp data/dialogues-a510/train.txt data/two-stage-v2/it-v2/train.txt
cp data/dialogues-a510/val.txt   data/two-stage-v2/it-v2/val.txt
python3 scripts/clean_v2_data.py            # smart-quote + 동적 char-freq 정제
cat data/two-stage-v2/base-v2/train.txt data/two-stage-v2/it-v2/train.txt > data/two-stage-v2/shared/train.txt
cat data/two-stage-v2/base-v2/val.txt   data/two-stage-v2/it-v2/val.txt   > data/two-stage-v2/shared/val.txt
./gradlew runStoriesBpe --args="data/two-stage-v2/shared 2000 skip-bin"   # ~3분
./gradlew runEncodeWithExistingMeta --args="data/two-stage-v2/shared/meta.json data/two-stage-v2/base-v2"  # ~40초
./gradlew runEncodeWithExistingMeta --args="data/two-stage-v2/shared/meta.json data/two-stage-v2/it-v2"   # ~15초

# 1. v2 BASE pretrain (~10.5h)
./gradlew runTwoStageBaseV2TrainVec

# 2. v2 IT finetune (~4.7h, BASE 끝난 후)
./gradlew runTwoStageITV2TrainVec --args="model/base-v2/vec/864000/v00XX"

# 3. 4-way 비교
./gradlew runSamplePromptsFromFile --args="model/dialogues-a510/vec/768000/v0013 topic-relevance-prompts.txt"
./gradlew runSamplePromptsFromFile --args="model/it/vec/768000/v0013 topic-relevance-prompts.txt"
./gradlew runSamplePromptsFromFile --args="model/base-v2/vec/864000/v00XX topic-relevance-prompts.txt"
./gradlew runSamplePromptsFromFile --args="model/it-v2/vec/864000/v00YY topic-relevance-prompts.txt"
```

## 코드 변경 요약

### 신규
- `scripts/build_base_v2_corpus.py` — Python streaming jsonl 정제
- `scripts/clean_v2_data.py` — smart-quote + 동적 char-freq 정제
- `scripts/build-base-v2-corpus.sh` — jq 버전 (web shard에서 OOM 발생, 사용 X)
- `src/main/kotlin/data/EncodeWithExistingMeta.kt` — 공유 meta.json으로 분리 .bin 인코딩
- `src/main/kotlin/train/experiments/TwoStageBaseV2TrainVec.kt`, `TwoStageITV2TrainVec.kt`
- `src/main/kotlin/train/experiments/TwoStageBaseTrainVec.kt`, `TwoStageITTrainVec.kt` (v1)
- `topic-relevance-prompts.txt` (8개 사실 질문)
- `docs/two-stage-experiments.md` (이 문서)

### 수정
- `src/main/kotlin/data/SimpleBPE.kt`
  - `countPairsCompressed`/`applyMergeCompressed`: unique-word 압축
  - `charStringCache`: Char → String 캐시로 OOM 방지
  - `coroutineScope` + `Dispatchers.Default` 병렬 (countPairs/applyMerge)
  - `encode()`: word-level cache (~수백× 가속)
  - `applyAllMergesToWord()`: encode helper
- `src/main/kotlin/data/StoriesBpePrep.kt`
  - main args[1] = vocab size, args[2] = "skip-bin"
  - unique_words.txt 생성 시 special token 미리 공백으로 둘러쌈 + `[^a-z\s]` → 공백 치환 (이전: 제거 → 합쳐짐 버그)
  - BPE train 직후 meta.json 즉시 저장 (encode 실패해도 vocab 보존)
- `src/main/kotlin/vec/VecAdamW.kt` — `resetState()` (v1)
- `src/main/kotlin/vec/VecTrainer.kt` — `initFrom="pretrain_weights"` 분기 + MixedDataLoader (v1)
- `src/main/kotlin/train/TrainConfig.kt` — 3 필드 추가 (v1)
- `src/main/kotlin/train/DataLoader.kt` — `BatchSource` interface + `MixedDataLoader` (v1)
- `build.gradle.kts` — 5개 task 추가 (`runEncodeWithExistingMeta`, `runTwoStageBaseTrainVec`, `runTwoStageITTrainVec`, `runTwoStageBaseV2TrainVec`, `runTwoStageITV2TrainVec`)

## 학습한 것

1. **BPE 학습 병렬화 가능**: countPairs/applyMerge가 word 단위 독립이라 partial map → final merge 패턴 적용. ~30× 가속.
2. **encode는 word-level 캐시가 결정적**: Zipf 분포 자연어에서 unique word 1/180 비율 → cache hit ratio 99%+ → ~수백× 가속.
3. **macOS jetsam 메모리 정책**: 32GB 시스템에서 JVM heap 12GB + sibling processes 합치면 SIGTERM(143/144) 위험. -Xmx 보수적 + skip-bin 옵션으로 회피.
4. **정제 단계의 동적 임계값**: 데이터마다 `_` 빈도가 자연어에서 일정한 cut-off 위치에 있어, 하드코딩 없이 임계값 자동 결정 가능.
5. **표현 다양성/정확성은 BASE 단계 의존**: IT는 형식 적응이지 새 표현 생성이 아님. 토픽 vocabulary는 BASE 데이터/모델 용량의 함수.
