# PikoGPT — Kotlin으로 구현한 미니 GPT

[nanoGPT](https://github.com/karpathy/nanoGPT)와 [micrograd](https://github.com/karpathy/micrograd)를
Kotlin으로 처음부터 다시 짠 교육용 프로젝트입니다. 외부 ML 라이브러리 없이 autodiff·텐서 연산·옵티마이저·
토크나이저까지 전부 in-repo로 구현되어 있습니다.

> ⚠️ 이 저장소의 코드/문서 대부분은 **Claude Code**와 **Gemini**가 작성했습니다. 사람이 직접 쓴 비중은 적습니다.

## 핵심 개념

PikoGPT는 **두 개의 병행 백엔드**를 가집니다.

| | Scalar (`Value` + `gpt/` + `train/Scalar*`) | Turbo (`turbo/`) |
|---|---|---|
| 데이터 단위 | `Value` 스칼라 객체(연산마다 노드 생성) | `TurboTensor` = `FloatArray + shape` |
| autograd | 동적 그래프 + 위상정렬 backward | 명시적 layer-별 forward / backward |
| 속도 | 71K param iter당 ~16초 | 1M param iter당 ~2.6초 |
| 가속 | 없음 | JDK 21 `jdk.incubator.vector` SIMD + ForkJoinPool 병렬 + KV cache |
| 모던 옵션 | LayerNorm + GELU | RMSNorm, SwiGLU, RoPE, GQA, qk-norm, fused QKV, z-loss |
| 목적 | **chain rule + transformer 수식 학습용 reference** | 실전 성능 |

스칼라 코드는 chain rule이 코드에 그대로 보이는 가장 단순한 micrograd 직계 포팅이고, turbo는 같은 수식을
SIMD + 명시적 backward로 다시 푼 실전용입니다. 두 백엔드는 서로를 대체하지 않고 **나란히** 존재합니다.

명명 규칙: 같은 패키지에 두 백엔드가 공존하므로 동명 클래스에 `Scalar`/`Turbo` 접두사를 붙입니다 —
`ScalarTrainer` vs `TurboTrainer`, `ScalarSampler` vs `TurboSampler` 등. micrograd 출처 클래스는
`Micrograd*` 접두사 (`MicrogradMLP`, `MicrogradNeuron`, ...).

## 빠른 시작

JDK 21 필요 (`jdk.incubator.vector` 모듈 사용).

```bash
# 빌드
./gradlew build

# 단위 테스트 (전체 ~1초, training/sampling 배관 검증 포함)
./gradlew test
```

### Scalar 백엔드 한 흐름 — 데이터 준비 → 학습 → 샘플링

```bash
./gradlew runAlphabetPrep   # data/alphabet/az.txt → train.bin/val.bin/meta.json
./gradlew runMiniTrainer    # ~10분, 알파벳 패턴 학습
./gradlew runSampler        # 최신 ckpt 자동 검색, 4개 프롬프트 샘플
```

전체 가이드: [docs/scalar-quickstart.md](docs/scalar-quickstart.md). 코드 자체를 읽으며 이해하고 싶다면
[docs/educational-walkthrough.md](docs/educational-walkthrough.md) (10단계 reading guide).

### Turbo 백엔드 — 실전 학습/샘플링

```bash
./gradlew runTinyHelenTrainTurbo            # 혼합 코퍼스 1M 모델 학습 (10k iter)
./gradlew runTinyHelenTrainTextbookTurbo    # textbook-only 6k iter
./gradlew runTinyHelenSampleTurbo           # 최신 turbo ckpt 샘플링
./gradlew runSamplePromptsFromFile \
          --args="<ckpt-dir> <prompts.txt>" # 커스텀 프롬프트 파일

# resume
./gradlew runTinyHelenTrainTurbo --args="resume"
./gradlew runTinyHelenTrainTurbo --args="20000 resume"
```

기타 스크립트: `runStoriesBpe`(BPE prep), `runStoryGenerator`(LM Studio 연동), `runBPETest`,
`runAnalyzeTokens`, `runDebugBPE`. 모두 `src/main/kotlin/`의 `fun main()` + `JavaExec` gradle 태스크입니다.

> NOTE: `application` 플러그인을 사용하지 않으므로 `./gradlew run`은 없습니다. 항상 `runXxx`를 쓰세요.
> `src/test/kotlin/`에는 `@Test`만 두고(전체 1초 이내), 오래 걸리는 학습/샘플링은 `src/main/kotlin/*Main.kt`로
> 둡니다.

## 디렉토리 구조

```
src/main/kotlin/
├── Value.kt                    # 스칼라 autodiff 핵심
├── grad/                       # Micrograd* MLP (가장 단순한 autodiff 예)
├── gpt/                        # 스칼라 백엔드 GPT
│   ├── ScalarPikoGPT.kt        # top-level model
│   ├── GPTConfig.kt            # 두 백엔드 공유 config
│   ├── ScalarTransformerBlock.kt
│   ├── ScalarCausalSelfAttention.kt
│   ├── ScalarFeedForward.kt
│   ├── ScalarLayerNorm.kt / Dropout / Linear / EmbeddingTable
│   └── Matrix.kt               # Value 2D — 스칼라 백엔드 유일 텐서 표현
├── train/
│   ├── ScalarTrainer.kt        # 스칼라 학습 루프
│   ├── ScalarAdamW.kt          # 스칼라 옵티마이저
│   ├── DataLoader.kt           # 두 백엔드 공유
│   ├── TrainConfig.kt          # 하이퍼파라미터 + 경로
│   ├── ScalarCheckpoint.kt / States.kt
│   ├── MiniTrainerMain.kt      # ★ Scalar quickstart 진입점
│   ├── TrainerMain.kt          # 옛 진입점 (deprecated)
│   └── experiments/            # ConvMix*/TinyHelen*/ThreeStage*/TwoStage*/Bench* — 30여 진입점 (모두 turbo)
├── sample/
│   ├── ScalarSampler.kt
│   ├── SamplerMain.kt          # ★ Scalar 샘플링 진입점
│   ├── SampleConfig.kt
│   ├── ChatTurbo.kt            # turbo ckpt 인터랙티브 대화
│   └── SamplePromptsFromFile.kt / SampleV4MergedSpaceSep3M.kt
├── data/
│   ├── CharBPE.kt              # char-level BPE (이전 이름 SimpleBPE)
│   ├── StoriesBpePrep.kt       # BPE 전처리
│   ├── AlphabetPrep.kt         # Scalar quickstart용 prep
│   ├── StoryGenerator.kt       # LM Studio 연동(선택)
│   └── MetaInfo.kt
├── turbo/                      # Turbo 백엔드
│   ├── TurboTensor.kt          # FloatArray + shape
│   ├── TurboSimdMath.kt        # JDK 21 Vector API helper
│   ├── ops/                    # 원자 연산 (forward + backward + 수식 주석)
│   ├── layer/                  # TurboLinear/LayerNorm/RMSNorm/MLP/SelfAttention/TransformerBlock/PikoGPT
│   ├── TurboAdamW.kt / TurboTrainer.kt / TurboSampler.kt
│   ├── TurboCheckpoint.kt / TurboKVCache.kt / TurboModelConfig.kt
│   ├── TurboParallel.kt        # ForkJoinPool 병렬 학습
│   └── bench/                  # MatMul shape별 마이크로벤치
└── util/FloatExtensions.kt

src/test/kotlin/                # @Test만, 전체 1초 이내
├── pipeline/FullPipelineTest.kt   # 합성 데이터 end-to-end 배관 검증
└── turbo/                         # 옵션별 회귀 (GQA, fused QKV, qk-norm, RMSNorm, z-loss, KV cache)
```

## 데이터/체크포인트 레이아웃

```
data/
├── alphabet/                   # Scalar quickstart (a-z 패턴 텍스트, ~6KB)
│   ├── az.txt                  # commit된 원본
│   ├── train.bin / val.bin     # runAlphabetPrep 산출물
│   └── meta.json
└── <dataset>/                  # 예: tinyhelen, three-stage-v4
    ├── train.txt / val.txt     # 분리 입력 (있으면 우선, vocab은 train에서만 학습 — leakage 차단)
    ├── stories.txt             # fallback (90:10 cut)
    ├── meta.json               # vocab + token<->id maps + BPE merges
    ├── train.bin / val.bin
    └── unique_words.txt        # optional vocab dump

model/                          # gitignored
└── <datasetName>/              # config.dataPath의 마지막 segment
    └── <expName>/              # 사람이 정한 실험 이름 (default "main")
        └── v0001/              # 4자리 zero-pad, 매 저장마다 +1
            ├── checkpoint.json
            ├── meta.json
            ├── model_weights.bin
            └── optimizer_state.bin   # Turbo만 (resume용)
```

**두 백엔드 동일 schema**입니다. `expName`은 사용자가 정한 식별자 — 같은 데이터셋을 공유하는 진입점만
unique 값을 명시하면 됩니다.

## 주요 학습 포인트

- **`Value.kt`** — 스칼라 autodiff 본체. `+ - * / pow ReLU GELU sigmoid` 오버로딩 + 위상정렬
  `backward()`. 코드를 한 번 읽으면 chain rule이 코드 그대로 보임.
- **`gpt/Matrix.kt`** — `Value` 2D 행렬에 `mapRows`/`zipWith` + `lastRow()`/`softmaxRows()`/`argMaxRows()`
  의미적 헬퍼.
- **`gpt/ScalarTransformerBlock.kt` / `ScalarCausalSelfAttention.kt`** — pre-LN → attention → residual →
  pre-LN → FFN → residual. attention forward는 5단계 헬퍼로 분해되어 있어 수식과 1:1 대응.
- **`turbo/layer/`** — 같은 수식을 SIMD + 명시적 backward로 다시 짠 버전. layer 파일을 forward / backward
  나란히 읽으면 chain rule을 수식 그대로 볼 수 있음.
- **`train/ScalarTrainer.kt` / `turbo/TurboTrainer.kt`** — 학습 루프(LR schedule, grad clipping, eval,
  checkpointing).
- **`data/CharBPE.kt`** — char pair encoding. 학습 시 옵션(`lowercase`, `useWordPreTokenize`,
  `splitSpaceAsToken`, `standardBpeScoring`)이 meta.json에 그대로 저장되어 샘플러가 학습과 정확히 같은
  토큰화를 재생.

## 디자인 원칙

- **Pure Kotlin, no ML libs.** autodiff·layer·optimizer·tokenizer 전부 in-repo.
- **Scalar autograd graph.** `Value`가 forward에서 동적 그래프를 만들고, `backward()`가 위상 역순으로 chain
  rule 적용.
- **Explicit layer backward (turbo).** autograd 없이 layer마다 forward/backward를 직접 짜서 SIMD에 맞춤.
- **kotlinx.serialization.** config·checkpoint·per-layer DTO가 `@Serializable`. 가중치는 `.bin` 별도.
- **CLI 통합 없음.** 모든 진입점은 `*Main.kt` + `runXxx` JavaExec 태스크. 테스트 디렉토리는 `@Test`만.

## 외부 의존성

- `org.jetbrains.kotlinx:kotlinx-serialization-json:1.5.0`
- `org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3`
- 테스트: `kotlin("test")` on JUnit Platform
- **JDK 21** (turbo의 `jdk.incubator.vector` 사용 — `--add-modules=jdk.incubator.vector`)

`StoryGenerator`만 외부 의존이 있습니다 — `http://127.0.0.1:1234`에서 `google/gemma-3-1b`를 서빙하는 LM
Studio. 다른 흐름엔 필요 없습니다.

## 더 읽어보기

- [docs/scalar-quickstart.md](docs/scalar-quickstart.md) — 데이터 → 학습 → 샘플링 한 흐름 가이드
- [docs/educational-walkthrough.md](docs/educational-walkthrough.md) — `Value.kt`부터 `ScalarTrainer.kt`까지
  10단계 코드 reading guide
- [docs/tinyhelen-training.md](docs/tinyhelen-training.md) — TinyHelen 코퍼스 학습 노트
- [docs/turbo-bench-results.md](docs/turbo-bench-results.md) — turbo MatMul/end-to-end 벤치
- [docs/three-stage-v4-recipe.md](docs/three-stage-v4-recipe.md), [docs/ccmc-v5-qa-plan.md](docs/ccmc-v5-qa-plan.md) — 코퍼스/실험 레시피

## 라이선스

MIT License
