# CLAUDE.md

Claude Code가 이 repo에서 작업할 때 알아야 할 컨벤션과 큰 그림. 디테일은 코드/`build.gradle.kts`/`docs/*`에 있다.

## Project

Kotlin 포트 of nanoGPT + micrograd. **외부 ML 라이브러리 없음** — autodiff/layer/optimizer/tokenizer 모두 in-repo. 교육용 + 실전 성능 두 마리 토끼.

## Conventions

- **`src/test/kotlin/`** = `@Test`만. 전체 suite **1초 이내** (필수). 실 학습/벤치는 main 소스셋의 `*Main.kt`로.
- **`src/main/kotlin/`** = 프로덕션 코드 + 실행 스크립트(`fun main()`). 통합 CLI 없음 — 각 진입점은 `build.gradle.kts`의 `runXxx` `JavaExec` task로 실행.
- **클래스명 접두사**로 백엔드 구분: `Scalar*` / `Turbo*` / `Micrograd*`. 같은 패키지 트리에 동명 클래스 공존 시 패키지 명시 없이 식별 가능 (`TurboTrainer` vs `ScalarTrainer`).
- **kotlinx.serialization**으로 config/checkpoint/per-layer state `@Serializable`. weight는 별도 `.bin`.
- **FQCN 피하고 import** 사용 (Kotlin style).

## 백엔드 3종 (병행, 서로 대체 아님)

| | 디렉토리 | 특성 | 선택 시점 |
|---|---|---|---|
| **Scalar** | `Value.kt` + `grad/` + `gpt/` + `train/Scalar*` | 교육용 micrograd 직계 포팅. autodiff graph. 71K params iter당 16s. | quickstart, 학습 알고리즘 읽기 |
| **Turbo** | `turbo/` | JDK 21 Vector API SIMD + ForkJoinPool + KV cache. **autograd 없이 layer마다 forward/backward 명시**. RMSNorm/SwiGLU/RoPE/GQA/qk-norm/fused QKV/z-loss 지원. 1M iter ~2.6s. | 실전 학습 (CPU) |
| **MPS Graph** | `mps/` + `src/main/objc/MpsGraphBridge.mm` + JNI | macOS Metal GPU. encodeToCommandBuffer fused step + prefetch + eval cache. 1M iter ~0.2s, 10M batch 64 iter ~0.085s. | macOS arm64 GPU 학습 |

자세한 mps backend 사양/한계: `docs/mps-graph-backend.md`.

## Build / Test / Run

```bash
./gradlew build                                       # 전체 빌드
./gradlew test                                        # 전체 test (1초 이내)
./gradlew test --tests "pipeline.FullPipelineTest"    # end-to-end 배관 smoke
./gradlew tasks --group=application 2>/dev/null \
  || ./gradlew tasks | grep '^run'                    # 전체 runXxx task 목록
./gradlew runXxx --args="..."                         # 개별 진입점
```

주의:
- `./gradlew run` 없음 (application 플러그인 안 씀).
- 모든 진입점은 `build.gradle.kts`의 `runXxx` task로 노출. description 확인.
- Resume: `--args="resume"` 또는 `--args="<maxIters> resume"` (turbo + mps graph 학습 진입점 공통).

## Memory 설정

| 작업 | -Xmx |
|---|---|
| `runBpe` | 12g |
| 5M+ params 학습/벤치 | 6~8g |
| 일반 turbo 학습 | 4~8g |
| 그 외 (대부분) | 2g |

## Toolchain 핵심

- **JDK 21** 필수 — turbo의 `jdk.incubator.vector` 모듈 (`--add-modules=jdk.incubator.vector`).
- Kotlin 1.9.0 (jvmTarget=21 미지원 → compile target 17, runtime JDK 21).
- macOS arm64에서만 mps graph 백엔드 dylib 빌드 가능 (`./gradlew buildMpsGraphLib`).

## Training Workflow (4 단계)

1. **Prep**: `runBpe` / `runAlphabetPrep` / `Ccmc*Prep` → `data/{dataset}/{train,val}.{txt,bin}` + `meta.json` (vocab + BPE merges + 플래그)
2. **Train**: 진입점 실행. 학습 옵션은 `TrainConfig` / `TurboTrainConfig` / `MpsGraphTrainConfig`.
3. **Checkpoint**: `model/{datasetName}/{expName}/v{NNNN}/` (`checkpoint.json`, `model_weights.bin`, `meta.json`, [turbo] `optimizer_state.bin`). 매 저장마다 v 번호 +1. `expName` default `"main"`.
4. **Sample**: `runSampler` (scalar) / `runChatTurbo` (REPL) / `runSamplePromptsFromFile` / `runInferenceApi` (HTTP).

자세한 quickstart: `docs/scalar-quickstart.md`. 추론 API 사양: `docs/inference-api.md`.

## Key files (시작점)

- `Value.kt`, `gpt/Matrix.kt` — scalar autodiff foundation.
- `gpt/ScalarPikoGPT.kt` + `gpt/GPTConfig.kt` + `train/ScalarTrainer.kt` — scalar 모델 + 학습.
- `turbo/TurboTensor.kt` + `turbo/layer/TurboPikoGPT.kt` + `turbo/TurboTrainer.kt` — turbo 모델 + 학습 (실전 성능).
- `data/CharBPE.kt` + `data/BpePrep.kt` — tokenizer.
- `mps/MpsGraphTrainer.kt` + `src/main/objc/MpsGraphBridge.mm` — MPS graph 학습 (자세한 건 `docs/mps-graph-backend.md`).
- `server/InferenceApiMain.kt` — Ktor 추론 API.

그 외 디렉토리/파일 구조는 코드 직접 탐색.
