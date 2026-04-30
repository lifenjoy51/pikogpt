# base-v3 데이터셋 레시피 — vital L1-L4 기반 BASE + IT 전이 학습 계획

Two-stage 학습의 BASE 단계용 코퍼스. **simplewiki vital articles Level 1-L4 단독 사용**.
IT 단계는 기존 `data/two-stage-v2/it-v2/` 그대로 유지.

## 1. 결정 요약

- 채택: **simplewiki vital articles Level 1-L4 전체 (8,942 docs, ~4.0 M words, ~5.6-6.4 M tokens)**
  - L4 11개 카테고리 모두 포함 (People 카테고리 ~1,580 docs 포함). 어른용 specific 인물(Hegel, Faulkner 등) 노이즈는 그대로 학습 신호로 흡수.
- 산출 파일: `data/simplewiki/simplewiki_vital_corpus.jsonl` 에서 `level <= 4` 필터
- BPE: **IT-V2 vocab 재사용** (`EncodeWithExistingMeta.kt`)
- IT 단계: `data/two-stage-v2/it-v2/` 그대로
- **추가 정제 없음**: B 가독성 컷 / D LLM 5세 적합 분류 / L5 선별 / simplify 모두 본 계획에서 적용 안 함

### 결정 변경 이력
1. **base-v2 (TinyHelen leaner/100M) 폐기** — wiki/textbook/book/web 4-shard 코퍼스가 mask placeholder 9.35% 폭증, vocab 결함 다수 (§부록)
2. **simplewiki 전체 (153K docs / 30M words / 42M tokens) 검토** — 깨끗하지만 4-6세 학습용으로는 niche 표제어가 다수 (예: 마이너 도시·인물·niche 학술)
3. **vital L1-L4 단독으로 결정** — Wikipedia 커뮤니티가 합의한 핵심 표제어 (Level 1: 10개·Level 2: 100개·Level 3: 1,000개·Level 4: 10,000개)만 사용. 5세 백과사전과 어울리는 vital 지식의 핵심 — 행성·동물·식물·기초 수학·기초 과학·세계 지리·역사 인물·기초 사회 개념·예술 명작 등.
4. **L5는 폐기** — Level 5 (50,000개)는 매칭 100건 샘플 결과 80-90%가 specific 인물·작품·niche 학술로 5세에는 너무 마이너 (예: `Hentai`, `Sexual fetishism`, `Diophantine equation`, `Anthony Fantano`).

### 가설 — 작은 BASE에 대한 IT 전이 보완
1M-param vec 모델 capacity가 작아 데이터 부족(Chinchilla 기준 권장의 1/4)에도 over-train으로 어느 정도 보완 가능. IT-V2가 dialogue 형식·일상 톤을 채워주는 구조이므로, 핵심 백과사전 사실(행성·국가·동식물·인물)이 BASE에 들어있기만 하면 IT가 형식 변환을 수행함. 이 가정 하에 vital L1-L4 단독을 BASE로 사용.

## 2. 코퍼스 사양

| Level | docs | words | avg w/doc | 매칭률 |
|---|---:|---:|---:|---:|
| L1 (인류 지식 최상위 10) | 10 | 41,570 | 4,157 | 100.0% |
| L2 (기초 100) | 88 | 130,555 | 1,484 | 97.8% |
| L3 (확장 1,000) | 881 | 707,654 | 803 | 97.5% |
| L4 (구체 10,000) | 7,963 | 3,132,473 | 393 | 89.7% |
| **TOTAL** | **8,942** | **~4.01 M** | — | — |

- chars: ~24 MB
- 추정 BPE 토큰 (vocab 2k, ratio 1.4-1.6/word): **5.6 - 6.4 M tokens**

### L4 카테고리 분포
| Category | docs | words |
|---|---:|---:|
| Biology and health sciences | 3,567 | 973K |
| Geography | 3,701 | 880K |
| People | 1,580+ | ~600K |
| Physical sciences | ~900 | ~320K |
| Society and social sciences | ~700 | ~250K |
| Arts | ~530 | ~250K |
| History | ~460 | ~260K |
| Technology | ~560 | ~190K |
| Everyday life | ~395 | ~160K |
| Philosophy and religion | ~325 | ~150K |
| Mathematics | ~228 | ~70K |

(L4만의 분포; L1-L3는 위계상 더 일반적인 표제어로 별도 카테고리 부여 안 함)

## 3. 데이터 정제 파이프라인

### 3.1 단계 다이어그램

```
[simplewiki dump XML.bz2]
       │
       ▼
WikiExtractor (--json, --no-templates, 4 procs)
       │
       ▼
[data/simplewiki/extracted/AA/wiki_*]   raw articles 390,369
       │
       ▼
clean_simplewiki_v2.py
  ├ title 강조 (text 첫머리 "# {title}\n\n")
  ├ redirect target 본문 가져옴 (chain 5단계, raw에 없으면 0건)
  ├ 하단 메타 5종 컷 (External links / References / Bibliography / Sources / Citations)
  ├ 마크업 제거 (공백 치환 → multi-space 정리)
  ├ 단락 50단어 컷 제거
  ├ is_list_only 제거
  ├ 본문 50자 미만만 컷 (정말 빈 페이지 방어)
  ├ 'may refer to' disambig 컷
  └ title-lower dedup만 (text MD5 dedup 제거 — 1글자 차이 oversensitive)
       │
       ▼
[data/simplewiki/simplewiki_clean.jsonl]   270,795 articles (v1: 153,967)
       │
       ▼
parse_vital_titles.py
  └ en wiki Vital Articles Level 1-L4 raw wikitext에서 표제어 추출
[data/external/vital_articles/vital_titles.json]   10,001 entries (L1+L2+L3+L4)
       │
       ▼
resolve_vital_titles.py (SKIP_EN=1)
  ├ exact (case-insensitive) match
  └ simple wiki API redirect 해소
[data/external/vital_articles/vital_titles_resolved.json]
       │
       ▼
recover_vital_from_raw.py  +  expand_vital_matches.py
  ├ raw extracted에서 cleaner v1이 컷한 vital 표제어 복구 시도
  └ disambiguator/comma/punctuation normalize 변형 매칭
  (v2 cleaner가 이미 관대해 raw_recovery 추가 0, normalized variants 매칭 590)
[vital_titles_resolved.json 갱신]
       │
       ▼
build_vital_corpus.py
[data/simplewiki/simplewiki_vital_corpus.jsonl]   30,311 docs (L1+L2+L3+L4+L5)
       │
       ▼  level <= 4 필터
       │
[BASE corpus: 8,942 docs / ~4.0 M words]
```

### 3.2 cleaner v2의 v1 대비 변경
| 항목 | v1 | v2 |
|---|---|---|
| title | 별도 필드 + 첫 줄 자름 | `# {title}\n\n` 본문에 추가, 자르기 제거 |
| redirect | drop | target raw 본문 가져옴 (chain 5) |
| 하단 메타 | 8종 컷 (See also/Notes/Further reading 포함) | 5종만 컷 (`External links / References / Bibliography / Sources / Citations`) |
| 마크업 치환 | 빈 문자열 (단어 붙음 버그) | 공백 치환 후 multi-space 정리 |
| 단락 50단어 컷 | 적용 | 제거 |
| `is_list_only` | 60% bullet/짧은라인 | 제거 |
| 본문 최소 길이 | 200자 | 50자 |
| text MD5 dedup | 적용 | 제거 |

이 변경으로 false cut 다수 회복 — Cleopatra(147c), Hong Kong(3,642c), Archaeology(6,884c), University(3,650c), Circle(3,956c) 등.

### 3.3 vital matching 결과
| 단계 | 매칭 수 | 누적 |
|---|---:|---:|
| exact (lowercase) | 28,185 | 28,185 |
| simple wiki API redirect | +2,492 | 30,677 |
| normalized variants (disambig/comma/punct) | +590 | 31,267 |
| **L1-L5 합** | **31,267 / 49,929 (62.6%)** | — |

L1-L4만: 9,054 매칭 → corpus dedup 후 **8,942 docs**.

## 4. 데이터 위치

```
data/simplewiki/
├── extracted/AA/wiki_*                          wikiextractor raw (390,369 articles)
├── simplewiki_clean_v1_legacy.jsonl             v1 cleaner 산출 보존 (153,967 docs)
├── simplewiki_clean.jsonl                       v2 cleaner 산출 (270,795 docs)
├── simplewiki_vital_extras.jsonl                cleaner 컷 + raw 복구 산출 (현 0건)
└── simplewiki_vital_corpus.jsonl                ★ 통합 vital 코퍼스 (L1-L5, 30,311 docs)
                                                  학습 시 level <= 4 필터해서 사용

data/external/vital_articles/
├── level{1,2,3}.wiki, level4/{11}.wiki          en wiki vital raw wikitext
├── vital_titles.json                            10,001 표제어 + level + category + subcategory
├── vital_titles_resolved.json                   simplewiki 매핑 결과
├── vital_simplewiki_titles.txt                  TSV (L{n}\t{cat}\t{sub}\t{title}\t{method})
└── vital_recovery_report.json                   raw 복구 상세

data/two-stage-v3/base-v3/                       (학습 단계에서 생성 예정)
├── train.txt / val.txt                          90:10 split, doc 단위 셔플
└── meta.json / train.bin / val.bin              IT-V2 vocab 재사용

data/two-stage-v2/it-v2/                         IT 단계, 변경 없음
└── meta.json / train.bin / val.bin
```

## 5. 코퍼스 record 형식

각 라인 = self-contained:
```json
{
  "title": "Sun",
  "text": "# Sun\n\nThe Sun is a star in the middle of the Solar System...",
  "level": 2,
  "category": null,
  "subcategory": null,
  "source": "clean"
}
{
  "title": "Marie Curie",
  "text": "# Marie Curie\n\nMarie Curie was a Polish-French physicist...",
  "level": 4,
  "category": "People",
  "subcategory": null,
  "source": "clean"
}
```

`level` 1-4 / `category` (L4 sub-page 11종) / `subcategory` (L5만, 본 BASE에선 사용 안 함) / `source` (`clean`).

## 6. BPE 및 bin 인코딩

### 6.1 BPE vocab 결정 — IT-V2 재사용
- IT-V2 vocab (2,048 tokens) 재사용 권장 (`EncodeWithExistingMeta.kt`)
- 이유: BASE → IT 전이 시 token 분포 일관성 확보 (vocab mismatch 방지)
- 단점: IT-V2가 dialogue 코퍼스 기반이라 백과사전 어휘에 sub-optimal — 일부 단어가 sub-token 다수로 쪼개질 수 있음. 그러나 base-v2 → IT-V2 전이도 같은 vocab으로 진행했으므로 일관성이 더 중요.

### 6.2 train/val split
- doc 단위 90:10 (random seed 고정 — 권장 `seed=42`)
- 라인당 1 doc, `<|bos|>{text}<|eos|>` 래핑

### 6.3 산출
```
data/two-stage-v3/base-v3/
├── train.txt          ~3.6 M words (90%)
├── val.txt            ~0.4 M words (10%)
├── meta.json          IT-V2 vocab 복제 (2,048 tokens)
├── train.bin          ~5.0 - 5.7 M tokens
└── val.bin            ~0.6 - 0.7 M tokens
```

### 6.4 빌드 명령 (예정)
```bash
# 1) corpus 재구성 (vital L1-L4 추출 + train/val split + bos/eos wrap)
python scripts/build_base_v3_train_val.py \
    --input data/simplewiki/simplewiki_vital_corpus.jsonl \
    --max-level 4 \
    --output-dir data/two-stage-v3/base-v3 \
    --val-frac 0.10 --seed 42

# 2) BPE 인코딩 (IT-V2 vocab 재사용)
./gradlew runEncodeWithExistingMeta \
    --args="data/two-stage-v3/base-v3 data/two-stage-v2/it-v2/meta.json"
```

## 7. BASE 학습 (vec 백엔드)

### 7.1 진입점 (예정)
- 신규: `src/main/kotlin/train/experiments/TwoStageBaseV3VitalTrainVec.kt`
- 모델: 1M-param vec backend (기존 `TwoStageBaseV2TrainVec.kt` 형상 그대로)
- 데이터: `data/two-stage-v3/base-v3/{train,val}.bin`
- 체크포인트: `model/base-v3/vec/${paramCount}/v0001/`

### 7.2 학습 hyperparameters
```kotlin
TrainConfig(
    dataPath = "data/two-stage-v3/base-v3",
    modelDir = "model",
    embeddingDimension = 256,
    numberOfLayers = 6,
    numberOfHeads = 8,
    maxSequenceLength = 512,           // 256 → 512 (long-context 흡수, simplewiki 본문 끝까지)
    dropout = 0.15f,                   // 0.1 → 0.15 (multi-epoch overfit 완화)
    batchSize = 64,
    learningRate = 6e-4f,
    minLearningRate = 6e-5f,
    warmupIters = 100,
    maxIters = 10_000,                 // 넉넉한 상한 (실제는 early stop으로 1-3k에서 동결)
    lrDecayIters = 10_000,             // cosine decay가 maxIters와 같이 진행
    weightDecay = 0.1f,
    gradClip = 1.0f,
    evalInterval = 100,                // 250 → 100 (early stop 정밀도↑)
    evalIters = 50,
    alwaysSaveCheckpoint = false,      // best val ppl 갱신 시만 저장
    earlyStopPatience = 5,             // val ppl best 대비 5회 연속 1% 이상 악화 → 중단
    backend = "vec",
)
```

#### iter 수 산정 근거
- batch 64 × seq 512 = **32,768 tokens/iter**
- 5.6 M tokens / 32,768 = **~171 iter/epoch**
- maxIters는 **상한**일 뿐. early stop이 best val ppl 시점에서 자동 동결.
- 예상 동결 지점: **5-15 epoch (~850-2,500 iter)**. 데이터 작아 그 이상 가면 overfit 명확.
- `maxIters=10,000` (~58 epoch 상한) 는 절대 안 닿을 보수적 천장. cosine decay도 동일 길이로 두어 학습 도중 lr이 너무 빨리 minLR로 떨어지지 않게 함.

#### Chinchilla 비교
- 권장: 1M params × 20 = 20 M tokens
- 실제: 5.6-6.4 M tokens (Chinchilla 1/3-1/4)
- multi-epoch + early stop으로 capacity-bound 한계까지 학습. **train ppl ↓ vs val ppl ↑ 분기점에서 ckpt 동결**.

### 7.3 검증 시점
- 매 250 iter eval
- val ppl이 직전 best 대비 0.5% 이상 악화 → early stop
- best val ppl ckpt를 IT 단계로 넘김

### 7.4 기대 perplexity
- BASE val ppl 기대치: ~30-50 추정 (TinyHelen-textbook 6k iter 기준 ~40 참고)
- 코퍼스가 작고 vocab 일치(IT-V2 재사용)하므로 base-v2 (49.5M tokens) 대비 ppl이 약간 낮을 가능성 (그러나 generalization은 약할 수 있음)

## 8. IT 전이 학습

### 8.1 진입점 (예정)
- 기존 `TwoStageITV2TrainVec.kt` 그대로 사용 가능 — `init_from` 만 base-v3-vital ckpt로 변경
- 또는 신규 wrapper: `TwoStageITV2OnBaseV3VitalTrainVec.kt`

### 8.2 IT 학습 hyperparameters
- 기존 IT-V2 학습 그대로
- 학습률: BASE의 1/2 (e.g. 3e-4)
- maxIters: 4-6k (IT 코퍼스 작아 빠르게 수렴)
- vocab: BASE와 동일 (IT-V2 vocab 재사용 보장)

### 8.3 검증 지표
- IT val ppl
- 정성 평가 — 부모-자녀 dialogue prompt에 모델 응답
- 비교: dialogues-a510 base-v2 → IT-V2 전이 결과 (49.5 M base 토큰 / 6,000 iter / IT val ppl ~XX)

## 9. 검증 계획 (BASE → IT 전이 후)

| 지표 | 측정 방법 | 비교 대상 |
|---|---|---|
| BASE val ppl | base-v3-vital 학습 중 best | base-v2 BASE 단계 ppl |
| BASE → IT val ppl | IT 전이 학습 중 best | dialogues-a510 base-v2 → IT-V2 전이 ppl |
| 샘플 텍스트 정성 | runSamplePromptsFromFile | 5세 친화 prompt 5종 (가족·동물·자연 etc) |
| Knowledge probing | "What is Saturn?" 등 vital 표제어 직접 prompt 응답 | base-v2 동급 |

기대치:
- **BASE val ppl**: 비슷하거나 약간 좋음 (코퍼스 깨끗하고 vocab 일치)
- **IT 후 val ppl**: base-v2 대비 약 5-15% 악화 추정 (BASE 토큰 1/8 → IT가 dialogue 형식만 학습 가능)
- **knowledge probing**: vital 핵심 사실(행성·동물·국가)에 강함, niche fact에 약함 — 의도된 trade-off

## 10. 알려진 trade-off

본 계획에서 데이터 추가·후처리는 **하지 않음**. 결과가 나쁘면 별도 단계로 확장 검토.

### Trade-off (감수)
1. **데이터 양 부족 (Chinchilla 1/4)** — multi-epoch + capacity-bound 가정으로 보완. 학습 결과로 검증.
2. **L4 People 카테고리 1,580 docs 포함** — 일부 specific 인물(Hegel, Faulkner, Carlos Slim 등)이 5세 영역 외. **그대로 학습 신호로 흡수**. D LLM 분류기 후처리는 적용 안 함.
3. **L4 일부 부적절 표제어 포함** — Wikipedia가 일반 백과사전이라 `Pedophilia`, `Sexual fetishism` 등이 vital list에 들어있음. 본 계획에서는 그대로 진입 (~5-10건). 학습 데이터 비율 0.1% 미만이라 모델 영향 미미하다고 판단.
4. **L5 미사용으로 specific knowledge 약함** — 마이너 도시·인물·niche 학술은 모델이 모르게 됨. 5세 BASE 의도에 부합.
5. **simplewiki 비격식 web/구어 부재** — IT-V2가 dialogue로 일부 보완.

### Fallback 옵션 (이번 계획 범위 외, 결과 보고 결정)
BASE → IT 전이 후 base-v2 대비 ppl이 크게 악화되면(예: 30% 이상) 별도 단계로 다음 중 하나 시도. 모든 옵션은 추가 스크립트를 새로 작성해야 함:
- **L5 선별 추가** — Animals/Cities/Astronomy 같은 5세 친화 sub-category만 → +2-4 M tokens
- **L1-L4 + LLM simplify (본문 5세 변환)** — 토큰 수 ±20%, 어휘 일관성↑
- **simplewiki 전체 회귀** — base-v3 simplewiki 전체 (42 M tokens)로 회귀
- **B+D 후처리** — 본문 가독성 메트릭(`avg_word_len`·`avg_sent_len`·`long_word_ratio`) 임계 컷 + LM Studio LLM 분류기로 마이너 항목 컷 (false REMOVE 위험)

## 11. 다음 실행 절차

### 11.1 코퍼스 빌드 (Python — 신규 스크립트)
```bash
python scripts/build_base_v3_train_val.py \
    --input data/simplewiki/simplewiki_vital_corpus.jsonl \
    --max-level 4 \
    --output-dir data/two-stage-v3/base-v3 \
    --val-frac 0.10 --seed 42
```

### 11.2 BPE 인코딩 (Kotlin)
```bash
./gradlew runEncodeWithExistingMeta \
    --args="data/two-stage-v3/base-v3 data/two-stage-v2/it-v2/meta.json"
```

### 11.3 BASE 학습 (Kotlin, 신규 진입점)
```bash
./gradlew runTwoStageBaseV3VitalTrainVec
# 또는 resume:
./gradlew runTwoStageBaseV3VitalTrainVec --args="resume"
```

### 11.4 IT 전이 학습
```bash
./gradlew runTwoStageITV2OnBaseV3VitalTrainVec
```

### 11.5 검증
```bash
./gradlew runSamplePromptsFromFile \
    --args="model/base-v3/vec/<param>/v0001 prompts/vital-base-prompts.txt"
```

## 12. 산출물 체크리스트

다음 작업 시 이 항목들을 모두 산출해야 완료:

- [ ] `scripts/build_base_v3_train_val.py` 신규
- [ ] `data/two-stage-v3/base-v3/{train,val}.txt` 생성
- [ ] `data/two-stage-v3/base-v3/{meta.json,train.bin,val.bin}` 생성 (BPE 인코딩)
- [ ] `src/main/kotlin/train/experiments/TwoStageBaseV3VitalTrainVec.kt` 신규
- [ ] `build.gradle.kts` 에 `runTwoStageBaseV3VitalTrainVec` 태스크 추가
- [ ] `src/main/kotlin/train/experiments/TwoStageITV2OnBaseV3VitalTrainVec.kt` 신규 (또는 기존 `TwoStageITV2TrainVec.kt` 의 init_from 인자화)
- [ ] BASE 학습 → ckpt 동결
- [ ] IT 전이 학습 → ckpt 동결
- [ ] 샘플링 결과 정성 평가
- [ ] base-v2 동급 모델과 ppl 비교

## 부록 — base-v2 폐기 사유 (보존)

품질을 두 축으로 측정 (mask placeholder 농도 / vocab 결함):

| Sub-corpus | docs | mask% avg | mask ≥5% 문서 | 비고 |
|---|---:|---:|---:|---|
| TinyH wiki | 10,197 | **9.35%** | **61.66%** | placeholder 폭증 — 학습 신호 1/12이 익명화 코드 |
| TinyH textbook | 7,667 | 1.07% | 5.66% | 깨끗하나 iOS/ObjC 코드 17종 누수 |
| TinyH book | 33 | 1.78% | 3.03% | 33권 / doc당 80K tokens — 보일러플레이트 집중 |
| TinyH web (4sh) | 32,642 | 3.36% | **23.67%** | `*website` 합성어 209종, LLM 생성 결함 93종, placeholder concat 156종 |

대안으로 측정한 simplewiki:

| Source | docs | mask% avg | mask ≥5% | 실 vocab 노이즈 |
|---|---:|---:|---:|---:|
| simplewiki_clean (v1) | 153,967 | **0.98%** | **4.80%** | ~50 / 30M (≈0%) |

base-v2 폐기 → simplewiki 전체 검토 → vital L1-L4로 좁힘 (본 문서).

---

작성: 2026-04-30. 결정 근거: vital articles 매칭 분석 §1, §2 / cleaner v2 §3.2 / 1M-param 모델 capacity 가설 §1.
