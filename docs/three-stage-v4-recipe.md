# three-stage-v4 데이터셋 레시피 — dict → wiki → conv 3단계 curriculum

미취학(3-6세) 타겟 LM의 학습을 인간 학습 순서로 모사한 3단계 curriculum + dict 데이터 추가 + multi-replay 설계.

본 문서는 **데이터 준비(Stage A-C) 까지** 완료된 상태를 기록. 학습 코드(Stage D — `ThreeStage*TrainV4Vec.kt`, `TripleDataLoader`)는 후속 작업.

## 1. 결정 요약

### 핵심 변경
- **3-stage curriculum**: dict (정의 grounding) → wiki (백과 맥락) → conv (대화 / instruction)
- **신규 dict 데이터셋 추가**: Simple English Dict + WordNet 병합, 자연어 문단 변환
- **vocab 단일 학습**: dict + wiki + conv 합본(`shared/`)으로 BPE 한 번 학습 (vocab=2000, cased), 모든 단계가 같은 token ID 공간 공유
- **multi-replay**: Stage 3에서 dict와 wiki를 **별도 replay path**로 등록 (단순 합본 시 dict가 1:8.7로 묻히는 문제 해결)

### v3 대비 차이
| 측면 | v3 (two-stage) | v4 (three-stage) |
|---|---|---|
| 학습 단계 | base(=wiki) → it(=conv) | dict → wiki → conv |
| 정의 패턴 | wiki 본문 안에 한 번 등장 | dict 단계에서 명시적 grounding |
| forgetting 완화 | replay 0.30 (단일 path) | replay 0.30 (multi-path: dict + wiki) |
| 데이터 양 | 4.0M words (wiki only) | 4.4M words (dict 0.36M + wiki 4.0M) + conv |

### 사용자 결정 (확정)
1. **다중 의미 보존** — entry당 최대 5 의미, niche 첫 의미 단독 채택 회피
2. **자연어 문장 doc** — `<|bos|>\n# Word\n...\n<|eos|>`, 1 line = 1 doc, doc 내부는 리터럴 `\n`
3. **모든 디렉토리 형식 통일** — wiki/conv도 1 line = 1 doc + 리터럴 `\n` (정규식으로 변환)
4. **별도 replay path** — Stage 3에서 dict/wiki 비율 정밀 제어
5. **dict 단계 10 epochs** — 정의 패턴 internalize
6. **conv 단계 명명** — v3 "it" → v4 "conv"로 리네임 (= conversation, 의미 동일)

## 2. 코퍼스 사양

### 데이터 소스 (dict)

| # | 데이터셋 | 형식 | 역할 | 매칭률 |
|---|---|---|---|---:|
| 1 | `nightblade9/simple-english-dictionary` (`processed/filtered.json`) | JSON, 121,297 표제어 | 정의 메인 소스 | 13,521 / 24,505 (55.2%) |
| 2 | NLTK WordNet | Synset, 117,659 | hypernym(상위 개념) 보강 | 8,144 / 13,521 (60.2%) |
| ❌ | `KidiIT/Free--Dictionary-API` | — | **제외** (1 entry만 있는 데모 파일) | — |

### 화이트리스트 전략

**v3 코퍼스 빈도 ≥ 10인 lowercase 표제어** (= 24,505개)만 사전에서 추출 → 이미 가진 코퍼스의 어휘에만 정의를 부여 → low-freq 문제 직접 보완 + niche 어른용 어휘 자동 차단.

추가 필터:
- 1글자 단어(a, i) 제외 — 정의 의미 없음
- 인명 휴리스틱 — 정의가 "United States/English/British/..." + ("who", "born", "novelist", "actor", "president" 등) 시작 시 차단 (452개)
- WordNet hypernym blacklist — `blood group`, `associate degree`, `letter` (lemma 첫 synset이 다른 의미일 때)

### 학습용 디렉토리 토큰 통계

| 디렉토리 | docs | words | tokens (cased BPE 2k) | 10 epoch maxIters (batch=8, blockSize=48, gradAccum=2) |
|---|---:|---:|---:|---:|
| `dict/` | 12,169 (val 1,352) | ~362K | **863,415** | ~11,242 |
| `wiki/` | 8,048 (val 894) | ~3.5M | **7,485,510** | ~97,467 |
| `conv/` | (v3 it 그대로) | ~10.7M | **16,271,975** | ~211,874 |
| sum | — | ~14.6M | ~24.6M | — |

### dict 추가 효과 (v3 union 대비)

- dict가 wiki 저빈도(≤5) 어휘 **5,796개**에 명시적 정의 부여
- dict가 conv 저빈도 어휘 **1,914개**에 명시적 정의 부여
- only_dict (wiki/conv 어디에도 없음) 신규 어휘 **8,433개**
- dict 자체 hapax 비율 40.07% (wiki 44.48% 대비 낮음 — 다중 의미로 같은 단어 반복 노출)

### 의미 분포 (dict)

- merged entries: 13,521
- entry당 의미: p50=1, p90=4, max=5
- total definitions: 26,484 (entry당 평균 1.96 의미)
- definition 길이: p50=8 words, p90=18 words, max=61

## 3. 데이터 빌드 파이프라인

### Stage A — dict 데이터 빌드

```bash
# A.1 사전 다운로드 + WordNet 추출
python3 scripts/download_kid_dictionaries.py
# → data/dictionaries/{simple_dict.json (20MB, 121K entries),
#                      wordnet_mini.json (19MB, 147K lemmas)}

# A.2 표제어 병합 + kid-safe 화이트리스트 + WordNet hypernym 보강
python3 scripts/merge_kid_dictionaries.py
# → data/dictionaries/merged_entries.jsonl (13,521 entries)
#   schema: {word, meanings: [{pos, definition, synonyms, antonyms}], hypernym, v3_freq}

# A.3 자연어 doc 렌더링 + 90:10 split (seed=42)
python3 scripts/render_dict_docs.py
# → data/three-stage-v4/dict/{train.txt, val.txt}
#   format: 1 line = 1 doc, doc 내부는 리터럴 \n
```

### Stage B — wiki / conv 복사 + 형식 통일 + shared 합본

```bash
# wiki: v3 base를 그대로 복사 (다단락 본문) → 1 line=1 doc + 리터럴 \n으로 변환
cp data/two-stage-v3/base/{train,val}.txt data/three-stage-v4/wiki/
python3 scripts/inline_wiki_docs.py    # 정규식 <|bos|>...<|eos|> 매칭

# conv: v3 it를 그대로 복사 (이미 single-line, 변환 불필요)
cp data/two-stage-v3/it/{train,val}.txt data/three-stage-v4/conv/

# shared: dict + wiki + conv 합본 (BPE 학습용 전용)
cat data/three-stage-v4/dict/train.txt data/three-stage-v4/wiki/train.txt \
    data/three-stage-v4/conv/train.txt > data/three-stage-v4/shared/train.txt
cat data/three-stage-v4/dict/val.txt data/three-stage-v4/wiki/val.txt \
    data/three-stage-v4/conv/val.txt > data/three-stage-v4/shared/val.txt
```

### Stage C — BPE 학습 + 인코딩

```bash
# C.1 shared에서 cased BPE 학습 (vocab=2000, skip-bin)
./gradlew runBpe --args="data/three-stage-v4/shared 2000 cased skip-bin"
# → data/three-stage-v4/shared/meta.json (vocab=2000, merges=1908)
# 소요: ~3분 (84MB 코퍼스, 9.5 merges/sec)

# C.2 shared/meta.json으로 3 학습용 디렉토리 인코딩
./gradlew runEncodeWithExistingMeta --args="data/three-stage-v4/shared/meta.json data/three-stage-v4/dict"
./gradlew runEncodeWithExistingMeta --args="data/three-stage-v4/shared/meta.json data/three-stage-v4/wiki"
./gradlew runEncodeWithExistingMeta --args="data/three-stage-v4/shared/meta.json data/three-stage-v4/conv"
# 각 → train.bin, val.bin, meta.json 사본
```

## 4. 데이터 형식

모든 디렉토리의 `train.txt` / `val.txt`는 **1 line = 1 doc, doc 내부는 리터럴 `\n`**.

### dict doc 예시
```
<|bos|>\n# Form\nForm means create (as an entity).\nForm can also mean develop into a distinctive entity.\nForm can also mean assume a form or shape.\nA form can also be a printed document with spaces in which to write.\nA form can also be (biology) a group of organisms within a species that differ in trivial ways from similar groups.\n\nSimilar: make, create.\n<|eos|>
```

규칙:
- 첫 의미: pos=Noun → "An/A {word} is {def}.", Verb/Adj/Adv → "{Word} means {def}."
- 추가 의미: "An/A {word} can also be {def}." 또는 "{Word} can also mean {def}."
- hypernym (Noun에만): "An/A {word} is a/an kind of {hypernym}."
- syns/ants: 본문과 빈 줄(리터럴 `\n\n`)로 분리, 형식 `Similar: ...`, `Opposite: ...` (단복수 통일)

### wiki doc 예시
```
<|bos|>\n# Grand Canyon\n\nThe Grand Canyon is a famous canyon in Arizona, ...\nThe Grand Canyon is 277 miles ...\n<|eos|>
```

### conv doc 예시 (multi-turn)
```
<|bos|>And they all lived happily ever after.<|turn|>Yes! I love when the dragon ...<|turn|>...<|eos|>
```

## 5. 학습 흐름 (Stage D — 후속 작업)

### 디렉토리별 역할
- `dict/`, `wiki/`, `conv/` — 각 단계의 primary
- `shared/` — BPE 학습 전용. 학습 시 사용 안 함

### Stage 흐름
| Stage | trainer | initFrom | dataPath | replay |
|---|---|---|---|---|
| 1 | `ThreeStageDictTrainV4Vec` | scratch | `data/three-stage-v4/dict` | 없음 |
| 2 | `ThreeStageWikiTrainV4Vec` | pretrain_weights ← Stage1 best | `data/three-stage-v4/wiki` | `dict/`, ratio=0.30 |
| 3 | `ThreeStageConvTrainV4Vec` | pretrain_weights ← Stage2 best | `data/three-stage-v4/conv` | **dict/** & **wiki/** 별도 path (multi-replay) |

### Multi-replay 필요성

dict + wiki 단순 cat 시 토큰 비율 1:8.7 → random offset sampling 결과 dict 3% / wiki 27% / conv 70%로 dict 묻힘.

별도 path로 두면 비율 자유 제어:
- conv 70% + dict 15% + wiki 15% (dict/wiki 균형)
- conv 70% + dict 11% + wiki 19% (자연 oversampling ×5 분포 동일)

구현 필요 사항 (Stage D):
- `train/DataLoader.kt` — `TripleDataLoader` (또는 multi-replay 일반화)
- `train/TrainConfig.kt` — `replayDataPath2` / `replayRatio2` (또는 list)
- `train/experiments/ThreeStage*TrainV4Vec.kt` — 3개 entry point
- `build.gradle.kts` — 3개 task

## 6. 영향받지 않은 파일

- `data/two-stage-v3/*` — 보존 (v4는 독립적, conflict 없음)
- 기존 Kotlin 학습/샘플링 코드 — Stage D에서 추가만, 기존 코드 변경 없음

## 7. 산출물 트리

```
data/dictionaries/
├── simple_dict.json         (20MB, 121K entries — Simple English Dict 원본)
├── wordnet_mini.json        (19MB, 147K lemmas — NLTK WordNet 추출)
└── merged_entries.jsonl     (13,521 entries — 화이트리스트 + 다중 의미 병합)

data/three-stage-v4/
├── README.md                (폴더 구조 + multi-replay 설계 요약)
├── dict/                    (Stage 1 학습 데이터)
│   ├── train.txt val.txt
│   ├── train.bin val.bin
│   └── meta.json
├── wiki/                    (Stage 2 학습 데이터, 1 line=1 doc 변환됨)
│   ├── train.txt val.txt
│   ├── train.bin val.bin
│   └── meta.json
├── conv/                    (Stage 3 학습 데이터, v3 it 그대로)
│   ├── train.txt val.txt
│   ├── train.bin val.bin
│   └── meta.json
└── shared/                  (BPE 학습 전용)
    ├── train.txt val.txt
    ├── meta.json            (vocab=2000, merges=1908, cased)
    └── unique_words.txt
```

신규 스크립트:
- `scripts/download_kid_dictionaries.py`
- `scripts/merge_kid_dictionaries.py`
- `scripts/render_dict_docs.py`
- `scripts/inline_wiki_docs.py` — wiki 다단락 → 1 line=1 doc + 리터럴 `\n` 변환 (정규식 `<|bos|>...<|eos|>` 매칭)
- `scripts/analyze_v4_low_freq.py` — v4 단어 분포 진단 (analyze_v3와 동일 cased regex, dict 보강 효과 측정)

## 8. 결정 이력

1. **dict 데이터셋 후보 검토**:
   - Simple English Dict (filtered.json) — 121K entries, kid-safe 명목이지만 사실 부적절 단어만 제거. 본격 정의 사전.
   - Free Dictionary API (KidiIT) — 다운로드해 보니 entry 1개(데모). **제외**.
   - WordNet — 117K synsets, hypernym 보강에만 사용.

2. **121K → 13K 필터링**: v3 코퍼스 빈도 ≥ 10 화이트리스트로 5만 → 13,521 (Simple Dict 매칭률 55%).

3. **단일 의미 → 다중 의미 보존**: 첫 의미만 채택 시 niche 의미가 첫 번째인 단어("aa"=lava, "ab"=11번째 달)로 잘못된 학습. 사용자 요청으로 max 5 의미 보존.

4. **doc 형식 다단락 → 1 line=1 doc + 리터럴 `\n`**: 모든 디렉토리 통일. wiki는 정규식 `<|bos|>...<|eos|>` 매칭으로 변환 (split("\n\n") 방식은 doc 안 단락도 자르는 버그 있음).

5. **base/ 합본 → multi-replay**: dict+wiki 단순 cat은 1:8.7로 dict 묻힘. 사용자 결정으로 dict/wiki 별도 replay path. base/ 디렉토리는 제거.

6. **명명 it → conv**: v4는 새 시작이라 "instruction-tuning" 약어 "it" 대신 의미 명확한 "conv"(conversation)로 리네임.

## 9. 후속 작업 (Stage D)

다음 세션에서 진행:
- `TripleDataLoader` (or multi-replay 일반화) — `train/DataLoader.kt`
- `TrainConfig` 다중 replay 필드 — `train/TrainConfig.kt`
- 3개 trainer entry point — `train/experiments/ThreeStage{Dict,Wiki,Conv}TrainV4Vec.kt`
- 3개 gradle task — `build.gradle.kts`
- Stage 3 dict/wiki replay 비율 결정 (현재 후보: 15%/15% 또는 11%/19%)
- 학습 + 단계별 ckpt 평가 (정의 패턴 출력 / 백과 ppl / 대화 ppl)
- v3 baseline (TwoStageBaseTrainV3Vec → TwoStageITTrainV3Vec) 대비 비교
