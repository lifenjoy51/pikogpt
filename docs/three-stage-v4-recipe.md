# three-stage-v4 데이터셋 레시피 — dict → wiki → conv 3단계 curriculum

미취학(3-6세) 타겟 LM의 학습을 인간 학습 순서로 모사한 3단계 curriculum + dict 데이터 추가 + multi-replay 설계.

본 문서는 **데이터 준비(Stage A-C) 까지** 완료된 상태를 기록. 학습 코드(Stage D — `ThreeStage*TrainV4Vec.kt`, `TripleDataLoader`)는 후속 작업.

## 0. 선결 조건

### 본 레시피의 범위
**입력 (원본 소스)**:
- dict: 웹에서 직접 다운로드 (Simple English Dict + NLTK WordNet)
- wiki: 웹에서 직접 다운로드 (simplewiki XML dump → vital articles 정제)
- conv: 웹에서 직접 다운로드 (HuggingFace `styfeng/TinyDialogues` age-5+age-10) **또는** git에 추적 중인 보존본 `data/{train,val}.txt.gz`

**출력**: `data/three-stage-v4/{dict,wiki,conv,shared}/` 완성 (학습용 `.bin` 포함)

**책임**: 3종 데이터를 원본 웹 소스부터 가져와 1 line=1 doc 형식으로 통일 + BPE + 인코딩까지. v3와 독립적으로 self-contained.

### 환경
- **Python 3.10+** (3.11 검증). `nltk` 3.9.x — `wordnet`는 다운로드 스크립트가 자동 호출.
- **JDK 17+** (Temurin 17). **Gradle 8.x** (wrapper로 자동).
- **디스크 공간**: 약 1.5 GB
  - simplewiki dump 250 MB (압축) + WikiExtractor 산출 ~600 MB
  - dict 다운로드 40 MB + 후처리 산출물 170 MB
  - TinyDialogues raw 30 MB (옵션 2)

### 결정 파라미터 (재현용 고정값)

| 파라미터 | 값 | 위치 | 역할 |
|---|---|---|---|
| `FREQ_THRESHOLD` | 10 | `merge_kid_dictionaries.py` | 입력 코퍼스 빈도 화이트리스트 컷 |
| `MAX_MEANINGS` | 5 | `merge_kid_dictionaries.py` | entry당 최대 의미 |
| `VAL_FRAC` | 0.10 | `render_dict_docs.py` | dict train/val split |
| `SEED` | 42 | `render_dict_docs.py` / `build_base_v3_train_val.py` | shuffle / split 재현 |
| `--max-level` | 4 | `build_base_v3_train_val.py` | vital articles L1-L4까지 |
| `vocab` | 2000 | `runBpe` 인자 | BPE vocab 크기 |
| BPE flags | `cased skip-bin` | `runBpe` 인자 | 대소문자 보존 + shared bin 생략 |

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

### Stage B — wiki / conv 원본 빌드 + 형식 통일

```bash
# 사전: dict/는 Stage A.3에서 자동 생성. 나머지는 cp/cat 전에 mkdir.
mkdir -p data/three-stage-v4/{wiki,conv,shared}
```

#### B.1 wiki — simplewiki XML dump → vital articles L1-L4 정제 → inline

상세 결정 이력은 `docs/base-v3-recipe.md` 참조. 본 v4용으로 필요한 절차만 집약:

```bash
# (1) simplewiki dump 다운로드 (~250 MB 압축)
mkdir -p data/simplewiki
wget -O data/simplewiki/simplewiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2

# (2) WikiExtractor로 raw articles 추출 (~390K)
#     로컬 wrapper(`scripts/_wikiextractor_local/`) 또는 `pip install wikiextractor`
python3 -m wikiextractor.WikiExtractor \
  --json --no-templates --processes 4 \
  --output data/simplewiki/extracted \
  data/simplewiki/simplewiki-latest-pages-articles.xml.bz2

# (3) cleaner v2 — title 강조, redirect 본문 회수, 메타 컷, dedup
python3 scripts/clean_simplewiki_v2.py
# → data/simplewiki/simplewiki_clean.jsonl (~270K articles)

# (4) en-wiki Vital Articles L1-L4 표제어 추출 + simplewiki 매핑
python3 scripts/parse_vital_titles.py
python3 scripts/resolve_vital_titles.py        # SKIP_EN=1로 simplewiki API redirect 만 사용
python3 scripts/recover_vital_from_raw.py      # cleaner 컷 vital 복구
python3 scripts/expand_vital_matches.py        # disambig/comma/punct 변형 매칭
# → data/external/vital_articles/vital_titles_resolved.json

# (5) vital corpus 빌드
python3 scripts/build_vital_corpus.py
# → data/simplewiki/simplewiki_vital_corpus.jsonl (L1-L5 30,311 docs)

# (6) base v3 train/val.txt 생성 (L1-L4 8,942 docs, paragraph 보존, seed=42)
python3 scripts/build_base_v3_train_val.py --max-level 4 --val-frac 0.10 --seed 42
# → data/two-stage-v3/base/{train,val}.txt

# (7) v4 wiki로 차용 + 1 line=1 doc + 리터럴 \n 변환
cp data/two-stage-v3/base/{train,val}.txt data/three-stage-v4/wiki/
python3 scripts/inline_wiki_docs.py
# → data/three-stage-v4/wiki/{train,val}.txt (8,048 / 894 docs)
```

#### B.2 conv — TinyDialogues age-5+age-10 정제 → v4 conv

원본은 EMNLP 2024 TinyDialogues. 두 옵션 중 택일.

**옵션 A — git 보존본 풀기 (가장 빠름, byte-identical 재현):**
```bash
gunzip -kc data/train.txt.gz > data/three-stage-v4/conv/train.txt
gunzip -kc data/val.txt.gz   > data/three-stage-v4/conv/val.txt
# → 51,229 train / 9,219 val conversations (이미 1 line = 1 doc)
```

**옵션 B — HuggingFace 원본부터 정제 (full self-contained):**
```bash
# (1) HuggingFace styfeng/TinyDialogues에서 individual_age_data.zip 다운로드
#     예: huggingface-cli download styfeng/TinyDialogues individual_age_data.zip --local-dir /tmp/td
#     압축 해제 후 4개 파일 사용:
#       tinydialogue_age-5_train.txt, tinydialogue_age-5_val.txt,
#       tinydialogue_age-10_train.txt, tinydialogue_age-10_val.txt

# (2) age-5 + age-10 concat (순서 고정, shuffle 없음)
cat /tmp/td/tinydialogue_age-5_train.txt /tmp/td/tinydialogue_age-10_train.txt > /tmp/td/train.raw.txt
cat /tmp/td/tinydialogue_age-5_val.txt   /tmp/td/tinydialogue_age-10_val.txt   > /tmp/td/val.raw.txt

# (3) 정규식 정제 — speaker 마커 → <|turn|>, emphasis/따옴표 제거,
#     <|endoftext|> 제거, 첫 turn 토큰 제거, <|bos|>...<|eos|> 래핑
#     상세는 docs/dialogues-a510-recipe.md §3.2 (Python inline 코드)
python3 -c "
import re
def clean(text):
    text = re.sub(r'\*\*[^*]+\*\*:\s*', '<|turn|>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\"([^\"]*)\"', r'\1', text)
    text = text.replace('<|endoftext|>', '').rstrip()
    if text.startswith('<|turn|>'): text = text[len('<|turn|>'):]
    return f'<|bos|>{text}<|eos|>'
for split in ('train', 'val'):
    with open(f'/tmp/td/{split}.raw.txt') as f: lines = f.read().splitlines()
    out = '\n'.join(clean(l) for l in lines if l.strip())
    open(f'data/three-stage-v4/conv/{split}.txt', 'w').write(out + '\n')
"
# → data/three-stage-v4/conv/{train,val}.txt (옵션 A와 byte-identical)
```

#### B.3 shared 합본 — BPE 학습용

```bash
cat data/three-stage-v4/{dict,wiki,conv}/train.txt > data/three-stage-v4/shared/train.txt
cat data/three-stage-v4/{dict,wiki,conv}/val.txt   > data/three-stage-v4/shared/val.txt
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

## 9. Stage D — Kotlin 학습 entry point (구현 완료)

데이터 빌드(Stage A-C)에 이어 학습 코드까지 구현. 본 레시피는 이제 처음부터 끝까지 self-contained.

### 코어 변경
- `train/TrainConfig.kt` — `replayDataPath2: String?` / `replayRatio2: Float = 0f` 필드 추가. 기존 `replayDataPath`/`replayRatio`와 동일 시맨틱 (두 번째 replay source).
- `train/DataLoader.kt` — `TripleDataLoader` 추가. `primary + replay1 + replay2`를 row별로 추첨:
  - `r < replay1Ratio` → replay1 (예: dict)
  - `r < replay1Ratio + replay2Ratio` → replay2 (예: wiki)
  - 그 외 → primary (예: conv)
  - `replay1Ratio + replay2Ratio ≤ 1.0` 검증.
- `vec/VecTrainer.kt` — replay 분기 확장:
  - `replayDataPath` & `replayDataPath2` 모두 지정 → `TripleDataLoader`
  - `replayDataPath`만 → 기존 `MixedDataLoader`
  - 둘 다 없음 → 기본 `DataLoader`

### 신규 trainer entry point (architecture: v2와 동일 — C=96, L=6, heads=3, swiglu, RoPE, tied)

| Stage | trainer | initFrom | maxIters | epoch | LR | replay |
|---|---|---|---:|---:|---:|---|
| 1 | `ThreeStageDictTrainV4Vec` | scratch | 2,500 | ~12 | 3e-4 | 없음 |
| 2 | `ThreeStageWikiTrainV4Vec` | pretrain_weights ← Stage1 ckpt | 6,000 | ~3.3 | 1e-4 | dict 30% |
| 3 | `ThreeStageConvTrainV4Vec` | pretrain_weights ← Stage2 ckpt | 8,000 | ~2.0 | 1e-4 | dict 15% + wiki 15% (multi) |

token/iter = 4096 (batch=2 × accum=32 × block=64). dict 코퍼스 863K tok 기준.

### 산출 ckpt 경로
- Stage 1 → `model/dict/vec/<paramCount>/v00XX/`
- Stage 2 → `model/wiki/vec/<paramCount>/v00XX/`
- Stage 3 → `model/conv/vec/<paramCount>/v00XX/`

### 사용법

```bash
# Stage 1 — scratch
./gradlew runThreeStageDictTrainV4Vec
# → 산출 ckpt 경로를 다음 stage에 인자로 넘긴다 (예: model/dict/vec/.../v0010)

# Stage 2 — Stage 1 ckpt에서 이어받기
./gradlew runThreeStageWikiTrainV4Vec --args="model/dict/vec/<paramCount>/v00XX"

# Stage 3 — Stage 2 ckpt에서 이어받기
./gradlew runThreeStageConvTrainV4Vec --args="model/wiki/vec/<paramCount>/v00XX"

# 각 단계 resume / maxIters override 지원:
#   runThreeStage*TrainV4Vec --args="resume"
#   runThreeStage*TrainV4Vec --args="<ckpt> 8000"
```

### 후속 작업 (학습 후)
- Stage 3 dict/wiki replay 비율 튜닝 (현재 15%/15% — 11%/19%로 ×5 oversampling 등가도 후보)
- 단계별 ckpt 평가: 정의 패턴 출력 / 백과 ppl / 대화 ppl
- v3 baseline (TwoStageBaseV2 → TwoStageITV2) 대비 perplexity 비교

## 10. Quick reference — 처음부터 끝까지 한 번에

웹 다운로드부터 학습용 .bin까지. **repo root에서** 실행.

- conv 옵션 A (gzip 보존본): 총 ~5-7분
- conv 옵션 B (HuggingFace) + wiki 신규 빌드: 총 ~30-60분 (WikiExtractor가 가장 큼)
- wiki 빌드를 이전 결과 캐시(`data/two-stage-v3/base/{train,val}.txt`)에서 재사용하면 wiki 빌드 단계 스킵 가능

```bash
mkdir -p data/three-stage-v4/{wiki,conv,shared}

# Stage A — dict (웹 다운로드 + 빌드)
python3 scripts/download_kid_dictionaries.py
python3 scripts/merge_kid_dictionaries.py
python3 scripts/render_dict_docs.py

# Stage B.1 — wiki (simplewiki dump → vital L1-L4)
# v3 base가 이미 있으면 (1)-(6) 스킵하고 (7)만 실행
if [ ! -f data/two-stage-v3/base/train.txt ]; then
  mkdir -p data/simplewiki
  wget -O data/simplewiki/simplewiki-latest-pages-articles.xml.bz2 \
    https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
  python3 -m wikiextractor.WikiExtractor --json --no-templates --processes 4 \
    --output data/simplewiki/extracted \
    data/simplewiki/simplewiki-latest-pages-articles.xml.bz2
  python3 scripts/clean_simplewiki_v2.py
  python3 scripts/parse_vital_titles.py
  python3 scripts/resolve_vital_titles.py
  python3 scripts/recover_vital_from_raw.py
  python3 scripts/expand_vital_matches.py
  python3 scripts/build_vital_corpus.py
  python3 scripts/build_base_v3_train_val.py --max-level 4 --val-frac 0.10 --seed 42
fi
cp data/two-stage-v3/base/{train,val}.txt data/three-stage-v4/wiki/
python3 scripts/inline_wiki_docs.py

# Stage B.2 — conv (옵션 A: gzip 보존본 풀기 — 빠름, byte-identical 재현)
gunzip -kc data/train.txt.gz > data/three-stage-v4/conv/train.txt
gunzip -kc data/val.txt.gz   > data/three-stage-v4/conv/val.txt
# (옵션 B: HuggingFace 원본 — §3 Stage B.2 옵션 B 참조)

# Stage B.3 — shared 합본
cat data/three-stage-v4/{dict,wiki,conv}/train.txt > data/three-stage-v4/shared/train.txt
cat data/three-stage-v4/{dict,wiki,conv}/val.txt   > data/three-stage-v4/shared/val.txt

# Stage C — BPE 학습 + 인코딩
./gradlew runBpe --args="data/three-stage-v4/shared 2000 cased skip-bin"
for d in dict wiki conv; do
  ./gradlew runEncodeWithExistingMeta \
    --args="data/three-stage-v4/shared/meta.json data/three-stage-v4/$d"
done
```

## 11. 빌드 후 검증

`expected vs actual`로 sanity check. 수치가 다르면 어떤 단계에서 어긋났는지 진단.

```bash
# 1) shared meta.json — vocab/merges 검증
python3 -c "
import json
m = json.load(open('data/three-stage-v4/shared/meta.json'))
assert m['vocabularySize'] == 2000, f'vocab mismatch: {m[\"vocabularySize\"]}'
assert m['useWordPreTokenize'] == True, 'pre-tokenize off'
assert len(m['merges']) >= 1900, f'merges 부족: {len(m[\"merges\"])}'
print(f'OK: vocab={m[\"vocabularySize\"]}, merges={len(m[\"merges\"])}')
"

# 2) 학습용 토큰 수 — 결정 파라미터 동일 + v3 입력 동일이면 동일 수치 나와야
python3 -c "
import os
expected = {'dict': 863_415, 'wiki': 7_485_510, 'conv': 16_271_975}
for d, e in expected.items():
    sz = os.path.getsize(f'data/three-stage-v4/{d}/train.bin') // 4
    diff_pct = 100 * abs(sz - e) / e
    status = 'OK' if diff_pct < 1.0 else f'WARN ({diff_pct:.1f}% off)'
    print(f'{d:<6} {sz:>12,} (expected {e:>12,}) {status}')
"

# 3) doc 수 vs 라인 수 — 1 line = 1 doc 검증
for d in dict wiki conv; do
    lines=$(wc -l < data/three-stage-v4/$d/train.txt)
    docs=$(grep -c "<|bos|>" data/three-stage-v4/$d/train.txt)
    [ "$lines" -eq "$docs" ] && echo "$d: OK ($lines lines = $docs docs)" \
        || echo "$d: MISMATCH (lines=$lines docs=$docs)"
done

# 4) 단어 분포 진단 — dict 보강 효과 (재현 시 동일 수치)
python3 scripts/analyze_v4_low_freq.py
# expected:
#   unique types: dict=38,422  wiki=101,675  conv=35,680  union=123,305
#   dict ∩ wiki-only & freq≤5 : 5,796
#   dict ∩ conv-only & freq≤5 : 1,914
#   only_dict : 8,433
```

## 12. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `merge_kid_dictionaries.py`가 화이트리스트 0개 | 입력 코퍼스(`data/two-stage-v3/{base,it}/train.txt`)가 없음 | Stage B.1 wiki / B.2 conv 빌드 선행, 또는 v3 산출물 캐시 활용 |
| WikiExtractor 미설치 | `wikiextractor` Python 패키지 부재 | `pip install wikiextractor` 또는 `scripts/_wikiextractor_local/` 사용 |
| simplewiki dump URL 만료 | `latest` 링크는 갱신되며 표제어 수가 약간 변함 | 특정 시점 dump(`pages-articles-YYYYMMDD.xml.bz2`)로 고정하여 재현성 확보 |
| `nltk.download('wordnet')` 실패 | 네트워크 차단 / proxy | `~/nltk_data/corpora/wordnet/`로 수동 배치 |
| `runBpe` OOM | 12GB heap 부족 (대규모 코퍼스) | `build.gradle.kts:37` `-Xmx12g` 상향 또는 `skip-bin` 확실히 적용 |
| `inline_wiki_docs.py` doc 수가 0 | v3 base가 `<|bos|>...<|eos|>` 형식 아님 | v3 base 형식 확인, 정규식 매칭 점검 |
| 인코딩 토큰 수가 expected와 ≥1% 다름 | shared/meta.json이 다른 코퍼스로 학습됨 | shared/ 재합본 후 BPE 재실행 |
