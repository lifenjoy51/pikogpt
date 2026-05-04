# three-stage-v4-ko 데이터셋 레시피 — 한국어 dict → wiki → conv 3단계 curriculum

영문 `three-stage-v4` (`docs/three-stage-v4-recipe.md`)의 한국어 짝. 영문판 학습이 검증한 "사전으로 lexical knowledge 주입 → 위키로 도메인 적응 → 대화로 톤 정렬" 커리큘럼을 한국어 코퍼스로 재현한다.

본 문서는 **데이터 준비(Stage A-C) + 학습 entry point(Stage D) 설계** 단계까지 기록. 영문 v4 파일/데이터/Gradle 태스크는 절대 변경하지 않으며, 한국어판 모든 자원은 `*-ko` 접미가 붙은 별도 경로에 둔다.

## 0. 선결 조건 — 사용자가 직접 처리할 항목

자동화가 가능한 부분(코드, kowiki dump 다운로드, BPE, 학습)은 모두 스크립트로 처리하지만, **아래 두 가지는 사용자가 직접 받아 배치해야 한다.**

### A. 한국어기초사전 (dict, 필수)

1. https://github.com/spellcheck-ko/korean-dict-nikl 의 README 절차 따라 한국어기초사전(krdict) ZIP 다운로드
   - 사이트: https://krdict.korean.go.kr/
   - 절차: 회원가입 → 로그인 → "내 정보 관리" → "사전 내려받기" → "전체 내려받기" (XML ZIP)
2. 받은 ZIP을 `data/dictionaries-ko/krdict/`에 압축 해제 (xml 파일이 보여야 함)
3. spellcheck-ko 레포의 빌드는 README상 FIXME(미완성). 우리는 raw XML만 직접 파싱

표준국어대사전·우리말샘은 fallback. 한국어기초사전 dump 형식이 우리 파서와 안 맞을 때만 추가 검토.

### B. 어린이 대화 코퍼스 (conv, 둘 중 하나 이상)

영문판이 어린이용 TinyDialogues를 쓰는 것에 짝을 맞춘다. 라이선스상 비상업/연구 한정.

**선택 1 — AI Hub 소아자유대화 #108 (권장)**
- https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=108
- 절차: 회원가입(휴대폰 본인 인증) → 데이터 활용 신청서 작성(연구 목적 명시) → 승인 후 다운로드
- WAV+JSON pair 중 **JSON만** `data/child_dialog_ko/aihub_108/`에 배치 (음성 파일은 학습에 불필요)

**선택 2 — 국립국어원 나사렛 말뭉치 (보강)**
- https://kli.korean.go.kr/ → 모두의 말뭉치
- 절차: 본인 인증 회원가입 → "나사렛 말뭉치" 검색 → 장바구니 → 신청서 작성 → 약정 전자서명 후 다운로드
- 받은 SJML/텍스트 파일을 `data/child_dialog_ko/nikl_nasaret/`에 배치
- 65,783어절, 2-5세 아동 발화 (전문가 3차 전사 고품질)

**선택 3 — fallback (자동, 어조는 백과적)**
- 둘 다 못 받았을 때 `build_korean_conv_v4.py --source synthetic`로 kowiki 첫 단락 Q-A 합성. CC BY-SA 4.0 깔끔. 학습은 진행되나 어조가 어린이 대화와 다름

### C. 환경

- **Python 3.10+**. 사용 모듈: `xml.etree.ElementTree`(표준), `urllib`(표준), `wikiextractor`(scripts/ 안에 이미 있음)
- **JDK 17+**, **Gradle 8.x** (wrapper로 자동)
- **디스크 공간**: 약 5 GB
  - kowiki dump 1.5 GB(압축) + WikiExtractor 산출 ~3 GB
  - krdict ZIP/XML ~200 MB
  - 어린이 대화 raw 변동 (AI Hub #108은 텍스트만 받으면 ~100 MB 이하)

### 결정 파라미터 (재현용 고정값)

| 파라미터 | 값 | 위치 | 역할 |
|---|---|---|---|
| `FREQ_THRESHOLD` | 10 | `merge_korean_dictionaries.py` | 입력 코퍼스 어절 빈도 화이트리스트 컷 |
| `MAX_MEANINGS` | 5 | `merge_korean_dictionaries.py` | 표제어당 최대 의미 |
| `WORD_RE_KO` | `[가-힣]{2,}` | `merge_korean_dictionaries.py` | 1글자 표제어 차단 |
| `VAL_FRAC` | 0.10 | `render_korean_dict_docs.py` / `build_kowiki_v4.py` / `build_korean_conv_v4.py` | train/val split |
| `SEED` | 42 | 동일 | shuffle / split 재현 |
| `vocab` | 8000 | `runBpe` 인자 | BPE vocab 크기 (한글 음절 분포 ≈ 7K + ASCII + merge 여유) |
| BPE flags | `cased skip-bin` | `runBpe` 인자 | 대소문자 보존(한글은 영향 없음, 영문 혼용 시 cased) + shared bin 생략 |

## 1. 결정 요약

### 핵심 변경 (영문 v4 대비)

- **데이터 출처**: Simple English Dict + WordNet → 한국어기초사전(krdict). simplewiki → kowiki. TinyDialogues → AI Hub #108 / 나사렛 말뭉치
- **vocab 크기**: 2000 → **8000** (한글 음절 7K + ASCII + BPE merge 여유)
- **doc 형식, 1 line=1 doc, multi-replay 정책, architecture(C=96, L=6, H=3, SwiGLU+RoPE)**: 영문판과 동일하게 유지
- **wiki vital 등가 필터**: simplewiki Vital Articles L1-L4 → ko 위키는 vital이 없으므로 (1) 영문 vital을 langlinks로 ko 매핑 + (2) 분류:기초_문서 트리 보강 union

### 영문 v4 대비 격리 전략

- 데이터 디렉토리 마지막 segment를 `dict-ko`/`wiki-ko`/`conv-ko`로 두어 ckpt가 자동으로 `model/dict-ko/...`로 분리됨 (영문은 `model/dict/...`)
- vocab 차이로 paramCount segment도 달라져 이중 분리
- 한국어판 문서/스크립트/Trainer 모두 `*-ko` 또는 `Ko*` 접두/접미. 영문판 파일은 read-only import만 (`text_cleaning.py`의 `SMART` 매핑 등)

### 사용자 결정 (확정)

1. **dict 출처**: spellcheck-ko/korean-dict-nikl 가이드 따라 한국어기초사전 우선. 표준국어대사전·우리말샘은 fallback
2. **conv 출처**: AI Hub 소아자유대화 #108 + 나사렛 말뭉치 둘 중 가용한 것(또는 union)
3. **vocab 8000**: tok/word 측정 결과 > 3.5면 16K로 재학습 (BPE만 재실행)
4. **격리 전략**: `*-ko` 접미 + 별도 디렉토리

## 2. 코퍼스 사양

### 데이터 소스

| Stage | 데이터셋 | 라이선스 | 자동화 | 상업 사용 |
|---|---|---|---|---|
| dict | 한국어기초사전 (krdict) via spellcheck-ko 가이드 | CC BY-SA 2.0 KR | 수동 (회원가입 후 ZIP) | 가능 (BY-SA 의무 표기) |
| wiki | kowiki dump (`dumps.wikimedia.org/kowiki/latest/`) | CC BY-SA 4.0 + GFDL | **자동** | 가능 (BY-SA 의무) |
| conv (1) | AI Hub 소아자유대화 #108 | 비상업/연구용 | 수동 (회원 인증 + 신청) | **불가** |
| conv (2) | 국립국어원 나사렛 말뭉치 | 모두의 말뭉치 약정 | 수동 (전자서명) | **불가** |
| conv (fallback) | kowiki Q-A 합성 | CC BY-SA 4.0 | 자동 | 가능 |

학술/실험 목적으로만 사용. 상업 활용 시 출처별 라이선스 의무를 별도 검토.

### 토큰 통계 (예상치, 빌드 후 측정해 갱신)

| 디렉토리 | docs (예상) | 어절 (예상) | 토큰 (vocab=8K, 예상) | 비고 |
|---|---:|---:|---:|---|
| `dict-ko/` | 12K-20K | 0.4M-0.8M | 1M-2M | 한국어기초사전 표제어 × 의미별 doc 분리 |
| `wiki-ko/` | 8K-12K | 3M-6M | 7M-12M | kowiki vital interlanguage + 분류:기초_문서 union |
| `conv-ko/` | 변동 | 변동 | 변동 | 출처에 따라 매우 가변 |

빌드 후 `train.bin` 크기/4 = 토큰 수, `train.txt` `wc -w` = 어절 수로 측정. **tok/word ≈ 2.0~3.5** 기대 (영문 ≈ 1.4 대비 한국어 음절·어미 분리로 더 높음). 4.0 이상이면 vocab 16K로 재학습 신호.

## 3. 데이터 빌드 파이프라인

### Stage A — dict 데이터 빌드

```bash
# A.0 (사용자) krdict ZIP을 받아 풀어 둔다
#     data/dictionaries-ko/krdict/*.xml

# A.1 XML → JSONL 변환
python3 scripts/build_krdict_jsonl.py
# → data/dictionaries-ko/nikl_entries.jsonl
#   schema: {word, pos, definitions: [str], synonyms: [str], antonyms: [str], hypernym}

# A.2 빈도 화이트리스트 + 정제 (wiki/conv 빌드 선행 필요)
python3 scripts/merge_korean_dictionaries.py
# → data/dictionaries-ko/merged_entries.ko.jsonl

# A.3 한국어 자연어 doc 렌더링 + 90:10 split (seed=42)
python3 scripts/render_korean_dict_docs.py
# → data/three-stage-v4-ko/dict-ko/{train.txt, val.txt}
#   format: 1 line = 1 doc, doc 내부는 리터럴 \n
```

### Stage B — wiki / conv 빌드 + 형식 통일

```bash
mkdir -p data/three-stage-v4-ko/{wiki-ko,conv-ko,shared-ko}
mkdir -p data/kowiki data/dictionaries-ko data/child_dialog_ko
```

#### B.1 wiki — kowiki dump → vital 등가 필터 → inline

```bash
# (1) kowiki dump 다운로드 (~1.5 GB 압축, 한 번만)
curl -L https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 \
  -o data/kowiki/kowiki-latest-pages-articles.xml.bz2

# (2) WikiExtractor (영문 v4가 사용하는 scripts/ 모듈 그대로 import)
python3 -m wikiextractor.WikiExtractor \
  --json --no-templates --processes 4 \
  --output data/kowiki/extracted \
  data/kowiki/kowiki-latest-pages-articles.xml.bz2

# (3) vital 등가 필터 + 정제 + train/val 작성
python3 scripts/build_kowiki_v4.py
# → data/three-stage-v4-ko/wiki-ko/{train.txt, val.txt} (다단락 형식, <|bos|>\n...\n<|eos|>)

# (4) 1 line = 1 doc + 리터럴 \n 변환
python3 scripts/inline_korean_wiki_docs.py
# → data/three-stage-v4-ko/wiki-ko/{train.txt, val.txt} (in-place)
```

vital 필터링 전략 (build_kowiki_v4.py 내부):
1. 영문 v4의 `data/simplewiki/simplewiki_vital_corpus.jsonl` 표제어를 langlinks(또는 wikipedia API)로 ko 매핑 → 영문판과 의미 정렬
2. `Category:기초_문서` 트리 N단계 자식 union으로 보강
3. 둘의 union을 `vital_titles_ko_resolved.json`으로 캐시

#### B.2 conv — 어린이 대화 코퍼스 정제 → conv-ko

```bash
# 출처 자동 감지 (--source 인자로 명시 가능)
python3 scripts/build_korean_conv_v4.py --source both       # 둘 다 union
# 또는
python3 scripts/build_korean_conv_v4.py --source aihub      # AI Hub만
python3 scripts/build_korean_conv_v4.py --source nasaret    # 나사렛만
python3 scripts/build_korean_conv_v4.py --source synthetic  # kowiki Q-A fallback
```

doc 형식 (영문 conv와 동일): `<|bos|>\n{발화1}\n<|turn|>\n{발화2}\n<|turn|>\n…\n<|eos|>` (1 line=1 doc, 리터럴 `\n`)

소스별 처리:
- AI Hub #108: JSON `stt` 필드 추출 + `convrsThema`로 같은 주제 발화를 multi-turn으로 묶음
- 나사렛 SJML: 화자 ID로 turn 분리, 발화별 `\n<|turn|>\n` 삽입
- 정제: `clean_inline_ko` (한글/ASCII printable만 보존) + 길이 컷(최대 1500자)
- 90:10 split, seed=42

#### B.3 shared 합본 — BPE 학습용

```bash
cat data/three-stage-v4-ko/{dict-ko,wiki-ko,conv-ko}/train.txt > data/three-stage-v4-ko/shared-ko/train.txt
cat data/three-stage-v4-ko/{dict-ko,wiki-ko,conv-ko}/val.txt   > data/three-stage-v4-ko/shared-ko/val.txt
```

### Stage C — BPE 학습 + 인코딩

```bash
# C.1 shared-ko에서 cased BPE 학습 (vocab=8000, skip-bin)
./gradlew runBpe --args="data/three-stage-v4-ko/shared-ko 8000 cased skip-bin"
# → data/three-stage-v4-ko/shared-ko/meta.json (vocab=8000)
# 소요: ~5-15분 (코퍼스 크기에 따라)

# C.2 shared-ko/meta.json으로 3 학습용 디렉토리 인코딩
for d in dict-ko wiki-ko conv-ko; do
  ./gradlew runEncodeWithExistingMeta \
    --args="data/three-stage-v4-ko/shared-ko/meta.json data/three-stage-v4-ko/$d"
done
# 각 → train.bin, val.bin, meta.json 사본
```

`runBpe`/`runEncodeWithExistingMeta`는 영문 v4가 쓰는 그 Gradle 태스크 그대로 — 인자만 다르게.

## 4. 데이터 형식

영문 v4와 동일한 규약. 모든 디렉토리의 `train.txt` / `val.txt`는 **1 line = 1 doc, doc 내부는 리터럴 `\n`**.

### dict-ko doc 예시 (의미별 분리)
```
<|bos|>\n# 사과\n사과는 사과나무의 열매이다.\n<|eos|>
<|bos|>\n# 사과\n사과는 자기의 잘못을 인정하고 용서를 비는 일이다.\n<|eos|>
<|bos|>\n# 사과\n사과는 과일의 한 종류이다.\n<|eos|>
<|bos|>\n# 사과\n비슷한 말: 능금.\n<|eos|>
```

규칙:
- 첫 의미: 명사 → "{단어}{은/는} {정의}이다.", 동사 → "{단어}: {정의}.", 형용사 → "{단어}{은/는} {정의}는 뜻이다."
- 추가 의미: 명사 → "{단어}{은/는} … 이다." (각 의미를 별도 doc으로 분리하므로 "또한"류 접속 불필요)
- hypernym (명사): "{단어}{은/는} {hypernym}의 한 종류이다."
- syns/ants: 별도 doc, 형식 `비슷한 말: …`, `반대말: …`

받침 분기는 `josa_ko(word, with_jong, without_jong)` 헬퍼로 처리:
- 사과(받침 X) + 은/는 → "사과는"
- 자동차(받침 X) + 을/를 → "자동차를"
- 책(받침 ㄱ) + 은/는 → "책은"

### wiki-ko doc 예시
```
<|bos|>\n# 한반도\n한반도는 동아시아의 반도이다. ...\n한반도의 면적은 약 ...\n<|eos|>
```

### conv-ko doc 예시
```
<|bos|>안녕! 오늘 뭐 했어?<|turn|>저는 친구랑 놀이터에서 놀았어요.<|turn|>...<|eos|>
```

## 5. 학습 흐름 (Stage D)

영문 v4의 trainer 3개를 그대로 복사 후 `dataPath`/`replay` 경로만 한국어 경로로 교체. architecture는 동일.

### 디렉토리별 역할

- `dict-ko/`, `wiki-ko/`, `conv-ko/` — 각 단계의 primary
- `shared-ko/` — BPE 학습 전용. 학습 시 사용 안 함

### Stage 흐름

| Stage | trainer | initFrom | dataPath | replay |
|---|---|---|---|---|
| 1 | `ThreeStageKoDictTrainV4Vec` | scratch | `data/three-stage-v4-ko/dict-ko` | 없음 |
| 2 | `ThreeStageKoWikiTrainV4Vec` | pretrain_weights ← Stage1 best | `data/three-stage-v4-ko/wiki-ko` | `dict-ko/`, ratio=0.30 |
| 3 | `ThreeStageKoConvTrainV4Vec` | pretrain_weights ← Stage2 best | `data/three-stage-v4-ko/conv-ko` | **dict-ko/** & **wiki-ko/** 별도 path (multi-replay 15%/15%) |

### Architecture (영문판과 동일)

| 항목 | 값 |
|---|---|
| `embeddingDimension` | 96 |
| `numberOfLayers` | 6 |
| `numberOfHeads` | 3 |
| `blockSize` | 64 |
| `batchSize` | 2 |
| `gradientAccumulationSteps` | 32 (4096 tok/iter) |
| `mlpActivation` | swiglu |
| `positionEncoding` | rope |
| `tieWeights` | true |
| `dropout` | 0.05 |
| `alwaysSaveCheckpoint` | true |
| `earlyStopPatience` | 10 |

`maxIters`는 토큰 수 측정 후 영문 비례로 재산정. 우선 default 22K/20K/24K로 시작.

### 산출 ckpt 경로

- Stage 1 → `model/dict-ko/vec/<paramCount>/v00XX/`
- Stage 2 → `model/wiki-ko/vec/<paramCount>/v00XX/`
- Stage 3 → `model/conv-ko/vec/<paramCount>/v00XX/`

영문판(`model/dict/...`, `model/wiki/...`, `model/conv/...`)과 자동 분리.

### 사용법

```bash
# Stage 1 — scratch
./gradlew runThreeStageKoDictTrainV4Vec

# Stage 2 — Stage 1 ckpt에서 이어받기
./gradlew runThreeStageKoWikiTrainV4Vec --args="model/dict-ko/vec/<paramCount>/v00XX"

# Stage 3 — Stage 2 ckpt에서 이어받기
./gradlew runThreeStageKoConvTrainV4Vec --args="model/wiki-ko/vec/<paramCount>/v00XX"

# 각 단계 resume / maxIters override 지원:
#   runThreeStageKo*TrainV4Vec --args="resume"
#   runThreeStageKo*TrainV4Vec --args="<ckpt> 8000"
```

## 6. 영향받지 않은 파일 (영문 v4 보존)

- `data/three-stage-v4/`, `data/dictionaries/`, `data/simplewiki/`, `data/two-stage-v3/` — 보존
- `scripts/{download_kid_dictionaries, merge_kid_dictionaries, render_dict_docs, build_base_v3_train_val, inline_wiki_docs, text_cleaning, clean_simplewiki_v2, parse_vital_titles, resolve_vital_titles, recover_vital_from_raw, expand_vital_matches, build_vital_corpus, analyze_v4_low_freq}.py` — 변경 0건 (한국어판은 read-only import만)
- `src/main/kotlin/train/experiments/ThreeStage{Dict,Wiki,Conv}TrainV4Vec.kt` 영문 3개 — 변경 0건
- `build.gradle.kts`의 영문 task 3개 (`runThreeStageDictTrainV4Vec` 등) — 변경 0건. 한국어 task 3개를 추가만 함
- `src/main/kotlin/data/SimpleBPE.kt`, `vec/VecTrainer.kt`, `vec/VecSampler.kt` 등 코어 — 변경 0건. 한국어판도 그대로 재사용

## 7. 산출물 트리

```
data/dictionaries-ko/
├── krdict/                       (사용자 수동 배치 — krdict ZIP 압축 해제)
│   └── *.xml
├── nikl_entries.jsonl            (build_krdict_jsonl.py 산출)
└── merged_entries.ko.jsonl       (merge_korean_dictionaries.py 산출)

data/kowiki/
├── kowiki-latest-pages-articles.xml.bz2  (자동 다운)
├── extracted/AA/wiki_00 …       (WikiExtractor 산출)
└── kowiki_vital_corpus.jsonl     (build_kowiki_v4.py 중간 산출, 캐시)

data/child_dialog_ko/
├── aihub_108/                    (사용자 수동 배치 — JSON만)
└── nikl_nasaret/                 (사용자 수동 배치 — SJML/text)

data/three-stage-v4-ko/
├── README.md                     (이 디렉토리 안의 빌드 가이드)
├── dict-ko/                      (Stage 1 학습 데이터)
│   ├── train.txt val.txt
│   ├── train.bin val.bin
│   └── meta.json
├── wiki-ko/                      (Stage 2 학습 데이터)
│   ├── train.txt val.txt
│   ├── train.bin val.bin
│   └── meta.json
├── conv-ko/                      (Stage 3 학습 데이터)
│   ├── train.txt val.txt
│   ├── train.bin val.bin
│   └── meta.json
└── shared-ko/                    (BPE 학습 전용)
    ├── train.txt val.txt
    └── meta.json                 (vocab=8000, cased)
```

신규 스크립트:
- `scripts/build_krdict_jsonl.py` — 한국어기초사전 XML → JSONL
- `scripts/merge_korean_dictionaries.py` — 빈도 화이트리스트 + 정제
- `scripts/render_korean_dict_docs.py` — 한국어 자연어 템플릿 + 받침 처리
- `scripts/build_kowiki_v4.py` — kowiki vital 등가 필터 + 정제
- `scripts/inline_korean_wiki_docs.py` — 1 line=1 doc 변환
- `scripts/build_korean_conv_v4.py` — AI Hub/나사렛/synthetic 출처별 처리
- `scripts/text_cleaning_ko.py` — 한글 보존 정제 (영문 `text_cleaning.py`는 read-only import)

신규 trainer:
- `src/main/kotlin/train/experiments/ThreeStageKoDictTrainV4Vec.kt`
- `src/main/kotlin/train/experiments/ThreeStageKoWikiTrainV4Vec.kt`
- `src/main/kotlin/train/experiments/ThreeStageKoConvTrainV4Vec.kt`

`build.gradle.kts`에 추가: `runThreeStageKo{Dict,Wiki,Conv}TrainV4Vec` 3개 task.

## 8. Quick reference — 처음부터 끝까지 한 번에

repo root에서 실행. 사용자 액션 항목(§0.A, §0.B)이 완료된 상태 가정.

```bash
mkdir -p data/three-stage-v4-ko/{wiki-ko,conv-ko,shared-ko} \
         data/kowiki data/dictionaries-ko/krdict data/child_dialog_ko

# (사전 준비) krdict XML, AI Hub/나사렛 데이터를 위 디렉토리에 배치 — §0 참조

# Stage B.1 — wiki-ko (kowiki dump 다운로드 + 추출 + vital 필터 + inline)
curl -L https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 \
  -o data/kowiki/kowiki-latest-pages-articles.xml.bz2
python3 -m wikiextractor.WikiExtractor --json --no-templates --processes 4 \
  --output data/kowiki/extracted data/kowiki/kowiki-latest-pages-articles.xml.bz2
python3 scripts/build_kowiki_v4.py
python3 scripts/inline_korean_wiki_docs.py

# Stage B.2 — conv-ko
python3 scripts/build_korean_conv_v4.py --source both
# 또는 --source aihub | --source nasaret | --source synthetic

# Stage A — dict-ko (wiki/conv 빈도 의존)
python3 scripts/build_krdict_jsonl.py
python3 scripts/merge_korean_dictionaries.py
python3 scripts/render_korean_dict_docs.py

# Stage B.3 — shared-ko 합본
cat data/three-stage-v4-ko/{dict-ko,wiki-ko,conv-ko}/train.txt \
    > data/three-stage-v4-ko/shared-ko/train.txt
cat data/three-stage-v4-ko/{dict-ko,wiki-ko,conv-ko}/val.txt \
    > data/three-stage-v4-ko/shared-ko/val.txt

# Stage C — BPE + 인코딩
./gradlew runBpe --args="data/three-stage-v4-ko/shared-ko 8000 cased skip-bin"
for d in dict-ko wiki-ko conv-ko; do
  ./gradlew runEncodeWithExistingMeta \
    --args="data/three-stage-v4-ko/shared-ko/meta.json data/three-stage-v4-ko/$d"
done

# Stage D — 학습
./gradlew runThreeStageKoDictTrainV4Vec
./gradlew runThreeStageKoWikiTrainV4Vec --args="model/dict-ko/vec/<P>/v00XX"
./gradlew runThreeStageKoConvTrainV4Vec --args="model/wiki-ko/vec/<P>/v00XX"
```

## 9. 빌드 후 검증

```bash
# 1) shared-ko meta.json — vocab 검증
python3 -c "
import json
m = json.load(open('data/three-stage-v4-ko/shared-ko/meta.json'))
assert m['vocabularySize'] == 8000, f'vocab mismatch: {m[\"vocabularySize\"]}'
print(f'OK: vocab={m[\"vocabularySize\"]}')
print('special[0..3]:', [m['indexToString'][str(i)] for i in range(4)])
# 기대: <|eos|>, <|unk|>, <|bos|>, <|turn|>
"

# 2) 토큰/어절 비 측정 — 4.0 이상이면 vocab 16K로 재학습 신호
python3 -c "
import os
for d in ['dict-ko','wiki-ko','conv-ko']:
    n_tok = os.path.getsize(f'data/three-stage-v4-ko/{d}/train.bin') // 4
    n_word = sum(1 for _ in open(f'data/three-stage-v4-ko/{d}/train.txt').read().split())
    ratio = n_tok / max(1, n_word)
    flag = '' if ratio < 3.5 else ' ⚠ vocab 16K 재학습 검토'
    print(f'{d:<10} tokens={n_tok:>12,} words={n_word:>12,} tok/word={ratio:.2f}{flag}')
"

# 3) doc 수 vs 라인 수 — 1 line = 1 doc 검증
for d in dict-ko wiki-ko conv-ko; do
    lines=$(wc -l < data/three-stage-v4-ko/$d/train.txt)
    docs=$(grep -c "<|bos|>" data/three-stage-v4-ko/$d/train.txt)
    [ "$lines" -eq "$docs" ] && echo "$d: OK ($lines lines = $docs docs)" \
        || echo "$d: MISMATCH (lines=$lines docs=$docs)"
done

# 4) smoke 학습 (각 stage 2 iter)
./gradlew runThreeStageKoDictTrainV4Vec --args="2"
# → model/dict-ko/vec/<P>/v0001/ 생성 확인 후 다음 stage smoke
```

## 10. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `build_krdict_jsonl.py`가 빈 JSONL | krdict XML 스키마가 우리 파서와 다름 (개정판) | XML 한 항목 dump 출력 → 우리 파서 필드 매핑 수정. 또는 표준국어대사전·우리말샘으로 폴백 |
| `merge_korean_dictionaries.py` 화이트리스트 0개 | wiki-ko/conv-ko가 아직 안 만들어졌음 | Stage B.1, B.2 먼저 실행 |
| WikiExtractor 미설치 | 패키지 부재 | `pip install wikiextractor` 또는 `scripts/wikiextractor/` 사용 |
| `build_kowiki_v4.py` vital 매핑 결과 부족 | 영문 vital 표제어가 ko 위키에 없음 | `--strategy union`으로 분류:기초_문서 트리 보강 |
| `runBpe` OOM | 8000 vocab + 큰 코퍼스 메모리 부족 | `build.gradle.kts`의 runBpe `-Xmx12g` 상향 |
| tok/word > 3.5 | vocab 8K 부족 | `runBpe`만 `--args="data/three-stage-v4-ko/shared-ko 16000 cased skip-bin"`로 재실행 후 인코딩 재실행 |
| Stage 2/3 ckpt 로드 실패 | 직전 stage paramCount segment가 다름 (vocab 변경) | vocab 변경 시 모든 stage 다시 인코딩 + Stage 1부터 재학습 |
| 한글이 깨짐 (`?` 표시) | 파일 charset 미명시로 인한 Windows 환경 문제 | macOS/Linux에서만 검증. Windows는 별도 PR로 charset 명시 추가 필요 |

## 11. 결정 이력

1. **dict 출처 선택**: 우리말샘은 기여도 100K+ 표제어로 가장 크지만 회원 신청 + XML dump 형식이 복잡. spellcheck-ko/korean-dict-nikl 가이드의 한국어기초사전이 가장 정제도 높고 자동화도 가능 (XML이라 직접 파싱). 표준국어대사전·우리말샘은 fallback 수준
2. **vocab 8K vs 16K**: 한글 음절 11,172자 중 코퍼스 등장 음절은 7K-9K로 추정. 8K가 음절 + ASCII + 약간의 BPE merge 여유. 16K는 더 효율적이지만 모델 embedding 크기가 8배 증가. 일단 8K로 빌드 후 tok/word 측정해 결정
3. **conv 출처 — 어린이 대화 강조**: KoAlpaca-v1.1a는 자동화 가능하나 라이선스 모호 + 출처가 네이버 지식인 + 어조가 어른용 instruction이라 영문판 어린이 톤과 맞지 않음. AI Hub #108 / 나사렛 말뭉치는 수동 다운이지만 어린이 발화 정밀 자료
4. **vital 등가 — 영문 vital interlanguage 우선**: ko 위키는 vital articles 분류가 없음. 영문 vital을 langlinks로 ko 매핑하면 8K 미만 가능 → 분류:기초_문서 트리 보강으로 8K-10K 보장
5. **격리 전략 — 디렉토리 segment에 `-ko` 접미**: `modelCheckpointDir` 옵션을 변경하지 않고도 자동으로 ckpt 경로 분리됨 (`model/dict-ko/...` vs `model/dict/...`). 영문판 학습이 진행 중이어도 충돌 없음

## 12. 비포함 (이번 레시피 범위 밖)

- 한국어용 형태소 분석기(KoNLPy/Mecab) 도입 — 의존성 무거움, 어절 기반 BPE로 충분
- byte-level BPE 구현 — `SimpleBPE` 코어 변경 필요, 영문판과 분리 원칙 위배
- 한국어 ckpt에 대한 정성 평가 자동화 — Stage 3 끝난 후 별도 작업
- 영문판 미커밋 변경(현재 진행 중인 git status 항목)을 한국어판에 추격 반영 — 영문판 학습 결과 본 뒤 재반영
- charset 미명시 보강 — macOS/Linux 환경 가정. Windows 대응은 별도 PR
