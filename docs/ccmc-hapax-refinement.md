# CCMC hapax 단어 5세 어휘 단순화 — 방법론

## Context

v6 multi-axis 합성 직후 `data/ccmc-all-raw/` 7개 파일(약 5M token, 16,908 unique)에 hapax(`count=1`) 단어가 **4,525건 (unique의 26.8%)** 존재. 5세 어휘 strict 정책상 학습 corpus에 들어가면 안 되는 hard word, 외래어 굴절, 비문법, LLM 오타 등이 섞여있어 BPE vocab/모델 일반화에 noise로 작용 가능.

목표: 각 hapax 단어를 더 쉽고 단조로운 표현으로 sentence 단위 재작성하거나, 의도된 사용(외래어/고유명사/의성어/시연)이면 그대로 유지.

## 의사결정

- **sentence 단위 교체 (word 단위 X)** — context-aware 단순화 위해. 5세 학생도 sentence 안에서 단어 의미를 파악하므로 단어만 바꾸면 어색.
- **LLM 판정 + 자동 적용** — 4,525건은 사람 검수 비현실. Flash/Pro로 판정 자동화, hard rule로 echo 차단.
- **multi-cycle iteration** — 한 번에 모든 hapax를 잡지 않고 cycle마다 corpus → hapax 재추출 → LLM rewrite → apply 반복. cycle마다 점진적 수렴.

## 파이프라인

```
scripts/
├── review_vocab.py            # 단어 빈도 집계 + 의심 카테고리 추출 (long/no_vowel/triple_repeat)
├── review_hapax_flash.py      # hapax 전수 검토 (Flash) — abnormal 식별
├── find_typos.py              # edit-distance=1 typo 후보 (false positive 많음)
├── verify_suspects.py         # 의심 word의 corpus 컨텍스트 출력 (수동 검증)
├── add_hapax_sentences.py     # rare_hapax.tsv에 sentence 컬럼 추가 (1차, '.' boundary)
├── refine_hapax_sentences.py  # sentence 컬럼 재추출 ([.!?] + special marker + literal \n boundary)
├── rebuild_rare_hapax.py      # 현재 corpus 기준 hapax + sentence 처음부터 재생성
├── rewrite_hapax_flash.py     # LLM rewrite 호출 (--model로 Flash/Pro 선택)
└── apply_rewrites.py          # rewritten sentence를 corpus에 in-place 교체 + rare_hapax.tsv 제거
```

산출물 위치: `data/ccmc-all-raw/_vocab_review/` (untracked — 검토 자산)
- `rare_hapax.tsv` — 현재 잔여 hapax 목록
- `rare_hapax_rewritten.tsv` — 최신 LLM rewrite 결과
- `rewrite_raw.jsonl` / `flash_abnormal.jsonl` — 배치 raw 응답
- `apply_log.tsv` — applied/skipped 기록
- `suspicious.tsv` / `typo_candidates.tsv` / `nonascii.txt` — 의심 카테고리 dump

## sentence 추출 boundary

초기 `add_hapax_sentences.py`는 `.`만 boundary로 사용 → dialogues.txt의 `<|turn|>` 마커 가운데 끼인 발화를 한 chunk로 잡는 문제. 9% skip 발생.

`refine_hapax_sentences.py` (이후 rebuild에 통합) — boundary 확장:
- `[.!?]` 문장 종결자
- `<|bos|>` / `<|eos|>` / `<|sep|>` / `<|turn|>` special marker
- literal `\n` / `\r` / `\t` (wiki.txt의 `title\n\nbody` 포맷 대응)

```python
BOUNDARY_RE = re.compile(r"<\|(?:bos|eos|sep|turn)\|>|\\[nrt]")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(line):
    out = []
    for chunk in BOUNDARY_RE.split(line):
        for s in SENT_SPLIT_RE.split(chunk):
            s = WS_RE.sub(" ", s).strip()
            if s:
                out.append(s)
    return out
```

## LLM rewrite spec

`rewrite_hapax_flash.py`의 system prompt 핵심:

1. **두 가지 출력만 허용**: 더 쉬운 rewrite OR 빈칸 `""`. **동일 sentence echo 금지**.
2. **단순화 기준** — 다음 중 ≥1 개선되어야 함:
   - 희귀어 → 흔한 동의어 (`infrastructure` → `roads and pipes`)
   - 5세 어휘 위반 사이드 단어 동시 단순화
   - 긴 절을 짧은 문장으로 분할
   - 추상/공식 표현 → 구체/구어
3. **5세 어휘 strict** (~2000 단어, Dolch + 1st-grade)
4. **빈칸 케이스**:
   - 고유명사 주어 (`Beyoncé`, `Popocatépetl`, `Notre-Dame`)
   - 외국어 학습 시연 (`Tschüss is German for goodbye`)
   - 의성어 (`sssssss`, `cu-ckoo`)
   - 어원/철자 lesson
   - 5세 대체불가 과학 용어 (`photosynthesis`가 sentence의 핵심)

코드 측면 echo 방어:

```python
if rw and rw == sent_by_word.get(w, ""):
    rw = ""  # 동일문장은 강제 빈칸
```

Prompt만으로는 강한 모델이 lazy echo하는 경우(Flash 1,613건, Pro 370건)가 있어 코드 측면에서 한 번 더 차단.

## 3 cycle 결과

| Cycle | 모델 | 입력 hapax | rewrites | 적용 | rare_hapax 잔여 |
|---|---|---:|---:|---:|---:|
| 1 | Flash | 4,525 | 3,897 (echo 다수 포함) | 1,877 | 2,648 |
| 1.5 | rebuild | (corpus 재집계) | — | — | 3,193 |
| 2 | Flash + 강화 prompt + echo filter | 3,193 | 481 | 476 | 2,717 |
| 3 | **Pro** + 강화 prompt | 2,717 | 1,004 | 972 | **1,745** |

- 누적 적용 **3,325건** sentence (corpus in-place)
- hapax **4,525 → 1,745** (-61.4%)
- 누적 비용: $0.05 (Flash) + $0.04 (Flash 강화) + $0.11 (Pro) = **~$0.20**

Cycle 1 vs 2 비교: 강화 prompt + echo filter 적용 후 rewrites 수가 3,897 → 481로 줄었지만 quality는 상승. Cycle 1의 3,897 중 1,837은 sentence 그대로 echo한 lazy 응답이었음.

Cycle 2 vs 3 비교: 같은 prompt에 모델만 Flash → Pro로 교체. Pro가 단순한 sentence에서도 적극적 단순화 시도(rewrites 481 → 1,004, 2.1×). Pro는 Flash 대비 ~3× 비싸지만 hapax 정제 task엔 cost-effective.

## 변경 패턴 (대표 예시)

| 카테고리 | 원문 | rewritten |
|---|---|---|
| 오타 자동 교정 | `Botony teaches me...` | `Botany teaches about...` |
| 외래어 영어화 | `The platos are in the box.` | `The plates are in the box.` |
| 5세 어휘 대체 | `The campus looks exciting...` | `The school area looks exciting...` |
| 외래악기 평이어화 | `I see two djembes on the floor.` | `I see two hand drums on the floor.` |
| 도구 평이어화 | `People put butane in lighters...` | `People put butane in fire starters...` |
| 복합어 풀어쓰기 | `Those are store-bought medicines.` | `Those are medicines you buy at a store.` |
| 절 분할 | `Discrimination is when people are not treated fairly because of who they are.` | `Some people are mean to others just because they look different. That is not fair.` |

## skip 사유 분석 (Pro cycle 기준)

빈칸 1,713건의 카테고리:
- **고유명사** (지명/인명/브랜드): Beyoncé, Popocatépetl, Notre-Dame, Hokkaidō, Kyrgyz, Wollstonecraft, FC Bayern 등
- **외래어 학습 시연**: Tschüss (독일어), olá (포르투갈어), dzień dobry (폴란드어), trdelník (체코어), 우리말 (한국어)
- **의성어**: sssssss(뱀), cu-ckoo(뻐꾸기), brr-brr, beeeeh, hoo-ray
- **어원/철자 lesson**: `Eur+Asia=Eurasia`, 시저 암호 `HELLO→IFMMP`, 룬 알파벳 `fuark`
- **다이아크리틱 잘림** (콘텐츠 결함): `rmqi`(Ürümqi), `Spmi`(Sápmi), `stika`/`nstika`(āstika/nāstika) — 3건만, 해당 라인 수동 수정 후보

## 한계

1. **wiki.txt의 `title\n\nbody` 첫 문장 일부 skip** — sentence가 title을 포함해 길어져 substring 매치 실패. Cycle 1에서 다수 skip (apply skip 8.9%), 이후 rebuild로 boundary 확장하며 해결.
2. **`<|turn|>` 가로지른 sentence 미보존** — refine 이후 해결.
3. **하이픈 합성어 일부 미처리** — `Notre-Dame`의 `dame`처럼 word boundary 룰상 단독 매치 X. apply skip 발생.
4. **LLM 판단 의존** — 외래어/시연 의도가 spec에 명시되지 않으면 LLM이 외래어를 영어화할 위험. Few-shot으로 완화.

## 다음 단계

- 잔여 1,745 hapax 추가 cycle (Pro 1회 더 또는 사람 검수)
- diacritic 누락 3건 수동 수정 (`rmqi`, `Spmi`, `stika`/`nstika`)
- BPE 재학습 후 unique vocab 변화 확인
- 다음 학습 cycle에서 정성 평가 비교 (v2048 baseline vs hapax 정제 후)

## Critical Files

신규:
- `docs/ccmc-hapax-refinement.md` (이 문서)
- `scripts/{review_vocab,review_hapax_flash,find_typos,verify_suspects,add_hapax_sentences,refine_hapax_sentences,rebuild_rare_hapax,rewrite_hapax_flash,apply_rewrites}.py`

수정:
- `data/ccmc-all-raw/*.txt` (7개 corpus 파일) — 누적 3,325 sentence 단순화

임시 (gitignored 대상 후보):
- `data/ccmc-all-raw/_vocab_review/` — 검토 자산 (rare_hapax.tsv 등)
