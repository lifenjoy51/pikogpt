# CCMC v6 — Multi-axis (A/B/C/D) anchor 종속 합성 계획 (2026-05-12)

## 배경

v2048 학습 (300k params, vocab=2048, val 2.81, bpc 1.035) 정성 평가에서 5개 단점이 명확히 관찰됨:

1. **비현실 구문** ("buy teeth at school") — fact grounding 부족
2. **의미 반복** ("interesting because it is interesting") — 빈 동어반복
3. **주제 점프** — paragraph 안 응집력 부족
4. **Pronoun 혼란** ("my dad's toy car... he... she...") — referent chain 부족
5. **숫자 grounding 약** ("buy two toys for the toy") — 수량 표현 부족

표면 영어는 자연스러우나 "surface form 흉내" 수준. 단어 분포·짧은 문법은 학습했으나 **사실 grounding · 인과관계 · reference chain · 양적 추론**이 부족.

대응: 부족 카테고리를 LLM 합성으로 신규 보강하되, lemma anchor universe 6,145개에 대해 **4축(A/B/C/D)을 anchor-종속으로 합성**. 같은 anchor가 4가지 다른 글 패턴에 등장해 어휘 grounding + 문체 패턴이 동시에 강화됨. 각 단점을 1:N axis에 명시적으로 매핑해 학습 후 단점별로 효과 측정.

## Anchor universe 통합

| 출처 | unique anchor | ARTICLE 보유 |
|---|---:|---|
| `data/ccmc-all-raw/lemma_anchors.tsv` (lemma 컬럼, lowercase) | 3,288 | 1,083 (wiki와 정확매칭) |
| `data/ccmc-all-raw/wiki.txt` titles (lowercase) | 3,940 | 3,940 (전부) |
| **합집합 (deduped)** | **6,145** | 3,940 (64%) / 없음 2,205 (36%) |

source 필드: `both` (1,083) / `lemma_only` (2,205) / `wiki_only` (2,857).

ARTICLE join — `both`+`wiki_only` anchor는 `data/wiki-topic-filter/judgments.jsonl`의 id를 통해 `data/external-all-raw/wiki.txt`의 line number(1~8942)로 본문 lookup. `BODY_CHAR_CAP=8000` 적용 (p90≈5,872, p99≈19,541 → 상위 10% trim).

category 필드 — lemma_anchors의 26개 taxonomy 우선, 없으면 judgments의 14개 taxonomy. category는 hint일 뿐 hard 룰 아님.

## 4축 spec 설계

각 axis spec 공통 출력 스키마: `{"text": "<...>", "reason": null}` 또는 `{"text": null, "reason": "<≤20자 Korean>"}`.

**공통 정책: No fixed length.** 길이는 anchor 정보량과 LLM 판단에 위임 (기존 `prompts/wiki_synth_system.txt` 정책 일관). hard 룰의 패턴 등장 횟수만 강제, padding/truncation 금지.

### A. wiki v2 (기존 `prompts/wiki_synth_system.txt` 강화)

- 형식: explainer, 첫 문장 정의. No fixed length.
- hard 룰:
  - **fact pattern ≥3** (IS / LOOKS / WHERE / DOES / EXAMPLE 중)
  - **referent 일관**: 첫 도입 후 it/they/he/she 중 하나로 고정, switching 금지
  - **빈 동어반복 금지**: tautology 패턴 (`X is X because it is X`) 차단
  - **5세 어휘 strict**: Dolch + 1st-grade reading ~2,000 단어
  - **ARTICLE 분기**: 있으면 fact 출처, 없으면 LLM 사전지식 (디테일 dates/names/numbers 최소화)
- skip: 거의 없음 (보장 axis — 6,145 중 ~97%)
- few-shot: 기존 Cat/Rain/Finance 3개 유지 + ARTICLE-less 예시 1개 신규

### B. 인과·절차 (신규)

- 형식: anchor가 인과 chain 또는 절차의 노드.
- hard 룰:
  - "If X, then Y" / "Because X, Y" / "First ~ then ~ last ~" 중 ≥1회
  - anchor가 인과/절차 내 등장 + 5세 어휘
- skip: function word / 인과 부적합 추상어 (예: `above`, `all` 같은 전치사·기능어)

### C. chained narrative (신규)

- 형식: short story, anchor가 main character 또는 key object.
- hard 룰:
  - 일관 main character, 도중 새 인물/객체 도입 금지
  - 도입 후 he/she/it/they로 anchor 또는 character를 **최소 2회 pronoun 재참조**
  - 5세 어휘
- skip: actor/object 불가능한 경우 (function word 등)

### D. counting (신규)

- 형식: 수량 grounding.
- hard 룰:
  - 숫자 1-10 중 ≥2회 등장
  - 비교어 (more / fewer / less / same / equal) ≥1회
  - anchor는 셀 수 있는 객체
- skip: 셀 수 없는 추상/감정/기능어/manner adverb / mass noun

## 호출 인프라

`scripts/run_wiki_synth.py` 패턴 재사용:
- ThreadPoolExecutor 16 worker
- OpenRouter, `reasoning: {enabled:false, exclude:true}`, provider DeepSeek 강제, `response_format: json_object`, `usage: {include:true}`
- `.env` 자동 로드 (`./.env` → `../llm-playground/.env`)
- 재시도 없음, `failed_{axis}.jsonl` 별도 보존
- `progress_{axis}.json` resume
- cost cap 도달 시 pending cancel
- axis 인자 (`--axis a|b|c|d` 또는 별도 4 스크립트)
- ARTICLE escape unescape (`body.replace("\\n", "\n")`) 그대로

## 호출 수·비용·시간

| axis | 시도 | skip률 추정 | records | 평균 output char | input tok/call |
|---|---:|---:|---:|---:|---:|
| A | 6,145 | 3% | ~5,960 | 460 | both/wiki_only ≈ 3,400 / lemma_only ≈ 1,400 |
| B | 6,145 | 18% | ~5,040 | 200 | ~1,100 |
| C | 6,145 | 14% | ~5,290 | 400 | ~1,100 |
| D | 6,145 | 56% | ~2,700 | 150 | ~1,100 |
| **합** | **24,580** | **27%** | **~18,990** | — | — |

input 약 36.6M tok / output 약 3.2M tok 추정.

**모델: `deepseek/deepseek-v4-pro`**

이전 Flash 1,000개 generation에서 instruction 일부 무시 사례 있음. axis spec은 hard 룰 + skip 룰 + few-shot이 복잡하고 ARTICLE-less anchor 2,205개에서 fact 정확도 우선이라 Pro 전체 선택.

- 비용: **~$42**
- 시간: **~17h** (16 worker × 평균 10초/Pro 호출 × 24,580 calls)
- autonomous run (도중 cost-cap / progress.json resume / failed.jsonl로 안전 처리)

## 파일 구조

```
prompts/
  v6_axis_a_wiki.txt        # A axis (wiki_synth_system.txt 강화 + ARTICLE optional + referent hard 룰 + 빈 반복 금지)
  v6_axis_b_cause.txt       # B axis (인과·절차)
  v6_axis_c_chained.txt     # C axis (chained narrative)
  v6_axis_d_counting.txt    # D axis (counting)

scripts/
  build_axis_universe.py    # lemma_anchors ∪ wiki titles → universe.jsonl + ARTICLE join
  run_v6_axes.py            # axis 인자, parallel 16 worker, skip 처리, resume
  build_v6_output.py        # raw_{a,b,c,d}.jsonl → 4 text files + stats.json

data/v6-axes/
  universe.jsonl            # 6,145 anchor — fields: anchor, source, category, article (or null)
  raw_a.jsonl, raw_b.jsonl, raw_c.jsonl, raw_d.jsonl
  progress_{a,b,c,d}.json
  failed_{a,b,c,d}.jsonl
  stats.json                # axis × category × skip_rate 매트릭스, reason 분포
  output/
    wiki_v2.txt             # A → title\n\nbody literal \n (기존 wiki.txt 포맷 유지)
    cause_seq.txt           # B → 1 line = 1 paragraph (normalize 적용)
    chained.txt             # C → 1 line = 1 short story (normalize 적용)
    counting.txt            # D → 1 line = 1 paragraph (normalize 적용)
```

## ccmc-all-raw 합본 변화

**기존 wiki.txt → wiki_v1.txt 리네임 + wiki_v2.txt 신규 (병행)**

| 파일 | 변화 | records | 비고 |
|---|---|---:|---|
| `lemma_sentences.txt` | 유지 | 3,209 | |
| `stories.txt` | 유지 | 41,619 | |
| `dialogues.txt` | 유지 | 25,512 | |
| `wiki.txt` → `wiki_v1.txt` | 리네임만 (내용 동일) | 3,940 | 명시성 + v3+ 확장 자연 |
| `wiki_v2.txt` | 신규 | ~5,960 | A axis 산출, title\n\nbody literal \n 포맷 |
| `cause_seq.txt` | 신규 | ~5,040 | B axis |
| `chained.txt` | 신규 | ~5,290 | C axis |
| `counting.txt` | 신규 | ~2,700 | D axis |

총 records 70,280 → ~93,270 (8 파일). token: 5.6M → ~7.6M (wiki_v1 0.9M 유지 + wiki_v2 1.0M + 신규 3 파일 1.1M = +2.0M).

같은 anchor에 대해 wiki_v1 + wiki_v2 두 explainer가 학습 데이터에 공존. 학습 신호는 모순 없는 보강(둘 다 같은 anchor의 5세 explainer, v2가 referent/fact pattern hard 룰 강화). 학습 후 정성평가에서 wiki_v2 hard 룰 효과가 v1 noise를 압도하는지 측정.

`scripts/build_ccmc_all_raw.py` 확장:
- 새 `build_cause_seq()`, `build_chained()`, `build_counting()` 함수 — 기존 `normalize()` 재사용 (`<|bos|>`, `<|eos|>`, `<|sep|>` 제거 + `\n` → 공백)
- wiki_v1/v2는 별도 (외부에서 원형 복사, normalize 미적용 — title\n\nbody literal \n 형식 유지)
- wiki.txt 리네임은 `git mv` 또는 build 스크립트가 wiki_v1.txt로 출력
- `main()`에 신규 3 build 호출 추가

## 단계별 진행

| 단계 | 작업 | 비고 |
|---|---|---|
| 0. docs 문서화 | 이 문서 (`docs/ccmc-v6-multi-axis-plan.md`) | git commit-ready |
| 1. Universe 빌드 | `scripts/build_axis_universe.py` 구현 + 실행 → `data/v6-axes/universe.jsonl` | 6,145 anchor + ARTICLE join 검증 |
| 2. Spec 일괄 작성 | `prompts/v6_axis_{a,b,c,d}.txt` 4개 한꺼번에 | 사용자 검토 1회 |
| 3. Dry-run 64건 | anchor 16개 sampling × 4 axis = 64 Pro 호출 | HTML viewer로 spot-check, ~$0.5 |
| 4. Spec 튜닝 | 어색 응답 / over-skip / under-skip 보고 hard 룰·few-shot 보강 | 필요 시 cycle |
| 5. Full run | axis 순차 24,580 Pro 호출 (resume 가능) | ~$42 / ~17h autonomous |
| 6. Build text | `scripts/build_v6_output.py` 4 text 생성 + `stats.json` | skip reason 분포 분석 |
| 7. ccmc-all-raw 통합 | wiki.txt → wiki_v1.txt 리네임 + wiki_v2.txt 신규 + 3 신규 파일 + `build_ccmc_all_raw.py` 확장 + README 갱신 | |
| 8. BPE 재학습 | vocab=2048 유지 또는 v4096 신규 | ~3min |
| 9. 다음 학습 cycle | 1M params, worker 4, 7.6M token corpus | ~30h (별개) |

## 평가 지표 (학습 후)

axis별 보강 효과 측정:

- **A**: paragraph 안 fact pattern 등장 (n-gram `is a`, `looks like`, `lives in`, `used for`), 빈 동어반복 n-gram (`X because X`) 빈도
- **B**: connective n-gram (`if X`, `so we`, `first`, `then`, `last`) 등장 빈도
- **C**: paragraph 안 referent consistency — 첫 도입 후 pronoun switching 비율
- **D**: 숫자 + 객체 collocation 자연도 (`two cats`, `three of`) 정확도

baseline은 v2048 학습 정성 평가 (T=0.5 샘플).

## Critical Files

신규:
- `docs/ccmc-v6-multi-axis-plan.md` (이 문서)
- `prompts/v6_axis_a_wiki.txt`, `v6_axis_b_cause.txt`, `v6_axis_c_chained.txt`, `v6_axis_d_counting.txt`
- `scripts/build_axis_universe.py`, `run_v6_axes.py`, `build_v6_output.py`
- `data/v6-axes/` 전체

수정:
- `scripts/build_ccmc_all_raw.py` — `build_cause_seq()`, `build_chained()`, `build_counting()` 추가, `main()` 호출 확장
- `data/ccmc-all-raw/wiki.txt` → `wiki_v1.txt` 리네임
- `data/ccmc-all-raw/wiki_v2.txt`, `cause_seq.txt`, `chained.txt`, `counting.txt` — 신규
- `data/ccmc-all-raw/README.md` — v6 4축 항목 추가, 정규화 예외 명시

재사용:
- `scripts/run_wiki_synth.py:138-204` — `call_one()` 패턴 (요청·응답·재시도 정책)
- `scripts/run_wiki_synth.py:64-81` — `.env` 로딩
- `scripts/run_wiki_synth.py:213-242` — `build_wiki_txt()` (A axis wiki_v2 빌드 시 그대로 재사용)
- `data/wiki-topic-filter/judgments.jsonl` (keep=true 3,940 anchor ARTICLE id mapping)
- `data/external-all-raw/wiki.txt` (8,942 lines, 평균 2,684 char/line, BODY_CHAR_CAP=8,000 trim)
- `prompts/wiki_synth_system.txt` (A axis spec의 base)

## 검증

1. Universe: 합집합 6,145 ± 약간, ARTICLE join 3,940 == keep=true 일치 검증
2. Dry-run 64건: spot-check (사람이 1건씩 확인), spec hard 룰 적합 / skip 적정성 / 5세 어휘 위반 없음
3. Full run 후 stats.json: axis별 skip률이 추정과 ±10% 안 / category × axis 매트릭스에 이상 outlier 없음
4. raw text 빌드: line 수 / 평균 char 추정과 일치
5. ccmc-all-raw 합본: line 수 70,280 → ~93,270 ± 1k
6. BPE 재학습 후 unique_words.txt 신규 vocabulary coverage
7. 다음 학습 후 정성 평가 — 위 4개 지표

## 결정 사항

1. **합성 모델**: Pro 전체 (`deepseek/deepseek-v4-pro`)
2. **기존 wiki**: 병행 (wiki_v1 + wiki_v2)
3. **spec 순서**: 4개 일괄 작성 후 dry-run 64건 → cycle 튜닝 → full run

## 학습 cycle 결정사항 (별개 cycle)

- BPE 재학습: v2048 유지 vs v4096 신규 — 7.6M token 코퍼스에 v4096이 자연스러우나 v2048과 직접 비교도 가치 있음
- 모델 크기: 다음 1M params (현재 300k의 ~3배)
- Chinchilla 20:1 → 1M params엔 20M token 권장, 7.6M은 ~7배 부족 → epoch 늘림 또는 추가 합성 cycle

## 영향 받지 않는 것

- 기존 lemma_sentences / stories / dialogues 3개 파일 — 유지
- 기존 ckpt (v256/v800/v1024/v2048) — baseline 보존
- external-all-raw — 사용 안 함

## 작업량 추정

| 단계 | 시간 |
|---|---|
| Universe 빌드 + spec 4개 일괄 작성 (사람) | 3~5h |
| Dry-run 64건 + 튜닝 1-2 cycle | 2~3h (Pro dry-run ~10분 × 2 cycle + 사용자 검토) |
| Full run (Pro autonomous) | ~17h |
| Build text + ccmc-all-raw 통합 (wiki 리네임 포함) | 30min |
| BPE 재학습 | ~3min |
| 다음 학습 cycle (1M params) | ~30h (별개) |

**총 합성+통합 ~23~25h** (Pro full run이 지배적). 학습 cycle 포함 시 ~2.5일.
