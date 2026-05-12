# CCMC all raw — v2/v3/v4/v5/v6 합본 (텍스트 전용, 내용 기준 그룹화)

생성:
- `lemma_sentences.txt` / `stories.txt` / `dialogues.txt` : `scripts/build_ccmc_all_raw.py`
  - 원천: `/Users/joey51/works/llm-playground/data/processed/ccmc_v*/`
- `wiki.txt` : v1 합성(`scripts/run_wiki_synth.py` 산출 3,940건) + v6 A axis 합성(`scripts/run_v6_axes.py --axis a` 산출 6,142건) 합본. 표제어 보존을 위해 ccmc 일반 정규화 미적용 (external-all-raw/wiki.txt 컨벤션과 동일하게 `title\n\nbody` 형식 유지)
  - 원천 (v1): `data/external-all-raw/wiki.txt` 본문 (SimpleWiki) + `data/wiki-topic-filter/kept_titles.txt` (anchor, 3,940)
  - 원천 (v6 A axis): `data/v6-axes/universe.jsonl` anchor 6,145개 (lemma_anchors ∪ wiki titles) + ARTICLE 3,940 join. referent 일관 / fact pattern ≥3 / 빈 반복 금지 hard 룰 강화. 자세한 흐름: `docs/ccmc-v6-multi-axis-plan.md`
- `cause_seq.txt` / `chained.txt` / `counting.txt` : v6 B/C/D axis 합성 (`scripts/run_v6_axes.py --axis b|c|d`, `scripts/build_v6_output.py`).
  - 원천: `data/v6-axes/universe.jsonl` anchor 6,145 (lemma_anchors ∪ wiki titles). ARTICLE은 사용 안 함 — anchor와 axis 패턴만으로 합성.
  - hard 룰: B 인과·절차 connective ≥1, C pronoun 재참조 ≥2 + 일관 character, D 숫자 ≥2 + 비교어 ≥1.

`data/external-all-raw/{dict,wiki,conv}.txt`와 동일하게 **content type별로 그룹화**.

## 파일

| 파일 | records | 내용 | 출처 |
|---|---:|---|---|
| `lemma_sentences.txt` | 3209 | 1 line = 1 lemma의 짧은 문장 묶음 (lemma 어휘 grounding). v2_pro 4축 + v3_extra 3축. | v2_pro: multi_role+sensory+category+contrast (1824 lemma). v3_extra: narrative+scenario+action_sequence (1385 lemma) |
| `stories.txt` | 41619 | 1 line = 1 short story (5-10 문장). DeepSeek v4 Pro+Flash 합성. dedup 적용. | v4_tinystories 9개 raw.jsonl (41619 입력) |
| `dialogues.txt` | 25512 | 1 line = 1 Q/A 또는 다턴 대화. turn 경계는 ` <\|turn\|> `. | v2_pro qa축 pair별 분리 (10815) + v5_qa_dialogues (14697) |
| `wiki.txt` | 10082 | 1 line = 1 wiki-style 5세 explainer. **표제어 보존 형식**: `title\n\nbody` (literal `\n`, external-all-raw/wiki.txt와 동일). 외부 SimpleWiki 의존 없는 합성본. v1(3,940) + v6 A axis(6,142) 합본. | v1: SimpleWiki 8,942 제목 → DeepSeek v4 Flash 5세 적합 필터(3,940 keep) → 본문 통째 입력해 5세 explainer 합성. v6 A: lemma_anchors ∪ wiki titles 6,145 anchor (ARTICLE 3,940 / 없음 2,205) → DeepSeek v4 Pro로 referent 일관·fact pattern≥3·빈 반복 금지 hard 룰 강화 explainer 합성 (ok 6,142 / skip 2 / fail 1). 자세한 흐름: `docs/ccmc-v6-multi-axis-plan.md` |
| `cause_seq.txt` | 5680 | 1 line = 1 인과·절차 paragraph. anchor가 cause-effect 또는 step-by-step의 노드. | v6 B axis: 6,145 anchor → DeepSeek v4 Flash로 `If/then/so/because` 또는 `First/then/last` 패턴 ≥1회 hard 룰. ok 5,680 / skip 464 (function word·인과 부적합) / fail 1. spec: `prompts/v6_axis_b_cause.txt` |
| `chained.txt` | 5834 | 1 line = 1 short story. anchor가 main character / key object / feeling. pronoun chain 강제. | v6 C axis: 6,145 anchor → DeepSeek v4 Flash로 일관 character + pronoun 재참조 ≥2회 hard 룰. ok 5,834 / skip 311 (function word) / fail 0. spec: `prompts/v6_axis_c_chained.txt` |
| `counting.txt` | 2747 | 1 line = 1 수량 grounding paragraph. anchor가 셀 수 있는 객체. | v6 D axis: 6,145 anchor → DeepSeek v4 Flash로 숫자 1-10 ≥2회 + 비교어 ≥1회 hard 룰. ok 2,747 / skip 3,398 (셀 수 없는 추상/감정/기능어/manner adverb / mass noun) / fail 0. spec: `prompts/v6_axis_d_counting.txt` |
| `lemma_anchors.tsv` | 3288 | 5세 적합 lemma anchor 통합본 (lemma + category + source). v6 합성용 입력. TSV 헤더: `lemma\tcategory\tsource` | v2_pro/v3_extra lemma 1826 + lemma-anchor-filter 신규 1462 (LLM 판정 1195 + 사용자 수동 267). 발굴/판정 흐름은 `data/lemma-anchor-filter/` 참조 |

## 정규화 공통 (lemma_sentences / stories / dialogues / cause_seq / chained / counting)

- `<\|bos\|>`, `<\|eos\|>`, `<\|sep\|>` → 제거
- 리터럴 `\n` / 실제 newline → 공백 (이 파일들은 표제어/paragraph 구조가 없음)
- `# Title` 마커 → 제거 (CCMC에는 없지만 안전망)
- `<\|turn\|>` → ` <\|turn\|> ` 양쪽 공백 통일
- cause_seq / chained / counting은 `build_v6_output.py`의 `normalize_inline()`로 합성 단계에서 정규화 적용 (newline → 공백, multiple space → single)

**wiki.txt는 예외**: 표제어가 있는 자료라 external-all-raw/wiki.txt 컨벤션을 따라 `title\n\nbody` (literal `\n`) 형식을 그대로 유지. ccmc 일반 정규화를 적용하면 표제어/본문 경계가 사라지므로 적용하지 않음.

## CCMC 합성 출처 (참고)

- v2_pro: 1,826 lemma × 5축. DeepSeek v2/Pro.
- v3_extra: 1,826 lemma × 3축. DeepSeek (v3 보강).
- v4_tinystories: 5,932 (verb,noun,adj) tuple × 9 generation. DeepSeek v4 Pro+Flash.
- v5_qa_dialogues: 14,780 spec → 4-7 turn QA. DeepSeek v4 Flash 5:Pro 2.
- wiki_synth: 3,940 SimpleWiki 제목 (5세 적합 anchor) × 본문 통째 → 5세 explainer. DeepSeek v4 Flash. (`scripts/run_wiki_synth.py`, anchor: `data/wiki-topic-filter/kept_titles.txt`)
- v6_axis_a (wiki v2): 6,145 anchor (lemma_anchors ∪ wiki titles) → 5세 explainer (referent 일관 / fact pattern ≥3 / 빈 반복 금지 hard 룰). DeepSeek v4 Pro. (`scripts/run_v6_axes.py --axis a`, anchor: `data/v6-axes/universe.jsonl`, spec: `prompts/v6_axis_a_wiki.txt`)
- v6_axis_b (cause_seq): 6,145 anchor → 인과·절차 paragraph (`If/then/so/because` 또는 `First/then/last` ≥1회). DeepSeek v4 Flash. (`scripts/run_v6_axes.py --axis b`, spec: `prompts/v6_axis_b_cause.txt`)
- v6_axis_c (chained): 6,145 anchor → short story (일관 character + pronoun 재참조 ≥2회). DeepSeek v4 Flash. (`scripts/run_v6_axes.py --axis c`, spec: `prompts/v6_axis_c_chained.txt`)
- v6_axis_d (counting): 6,145 anchor → 수량 grounding paragraph (숫자 1-10 ≥2회 + 비교어 ≥1회). DeepSeek v4 Flash. (`scripts/run_v6_axes.py --axis d`, spec: `prompts/v6_axis_d_counting.txt`)
