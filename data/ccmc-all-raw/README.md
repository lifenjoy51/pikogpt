# CCMC all raw — v2/v3/v4/v5 합본 (텍스트 전용, 내용 기준 그룹화)

생성:
- `lemma_sentences.txt` / `stories.txt` / `dialogues.txt` : `scripts/build_ccmc_all_raw.py`
  - 원천: `/Users/joey51/works/llm-playground/data/processed/ccmc_v*/`
- `wiki.txt` : `scripts/run_wiki_synth.py` 산출(`data/wiki-synth/wiki.txt`)을 원형 그대로 복사. 표제어 보존을 위해 ccmc 일반 정규화 미적용 (external-all-raw/wiki.txt 컨벤션과 동일하게 `title\n\nbody` 형식 유지)
  - 원천: `data/external-all-raw/wiki.txt` 본문 (SimpleWiki) + `data/wiki-topic-filter/kept_titles.txt` (anchor)

`data/external-all-raw/{dict,wiki,conv}.txt`와 동일하게 **content type별로 그룹화**.

## 파일

| 파일 | records | 내용 | 출처 |
|---|---:|---|---|
| `lemma_sentences.txt` | 3209 | 1 line = 1 lemma의 짧은 문장 묶음 (lemma 어휘 grounding). v2_pro 4축 + v3_extra 3축. | v2_pro: multi_role+sensory+category+contrast (1824 lemma). v3_extra: narrative+scenario+action_sequence (1385 lemma) |
| `stories.txt` | 41619 | 1 line = 1 short story (5-10 문장). DeepSeek v4 Pro+Flash 합성. dedup 적용. | v4_tinystories 9개 raw.jsonl (41619 입력) |
| `dialogues.txt` | 25512 | 1 line = 1 Q/A 또는 다턴 대화. turn 경계는 ` <\|turn\|> `. | v2_pro qa축 pair별 분리 (10815) + v5_qa_dialogues (14697) |
| `wiki.txt` | 3940 | 1 line = 1 wiki-style 5세 explainer. **표제어 보존 형식**: `title\n\nbody` (literal `\n`, external-all-raw/wiki.txt와 동일). 외부 SimpleWiki 의존 없는 합성본. | SimpleWiki 8942 제목 → DeepSeek v4 Flash로 5세 적합도 필터(3940 keep) → 본문 통째 입력해 5세 explainer 합성 (Flash) |
| `lemma_anchors.tsv` | 3288 | 5세 적합 lemma anchor 통합본 (lemma + category + source). v6 합성용 입력. TSV 헤더: `lemma\tcategory\tsource` | v2_pro/v3_extra lemma 1826 + lemma-anchor-filter 신규 1462 (LLM 판정 1195 + 사용자 수동 267). 발굴/판정 흐름은 `data/lemma-anchor-filter/` 참조 |

## 정규화 공통 (lemma_sentences / stories / dialogues)

- `<\|bos\|>`, `<\|eos\|>`, `<\|sep\|>` → 제거
- 리터럴 `\n` / 실제 newline → 공백 (이 세 파일은 표제어/paragraph 구조가 없음)
- `# Title` 마커 → 제거 (CCMC에는 없지만 안전망)
- `<\|turn\|>` → ` <\|turn\|> ` 양쪽 공백 통일

**wiki.txt는 예외**: 표제어가 있는 자료라 external-all-raw/wiki.txt 컨벤션을 따라 `title\n\nbody` (literal `\n`) 형식을 그대로 유지. ccmc 일반 정규화를 적용하면 표제어/본문 경계가 사라지므로 적용하지 않음.

## CCMC 합성 출처 (참고)

- v2_pro: 1,826 lemma × 5축. DeepSeek v2/Pro.
- v3_extra: 1,826 lemma × 3축. DeepSeek (v3 보강).
- v4_tinystories: 5,932 (verb,noun,adj) tuple × 9 generation. DeepSeek v4 Pro+Flash.
- v5_qa_dialogues: 14,780 spec → 4-7 turn QA. DeepSeek v4 Flash 5:Pro 2.
- wiki_synth: 3,940 SimpleWiki 제목 (5세 적합 anchor) × 본문 통째 → 5세 explainer. DeepSeek v4 Flash. (`scripts/run_wiki_synth.py`, anchor: `data/wiki-topic-filter/kept_titles.txt`)
