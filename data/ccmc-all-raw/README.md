# CCMC all raw — v2/v3/v4/v5 합본 (텍스트 전용, 내용 기준 그룹화)

생성: `scripts/build_ccmc_all_raw.py`
원천: `/Users/joey51/works/llm-playground/data/processed/ccmc_v*/`

`data/external-all-raw/{dict,wiki,conv}.txt`와 동일하게 **content type별로 그룹화**.

## 파일

| 파일 | records | 내용 | 출처 |
|---|---:|---|---|
| `lemma_sentences.txt` | 3209 | 1 line = 1 lemma의 짧은 문장 묶음 (lemma 어휘 grounding). v2_pro 4축 + v3_extra 3축. | v2_pro: multi_role+sensory+category+contrast (1824 lemma). v3_extra: narrative+scenario+action_sequence (1385 lemma) |
| `stories.txt` | 41619 | 1 line = 1 short story (5-10 문장). DeepSeek v4 Pro+Flash 합성. dedup 적용. | v4_tinystories 9개 raw.jsonl (41619 입력) |
| `dialogues.txt` | 25512 | 1 line = 1 Q/A 또는 다턴 대화. turn 경계는 ` <\|turn\|> `. | v2_pro qa축 pair별 분리 (10815) + v5_qa_dialogues (14697) |

## 정규화 공통

- `<\|bos\|>`, `<\|eos\|>`, `<\|sep\|>` → 제거
- 리터럴 `\n` / 실제 newline → 공백 (CCMC raw에는 paragraph 구조 없음)
- `# Title` 마커 → 제거 (CCMC에는 없지만 안전망)
- `<\|turn\|>` → ` <\|turn\|> ` 양쪽 공백 통일

## CCMC 합성 출처 (참고)

- v2_pro: 1,826 lemma × 5축. DeepSeek v2/Pro.
- v3_extra: 1,826 lemma × 3축. DeepSeek (v3 보강).
- v4_tinystories: 5,932 (verb,noun,adj) tuple × 9 generation. DeepSeek v4 Pro+Flash.
- v5_qa_dialogues: 14,780 spec → 4-7 turn QA. DeepSeek v4 Flash 5:Pro 2.
