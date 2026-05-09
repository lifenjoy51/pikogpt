# CCMC v5_qa — WH/명령/감탄 anchor + 2-stage LLM 합성 계획 (2026-05-09)

## 배경

3M 모델 (val 1.84, v0022)가 narrative continuation은 잘하지만 **질문에 대한 답변
능력이 약하다**. 진단:

1. v2-pro stage2가 이미 `<|bos|>Q?<|turn|>A.<|turn|>...<|eos|>` 형식의 multi-turn
   QA 코퍼스로 학습에 포함되어 있음 → 모델은 **형식은 학습했지만 의미 매핑은 약함**
2. greedy decoding (T=0)에서 "in the park." 등 mode collapse → T=0.8 + topK=40 +
   topP=0.9 + repPenalty=1.15 조합으로 해소 검증됨 (학습 문제 아님)
3. wh- 질문에 답형이 부적절: Where→다른 위치, Why→무관 답, Who→사물 답
4. yes/no 질문도 narrative로 대체됨

→ **QA 비중 강화 + 발화 타입 다양성 확장** 필요. 단순히 wh-question만으로는 부족
하므로 명령/감탄/제안/polar 모두 학습시켜야 함.

## 데이터 자산

- 5932개 anchor tuple: `llm-playground/data/processed/ccmc_v4_tinystories/curated_tuples.jsonl`
  - 각 tuple = `{verb, noun, adj, anchor, anchor_pos}` (3-PoS 묶음)
  - 이미 v4 TinyStories 41,619편 생성에 사용된 검증된 tuple
- DeepSeek v4 (Pro/Flash) — OpenRouter 경유, implicit prefix cache 지원

## 합성 전략 — 2-stage LLM 파이프라인

### Stage 1: spec synthesis (LLM이 발화 spec 작성)

per-tuple 단일 호출. **배치 없음** (사용자 요구). 출력 schema:

```json
{
  "tuple_id": "T00001",
  "topic_thread": "1-sentence summary",
  "setting": "park|garden|kitchen|room|school|home|outside|...",
  "lemma_roles": {"<lemma>": "agent|patient|action|manner|location|quality|state"},
  "turns": [
    {
      "act": "imperative|exclamation|polar_q|wh_q|declarative|suggestion",
      "wh": "What|Where|When|Who|Why|How|null",
      "aux": "Do|Does|Did|Is|Are|Has|Have|Can|Could|Would|Should|Will|null",
      "marker": "Wow|Yes|No|Sure|Well|Oh|Of course|Maybe|null",
      "lemma_focus": "<lemma> or [<lemma>, ...]"
    }
  ],
  "expected_turn_count": 4-7,
  "rejection": null OR "abstract concept, hard for 5yo QA"
}
```

핵심 아이디어: LLM이 "이 lemma 묶음에 어떤 발화 타입 분포가 자연스러운지" 판단 →
의미적으로 정당한 spec 생성. 단순 템플릿보다 풍부한 결과 기대.

### Stage 2 (예정): QA synthesis

Stage 1 spec을 따라 실제 multi-turn QA 텍스트 생성. 각 turn의 act/aux/marker/lemma
지시 사항을 따라 자연스러운 대화 생성. (현재 미구현 — Stage 1 결과 보고 진행 결정)

## Stage 1 구현 — 3중 JSON 보장 + 자가 수정 retry

```python
# OpenRouter extra_body 핵심 (기존 ccmc_unified_v2 task 패턴 재사용)
extra={
    "provider": {"only": ["DeepSeek"], "allow_fallbacks": False},
    "reasoning": {"enabled": False, "exclude": True},   # ★ Pro reasoning 비활성 (cost 1/3)
    "response_format": {"type": "json_object"},
}
```

```python
# Pydantic schema + field_validator 정규화
@field_validator("wh", "aux", "marker", mode="before")
def _norm(cls, v):
    if v in (None, "", "null", "None"): return None         # "null" 문자열 → None
    return v.split()[0].capitalize() + " ".join(v.split()[1:])  # "is" → "Is"
```

```python
# 검증 실패 시 에러 메시지를 다음 호출 user prompt에 첨부 → LLM 자가 수정
def call_with_retry(client, model_id, tup, max_retry=2):
    for attempt in range(max_retry + 1):
        text, meta = call_model(client, model_id, tup, error_feedback=last_err)
        spec, err = validate_spec(text, tup)
        if spec and not err: return spec, ...
        last_err = err
```

## Pilot 검증 (3회 반복)

| 회차 | n×model | reasoning | retry | Flash 성공 | Pro 성공 | Flash 단가 | Pro 단가 |
|---|---|---|---|---|---|---|---|
| v1 | 5×2 | ON (default) | ❌ | 3/5 (60%) | 3/5 (60%) | $0.0003 | $0.0014 |
| v2 | 5×2 | ON (잘못된 키) | ❌ | 5/5 | 5/5 | $0.00059 | $0.00255 |
| v3 | 3×2 | **OFF (정확한 키)** | ✓ (max=2) | 3/3 (1차) | 3/3 (1차) | **$0.0001** | **$0.00033** |

**v1 → v3 개선**:
- `reasoning.max_tokens=0` (잘못) → `reasoning.enabled=False + exclude=True` (정답)
- reasoning_tokens: 1100~4192 → **0**
- Flash 비용 -83%, Pro 비용 -87%
- 응답 시간 86~88% 단축 (Pro 90s → 10s)
- 100% 첫 시도 성공 (재시도 미발동)

## Pilot v3 spec 품질 (3 tuple × 2 model)

발화 타입 다양성 검증 — 모든 spec이 4~6 turn에 3종 이상 act 혼합:

| tuple | model | turn 수 | act 조합 | marker/aux |
|---|---|---|---|---|
| think/civilization/important | Flash | 4 | wh_q + imperative + polar_q + exclam | What/Well/Is/Wow |
| think/civilization/important | Pro | 6 | wh_q + declarative×2 + polar_q + exclam + imperative | What/Well/Is/Wow/Yes |
| learn/instrument/new | Flash | 5 | wh_q + declarative + imperative + polar_q + exclam | What/Are/Yes/Can/Wow |
| learn/instrument/new | Pro | 6 | exclam + wh_q + declarative + imperative + polar_q + declarative | Wow/What/Did/Well/Sure/Of course |
| love/planet/beautiful | Flash | 6 | 2×wh_q + 2×declarative + exclam + polar_q | What/Do/Yes/Where/Is/Oh/Wow |
| love/planet/beautiful | Pro | 5 | exclam + wh_q + declarative + imperative + polar_q | Wow/Why/Do/Well/Is |

**discourse marker 다양성**: Wow, Yes, Oh, Well, Sure, Of course, Maybe — 모두 등장.
**aux 다양성**: Do, Does, Did, Is, Are, Can, Will — 모두 등장.

## Full Run 설정 (시작: 2026-05-09)

사용자 결정: **5932 tuples 랜덤 셔플 후 5:2 split → Flash 단독 / Pro 단독 처리**
(에폭 다회 X, batch X)

| 모델 | tuples | workers | 단가 | 예상 비용 | 예상 시간 |
|---|---|---|---|---|---|
| Flash | 4237 (71.4%) | 16 | $0.0001 | **$0.42** | ~18min |
| Pro | 1695 (28.6%) | 8 | $0.00033 | **$0.56** | ~35min |
| **합계** | **5932** | — | — | **~$1** | **~55min** |

cost cap: $3 (안전 여유).

산출 디렉터리:
```
llm-playground/data/processed/ccmc_v5_qa_specs/
├── flash/{raw.jsonl, progress.json}
└── pro/{raw.jsonl, progress.json}
```

각 record:
```json
{"tuple_id": "T00001", "verb": "...", "noun": "...", "adj": "...",
 "model": "flash|pro", "ok": true, "attempts": 1,
 "spec": {...}, "meta": {"elapsed_s": 4.2, "usage": {...}}}
```

## 코드

- `llm-playground/tools/v5_qa_pilot_stage1.py` — pilot (n tuple 직접 지정, 단일 thread)
- `llm-playground/tools/v5_qa_stage1_full.py` — full run (5:2 split, ThreadPool, progress.json, cost cap)

## 다음 단계 (Stage 1 완료 후)

1. spec 풀 통계 분석:
   - act type 실제 분포 (LLM이 실제로 선택한 비율)
   - rejection 비율 + 이유
   - lemma_role 패턴 (noun=agent? verb=action?)
   - setting 클러스터링
2. Stage 2 pilot — 같은 spec으로 실제 multi-turn QA 텍스트 생성 (Flash/Pro 비교)
3. Stage 2 full run + validate_v5_qa 게이트 (어휘/형식/분포 검증)
4. pikogpt prep — stage2 형식 wrap (`<|bos|>Q?<|turn|>A...<|eos|>`)
5. **v0022(val 1.84)에서 finetune** — narrative 능력 보존하며 QA 능력 강화

## 위험·완화

| 위험 | 완화 |
|---|---|
| reasoning OFF로 spec 품질 저하 | Pilot v3에서 6/6 양질 spec 검증 (act 다양성·marker 다양성 OK) |
| Pro 비용 폭주 | reasoning.enabled=false + cost cap $3 + thread pool 8로 제한 |
| 재시도 폭주 | max_retry=2 (총 3회 시도) + 자가 수정 프롬프트 |
| LLM이 input lemma 외 단어 사용 | validate_spec에서 lemma_focus 검증 + 실패 시 retry |
| OpenRouter 일시 장애 | OpenRouterClient의 자체 backoff retry (3회) + future-level retry |

## 참고

- `docs/small-lm-semantic-learning-2026-05-07.md` — 전체 학습 컨텍스트 (3M 모델 학습 결과)
- v2-pro stage2 형식: `<|bos|>Question?<|turn|>Answer.<|turn|>Question?<|turn|>Answer.<|eos|>`
- DeepSeek implicit cache: 동일 system prompt + 비슷한 user prompt → cached_tokens 384 (모든 호출)
