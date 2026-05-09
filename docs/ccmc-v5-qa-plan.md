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

### Stage 2 (완료): QA synthesis

Stage 1 spec을 따라 실제 multi-turn QA 텍스트 생성. 각 turn의 act/aux/marker/lemma
지시 사항을 따라 자연스러운 대화 생성. **2026-05-09 완료** — 자세한 결과는
아래 "Stage 2 최종 결과" 섹션 참고.

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

## Stage 1 최종 결과 (2026-05-09 완료)

### 4단계 처리 흐름 (모든 5932 anchor가 양 모델 모두에서 spec 보유 목표)

| 단계 | 시점 | tuples | Flash ok | Pro ok | fail | 비용 | 시간 |
|---|---|---|---|---|---|---|---|
| **1. e1 full** (seed=42, schema 좁음) | 14:04 | 5932 | 4227 | 1692 | 13 | $0.974 | 60min |
| **2. retry_pro** (e1 fail 13건 Pro 재시도) | 14:35 | 13 | — | 12 | 1 | $0.005 | 3min |
| **3. e2 Flash only** (seed=142, schema 확장) | 14:48 | 4237 | 4201 | — | 36 | $0.492 | 30min |
|   ↳ Pro 단계는 시작 직전 watchdog kill (모델 다양성 확보 후 fill로 전환) |
| **4. fill** (Flash 미커버 479 + Pro 미커버 4228) | 15:43 | 4707 | 472 | 4176 | 59 | $1.560 | 104min |
| **누적** | — | — | **8900** | **5880** | **109** | **$3.03** | **~3.3h** |

### Schema 확장 (e2 + fill 적용)

기존 e1 fail 분석 결과 5세 영어 학습 코퍼스에 자연스러운 토큰이 schema에서 누락되어 있었음:

```python
# e1 (좁은 schema, 12+6+8+6 = 32 tokens)
ActType    = imperative|exclamation|polar_q|wh_q|declarative|suggestion        # 6
WhType     = What|Where|When|Who|Why|How                                        # 6
AuxType    = Do|Does|Did|Is|Are|Has|Have|Can|Could|Would|Should|Will            # 12
MarkerType = Wow|Yes|No|Sure|Well|Oh|Of course|Maybe                            # 8

# e2/fill (확장 schema, +20 토큰 = 51 tokens)
ActType    += greeting|apology|thanks                                           # +3
WhType     += Which|Whose                                                       # +2
AuxType    += Was|Were|Am|May                                                   # +4
MarkerType += Please|Thanks|Thank you|Sorry|OK|Okay|Hi|Hello|Hmm|Uh-oh          # +10
```

검증: e1 통과 spec 풀 5919건/30,711 turns 전수조사 결과 신규 토큰 0건 → e1 spec 호환성 유지.

### 누적 spec 풀

```
llm-playground/data/processed/ccmc_v5_qa_specs/
├── flash/      4227 ok  (e1 seed=42)
├── flash_e2/   4201 ok  (e2 seed=142, 확장 schema)
├── flash_fill/  472 ok  (Flash 미커버 채움)
├── pro/        1692 ok  (e1 seed=42)
├── retry_pro/    12 ok  (e1 fail Pro 재시도)
└── pro_fill/   4176 ok  (Pro 미커버 채움)
                 ─────
                 14,780 specs
```

### 모델 커버리지 (5932 anchor 중)
- **Flash 1+회**: 5,925 (99.9%)
- **Pro 1+회**: 5,880 (99.1%)
- **양 모델 모두 1+회**: ~5,880 (99.1%)

### 코드

- `llm-playground/tools/v5_qa_pilot_stage1.py` — schema/Pydantic/프롬프트/call_with_retry 정의
- `llm-playground/tools/v5_qa_stage1_full.py` — full run (`--suffix _e2`, `--seed 142`로 e2 재실행 가능)
- `llm-playground/tools/v5_qa_stage1_retry.py` — fail 레코드 Pro 재시도
- `llm-playground/tools/v5_qa_stage1_fill.py` — 미커버 anchor만 모델별 채움 (`--dry-run`으로 분류 미리 보기)

## Stage 2 최종 결과 (2026-05-09 완료)

### 파이프라인

| 단계 | 도구 | 입력 | 출력 |
|---|---|---|---|
| Pilot | `tools/v5_qa_pilot_stage2.py` | 10 spec random sample | `data/interim/v5_qa_pilot_stage2/results.jsonl` |
| Full | `tools/v5_qa_stage2_full.py --model flash` | 14,780 ok specs (6 dirs) | `data/processed/ccmc_v5_qa_dialogues/flash/raw.jsonl` |

### 사용자 결정

1. **입력 spec 풀**: 14,780 ok specs 전부 (`{flash, flash_e2, flash_fill, pro, pro_fill, retry_pro}`).
   같은 anchor에 spec 여러 개 있어도 모두 별도 호출 (학습 데이터 다양성).
2. **출력 포맷**: JSON `{"spec_id": "...", "turns": ["Q1", "A1", "Q2", "A2", ...]}`.
   짝수 인덱스 = Q, 홀수 인덱스 = A. pikogpt prep 단계에서
   `<|bos|>Q1<|turn|>A1<|turn|>...<|eos|>` 로 변환 예정.
3. **모델**: Flash 100% (속도/비용 우선). Pro 추가 합성은 결과 보고 결정.

### 검증 게이트 (Pydantic + 의미)

| 검증 | 정책 | 비고 |
|---|---|---|
| JSON parse | `response_format=json_object` | 100% 통과 |
| Pydantic schema | `turns: list[str], 4~14 length` | 통과 |
| 짝수 turn 보장 | 홀수면 마지막 미응답 Q **자동 truncate** | Q/A pair 보존 |
| anchor 등장 | verb/noun/adj lowercase substring 1회+ | retry on miss |
| turn 길이 | 1~30 단어 | retry on out-of-range |
| ~~turn 수 strict~~ | **제거** (`len == expected*2` 강제 X) | Flash가 짧게 자르는 경향 허용 |

### 결과 통계 (Flash 100%, 16 workers)

| 항목 | 값 |
|---|---|
| 처리 spec | **14,780 / 14,780 (100%)** |
| ok / fail | 14,699 / 81 |
| fail rate | **0.55%** |
| 1회 통과 | 14,536 (98.4%) |
| 2회 통과 | 132 |
| 3회 통과 | 112 |
| 누적 비용 | **$1.230** |
| Elapsed | 36.4 분 |
| Rate | 6.77 calls/s |
| 출력 크기 | `flash/raw.jsonl` 15M, 14,780 records |

### Pilot 발견

10 spec × Flash/Pro 비교에서 (`data/interim/v5_qa_pilot_stage2/results.jsonl`):

- **Pro는 spec.expected_turn_count×2를 거의 정확히 따름** (5 pair → 10 turns).
- **Flash는 종종 짧게 끊음** (5 pair인데 4 turns만 등) — turn 수 strict 검증을
  제거하고 짝수만 강제하도록 정책 조정.
- **anchor 자연스러움**: Pro가 더 풍부한 detail (e.g. "piece of paper from Grandma",
  Sikhism의 "ten gurus"). Flash는 anchor 반복 강조 경향.
- pilot 자체는 20/20 통과 ($0.00191).

### 다음 단계

1. (선택) **Pro로 일부 spec 추가 합성** — 다양성 보강. 비용 추정 $4 (14k Pro full)
   또는 부분 (예: 5k = $1.5).
2. **Fail 81건 retry** — `tools/v5_qa_stage2_retry.py` (Stage 1 retry 패턴 답습),
   예상 $0.05 / 2분.
3. **결과 분석** — anchor coverage, turn 길이 분포, vocab 분석 (5세 외 단어).
4. **pikogpt prep** — `flash/raw.jsonl` → `<|bos|>Q1<|turn|>A1<|turn|>...<|eos|>` 형식
   bin 변환 (별도 plan).
5. **v0022(val 1.84)에서 finetune** — narrative 능력 보존하며 QA 능력 강화 (별도 plan).

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
