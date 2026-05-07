# CCMC v4 — TinyStories Stories Generation Plan

작성: 2026-05-08
선결: `data/processed/ccmc_v4_tinystories/curated_tuples.jsonl` (5932 tuples) 보유
하드 캡: **총 비용 $5.0** — Pro와 Flash 각 $2.5씩
출력: `data/processed/ccmc_v4_tinystories/raw.jsonl` (model tag 포함)

---

## 1. 목표

큐레이션된 5932개의 자연 (verb, noun, adj) tuple을 입력으로 짧은 children's story를 일괄 생성. small-lm semantic-learning 처방 1순위(데이터 노출량 5~10×)의 첫 본격 라운드.

- 정통 TinyStories(Eldan & Li, 2023) 방법론
- 풀 외 단어는 CEFR A1/A2
- 5~10 문장 / story, lemma 각 ≥2회
- 모델 비교: Flash와 Pro의 산출물이 학습 효과에 차이를 만드는지 확인 (다음 라운드 모델 선정 근거)

## 2. 비용 모델 (경험치 기반)

`ccmc_tuple_curate_v4` 큐레이션 실측치:
- Pro 43 calls = ~$0.85 → **~$0.020/call** (input 2k + output ~5k)
- max_tokens=8000 환경에서 평균 출력 ~5k

Stories 호출당 토큰량 (예측):
- input: system ~150 + 프롬프트 prefix ~600 + N tuples × ~25 = ~750 + 25N
- output: N stories × ~180 = 180N
- N=10이면 input ~1000, output ~1800

OpenRouter / DeepSeek 추정 단가:
| 모델 | $/M in | $/M out | per call (N=10) |
|---|---|---|---|
| deepseek-v4-flash | ~$0.07 | ~$0.30 | **~$0.0006** |
| deepseek-v4-pro   | ~$0.27 | ~$1.10 | **~$0.0023** |

큐레이션 실측이 이론치의 ~3.5× → 안전마진 4× 적용:
- **Flash empirical 추정: $0.0024/call**
- **Pro empirical 추정: $0.0092/call**

## 3. 분배 전략 — "$2.5 each, 50/50 by budget"

`반반`을 비용 50/50으로 해석. Tuple count로 50/50 하면 Pro 쪽이 $5.5+ 들어 단독 한도($2.5)를 초과.

### 3.1 분배 결과 (예측)

| 모델 | 예산 | tuples-per-call | calls | tuples 처리 | stories 산출 |
|---|---|---|---|---|---|
| Flash | $2.5 | 10 | ~1042 | **5932 (all)** + 4488 second pass | ~10,420 |
| Pro   | $2.5 | 10 | ~272  | **2720** (random subset) | ~2,720 |
| **합계** | **$5.0** | — | ~1314 | overlap 포함 | **~13,140 stories** |

- Flash는 전체를 cover하고 남는 예산으로 일부 tuple에 second pass(설정/캐릭터 다른 second story)
- Pro는 random subset 2720개를 1 story씩 고품질 생성

### 3.2 토큰 산출 추정

- avg story = ~180 tokens
- 13,140 × 180 = **~2.37M tokens**

목표(100M+)에는 못 미치지만 stage1 v2-pro(~0.6M) 대비 **~4×**, 다음 라운드 budget 결정 근거로 충분.

## 4. 분배 절차 (재현 가능 random)

```python
seed = 42
all_tuples = read_jsonl("curated_tuples.jsonl")  # 5932
random.Random(seed).shuffle(all_tuples)

# Pass 1 — 둘 다 cover (단, Flash는 전체 / Pro는 subset)
flash_pass1 = all_tuples                # 5932
pro_subset = all_tuples[:2720]          # 처음 2720 (shuffled 후 → random subset 효과)

# Pass 2 — Flash가 남는 예산으로 다양성 강화
flash_pass2 = all_tuples[5932 - 4488:]  # 끝 4488 (다른 시드 시도해도 됨)
```

## 5. 안전장치 — 하드 캡 강제

cefr-kb의 `budget.warn_after_dollars` 만으로는 부족(경고만). 호출 사이드에서:

1. **per-model running cost tracker** — `usage` 응답의 `total_cost` 누계 (OpenRouter는 `x-ratelimit-cost` 헤더 또는 별도 endpoint)
2. **체크포인트** — 매 100 calls마다 `progress.json`에 `cost_so_far` 누적 후 `> 0.95 × cap`이면 즉시 중단
3. **two-process** — Flash와 Pro를 별 프로세스로 띄우고 각각 $2.5 cap

`config/llm.yaml`의 `budget.warn_after_dollars = 2.5` 설정을 둘로 분리하거나, 호출 워커가 직접 measure.

## 6. 명령 시퀀스 (예시)

llm-playground에서:

```bash
# 0. 사전: 출력 디렉토리 비움 (raw.jsonl은 별도 보존)
mkdir -p data/processed/ccmc_v4_tinystories/{flash,pro}

# 1. Flash 단독 — 5932 tuples
PYTHONPATH=src python -m cefr_kb.cli ccmc synth-tinystories \
  --tuples-file data/processed/ccmc_v4_tinystories/curated_tuples.jsonl \
  --task ccmc_tinystories_v4_flash \
  --tuples-per-call 10 \
  --workers 4 \
  --cost-cap-usd 2.5 \
  --out-jsonl data/processed/ccmc_v4_tinystories/flash/raw.jsonl

# 2. Pro — random subset 2720 (재현용 seed)
PYTHONPATH=src python -m cefr_kb.cli ccmc synth-tinystories \
  --tuples-file data/processed/ccmc_v4_tinystories/curated_tuples.jsonl \
  --subset-seed 42 --subset-size 2720 \
  --task ccmc_tinystories_v4 \
  --tuples-per-call 10 \
  --workers 4 \
  --cost-cap-usd 2.5 \
  --out-jsonl data/processed/ccmc_v4_tinystories/pro/raw.jsonl

# 3. 합치기 — model 태그 보존
jq -c '. + {model:"flash"}' data/processed/ccmc_v4_tinystories/flash/raw.jsonl  > raw.flash.jsonl
jq -c '. + {model:"pro"}'   data/processed/ccmc_v4_tinystories/pro/raw.jsonl    > raw.pro.jsonl
cat raw.flash.jsonl raw.pro.jsonl > data/processed/ccmc_v4_tinystories/raw.jsonl
```

> `--subset-size`, `--cost-cap-usd` 옵션은 `cli.py`에 추가 필요 (현재 미존재).

## 7. 사후 처리 — pikogpt 학습 데이터화

```bash
cd /Users/joey51/works/pikogpt
./gradlew runCcmcV4TinyStoriesPrep
```

- `CcmcV4TinyStoriesPrep`이 `raw.jsonl` 읽고 stage1 BPE meta로 `train.bin` / `val.bin` 생성 (기존 90:10 분할).
- model 태그는 학습엔 사용 안 함, 분석용으로만 보존.

## 8. 검증 — 산출물 비교

`raw.jsonl` 결합 직후, A/B 정성 비교용 샘플 100편 추출:
- model=flash 50편, model=pro 50편 → 동일 tuple 페어 가능하면 우선 매칭
- naturalness, vocab compliance, repetition, plot diversity 4축 평가
- 결과 → `docs/ccmc-v4-stories-quality.md`에 기록 (다음 라운드 모델 선정 근거)

## 9. 위험 / 트레이드오프

- **단가 변동** — OpenRouter 라우팅 모델/리전에 따라 ±30%. Pro 4× 마진 잡았지만 max_tokens 자주 hit하면 더 들 수 있음. 모니터 임계 0.9·cap에서 차단.
- **Flash 품질 폴백** — Flash 출력이 vocab compliance 낮으면 다음 라운드는 Pro 비중 ↑ (예: 70/30).
- **subset bias** — Pro subset이 random shuffle 의존이라 anchor 빈도 분포에 따라 sparse anchor 누락 가능. 필요 시 stratified split (anchor당 최소 1 story Pro) 옵션 추가.
- **CLI 옵션 부재** — `--subset-seed`, `--subset-size`, `--cost-cap-usd`는 cli.py에 신규 옵션. 진행 전 1~2시간 추가 작업.

## 10. 진행 게이트

- [ ] cli.py에 `--subset-seed/--subset-size/--cost-cap-usd` 추가 + cost tracker 구현
- [ ] Flash sanity 50 tuples 시범 → 단가 실측 + 품질 1차 점검
- [ ] Pro sanity 50 tuples 시범 → 단가 실측
- [ ] 실측이 4.1·cap 위면 plan 수정 (예: tuples-per-call ↑ 또는 subset ↓)
- [ ] 본 실행 (Flash 우선, 끝나면 Pro)
- [ ] raw.jsonl 합치기 → prep → 학습 라운드 진입
