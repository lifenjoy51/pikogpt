# 초소형 LM 의미 학습 — 사례 분석과 pikogpt 처방

작성: 2026-05-07
배경: CCMC v2-pro Stage 1/2 (1.09M params) 학습 후 의미 binding이 일관되게 정착하지 않음.
"500M 이하 모델도 의미를 잡는데 왜 안 되는가?" 질문에서 출발해, 동급/초소형 모델의 성공 사례를 조사하고 pikogpt에 적용할 처방을 정리.

---

## 0. 결론 (TL;DR)

작은 모델은 의미를 학습 가능하다. 단 *조건*이 있다 — 성공 사례 4건의 공통 공식:

> 어휘는 줄이거나 / 토큰 노출은 폭증하거나 / 둘 다.
> 그리고 embedding dim은 의미 표현의 최소 임계 (~128) 위로 둔다.

pikogpt가 못 잡는 이유는 알고리즘이나 capacity가 본질이 아니라:

- **데이터 노출량이 TinyStories 1/25, Gemma 270M 1/300,000**
- **embedding dim 96이 의미 표현 emergent 임계 (TinyStories 측정값 ~128) borderline**

→ 분포 가설 (distributional hypothesis) 이 작동할 임계 미만.

---

## 1. 일반 LLM의 의미 학습 메커니즘

LLM에 "의미" 모듈은 없다. 다음 토큰 예측 loss 하나로 학습되며, 의미는 부산물(emergent).

### 분포 가설 (Firth, 1957)

> "단어의 의미는 그 단어가 어떤 단어들과 함께 나타나는가로 정의된다."

- `cat`이 `dog, pet, fur, meow, animal, sleep`과 자주 같이 나오면, 모델 입장에서 cat의 의미는 **이 단어들과의 통계적 관계 그 자체**
- word2vec (Mikolov 2013): 16억 토큰 학습으로 `king − man + woman ≈ queen` 같은 벡터 산수 성립 → 의미가 임베딩 공간의 방향과 거리로 압축됨

### Transformer가 이걸 어디서 하는가

| 컴포넌트 | 역할 | 의미 학습 기여 |
|---|---|---|
| token embedding | id → 벡터 | 비슷한 단어는 비슷한 벡터로 수렴 (분포 가설의 핵심 저장소) |
| self-attention | 문맥 내 토큰 합성 | 동일 토큰이 맥락별 다른 의미 ("bank") |
| MLP (FFN) | feature transformation | 사실/연관 저장소 ("Paris is in __") — key-value memory 해석 |
| layer stack | 점진적 추상화 | 하위 = syntactic, 상위 = semantic |

핵심: 이 모든 게 다음 토큰 예측 loss 하나로 학습된다.

---

## 2. 초소형 LM 성공 사례 분석

### 2.1 TinyStories (Eldan & Li, 2023) — 1M ~ 33M params

가장 직접적인 비교 대상. pikogpt와 같은 sub-100M 영역에서 의미 있는 텍스트 생성 성공.

| 항목 | 값 |
|---|---|
| Params | 1M ~ 33M |
| Vocab | **3~4세 아이 단어 ~1,500개**로 강제 제한 |
| 데이터 | GPT-3.5/4 생성 합성 짧은 이야기 (~수억 tokens) |
| 다양성 확보 | 매 생성 prompt에 **무작위 verb/noun/adj 3개 조합** 강제 |
| 학습 시간 | 단일 GPU 30시간 |

**핵심 발견 (논문 §)**:

- **embedding dim 64→128 사이에서 의미 표현 emergent.**
  - dim이 사실 지식·단어 의미를 결정
  - layers 깊이는 long-range 의존성·instruction을 결정
- 1 layer 21M 모델도 일부 task 성공
- 문법은 작은 모델에서 빠르게 안정화 (그러나 의미는 dim·data가 핵심)

**핵심 트릭**: 어휘는 줄이고, *같은 어휘가 등장하는 맥락의 다양성과 양을 폭증*. 정의를 가르치는 게 아님.

### 2.2 Microsoft phi-1 / phi-1.5 — 1.3B params

"Textbooks Are All You Need" 시리즈.

| 항목 | phi-1.5 |
|---|---|
| Params | 1.3B |
| 데이터 | 합성 교과서 27B tokens (GPT-3.5 생성) + phi-1 데이터 7B |
| 결과 | 5× 큰 LLM과 비등한 reasoning |
| 인사이트 | 자연 웹 데이터 노이즈 제거 + 일관된 설명 톤 → 작은 모델 학습 효율 극대화 |

### 2.3 SmolLM-135M (HuggingFace, 2024)

| 항목 | SmolLM-135M |
|---|---|
| Params | 135M |
| 데이터 | Cosmopedia v2 (28B 합성) + FineWeb-Edu (220B 필터링) + Stack-Edu (4B) = **600B tokens** |
| Vocab | 49,152 |
| 아키텍처 | depth-priority, width-minimal (MobileLLM 기반, Grouped-Query Attention) |
| Context | 2048 |
| 학습 발견 | Chinchilla optimal (3T) **초과** 학습이 더 효율적 → 600B 선택 |

**핵심**: 합성 데이터 + 필터링된 자연 웹 + 충분한 학습 token. depth 우선·width 최소가 efficient.

### 2.4 Google Gemma 3 270M (2025)

| 항목 | Gemma 3 270M |
|---|---|
| **Params 분포** | 총 270M 중 **170M이 embedding** (vocab 256K), transformer block은 100M만 |
| 학습 토큰 | **6 trillion tokens** (270M인데도 거대 데이터) |
| 설계 의도 | task-specific fine-tune base — pretrain은 의미·언어 구조용 거대 base |
| 아키텍처 | 12 layers, hidden 1024, sliding window + RoPE + RMSNorm, QAT |

**핵심**: 모델은 작아도 **본 텍스트의 양은 거대**. 작은 capacity를 데이터 양·vocab 크기로 보완.

### 2.5 BabyLM Challenge (NeurIPS/EMNLP 2023~2026)

- 학습 데이터 **100M tokens** 또는 **10M tokens**로 강제 제한 (어린이 13세까지 받는 언어 input과 동등)
- 결론:
  - 이 budget으로도 의미·문법 학습 가능
  - **tokenization·sequence length·optimizer 같은 작은 design choice가 architecture 변경보다 큰 영향**
  - 어린이는 ~100M words로 언어 학습 — 인간 baseline 수준의 sample efficiency가 ANN으로도 가능함을 보임

---

## 3. 공통 패턴 — 의미 학습 emergent의 *최소 조건*

| 조건 | TinyStories | phi-1.5 | SmolLM-135M | Gemma 270M | **pikogpt 현재** |
|---|---|---|---|---|---|
| Params | 1M ~ 33M | 1.3B | 135M | 270M | **1M** |
| Vocab | **1,500** (작게) | ~50K | 49K | **256K** (크게) | 2,000 |
| 학습 tokens | ~500M | 27B | 600B | **6T** | **20M** |
| 합성 vs 자연 | 100% 합성 | 100% 합성 | 합성+필터링 웹 | 자연 웹 | 합성 (CCMC) |
| 어휘 다양성 | 1,500 단어 | 자연어 전체 | 자연어 전체 | 자연어 전체 | **1,825 lemma** |
| Embedding dim | 128 ~ 256 | 2048 | 576 | 640 | **96** |
| 의미 학습 success | yes | yes | yes | yes | **no** |

### pikogpt vs 성공 사례 — 결정적 차이 두 가지

**A. 데이터 토큰 수가 25 ~ 300,000배 부족**

- TinyStories조차 ~500M tokens (pikogpt 25배)
- Gemma 270M은 6T (300,000배)
- 분포 가설은 같은 단어를 수만~수백만 맥락에서 봐야 작동
- pikogpt는 lemma별 평균 맥락 노출이 수십~수백 → 임계 미만

**B. embedding dim 96은 의미 표현 임계 borderline**

- TinyStories 측정: dim 64→128에서 의미 emergent
- pikogpt 96은 임계 위에 살짝 걸쳐 있지만 1825 lemma를 distinct 분리하기엔 부족
- TinyStories: 1500 어휘 / dim 128~256 → 어휘:dim 비율 6~12
- pikogpt: 1825 lemma / dim 96 → 비율 19 (4배 빡빡)

---

## 4. pikogpt 처방 (우선순위순)

기존 plan v4의 "정의 augmentation" 옵션은 TinyStories/phi/SmolLM 어디에서도 핵심이 아니었다. 그들의 공통 트릭은 "정의를 가르치기"가 아니라 **"같은 단어가 등장하는 다양한 합성 맥락의 양을 폭증"**.

### 1순위: 데이터 양 폭증 (TinyStories 방식)

- 현재 ~20M → **최소 200M ~ 500M tokens**로 확장
- LM Studio (gemma-3-1b 또는 더 큰 모델)로 CCMC 1825 lemma를 사용한 짧은 이야기/대화/사실 설명 대량 생성
- 핵심: 매 prompt에 **무작위 lemma 3개 조합 강제** → 같은 lemma가 수천 다양한 맥락에서 등장 (TinyStories 정확한 방식)
- 정의 augmentation 8개/lemma 추가는 효과 작음. **다양한 맥락 1000개/lemma**가 본질

### 2순위: embedding dim 96 → 128 또는 192

- TinyStories emergent 임계 위로 확실히 올림
- 모델 capacity 1M → 2~3M (plan v4 옵션 N과 결합)
- 어휘:dim 비율 6~12 영역으로 진입

### 3순위: vocab 줄이기 (선택적)

- TinyStories식 어휘 제한: 2000 → 1500 또는 lemma 수 1825 → 700
- 같은 token budget으로 lemma별 노출 ~3배 증가
- 도메인 요구가 1825 lemma 전체라면 skip

### 4순위: 기존 plan v4 옵션들 (보조)

- 옵션 L (정규화 약화): 위 1~3 fix 후 보조 효과
- 옵션 K (blockSize 64): 1순위 진행 시 contrast 학습 회복용
- 옵션 M (iter 2×): 데이터 양 확장과 자연스럽게 결합

---

## 5. 1차 검증 실험 제안

가장 ROI 높은 ablation:

```
Step 1. 합성 데이터 5~10× 확장
  - LM Studio로 CCMC lemma 기반 단편 이야기 100~200M tokens 생성
  - TinyStories 식 무작위 3-lemma 조합 prompt
  - 데이터셋: data/ccmc-v3/stage1/

Step 2. dim 128 + 모델 capacity 약 2~3×
  - embeddingDimension 96 → 128
  - numberOfLayers 8 유지 (또는 깊이 우선 10~12)
  - 결과 params ~2~3M

Step 3. chunk-anchored sampling 그대로 + maxIters 적정 조정
  - 100M tokens / blockSize 32 / batch 64 → epoch 약 3~5회 분량
```

예상 학습 시간: ~6h (현재 1.7s/iter × 12000 iter 추정)

**판정 기준**:

- 정의 binding 정착 ("A cat is an animal" 형태의 정확한 응답) → 데이터 부족이 main culprit 확정
- 여전히 약함 → 5순위로 옵션 N (모델 5~10× capacity, ~10M params) 진행

---

## 6. 핵심 참고 자료

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English? (Eldan & Li, 2023)](https://arxiv.org/abs/2305.07759)
- [Textbooks Are All You Need II: phi-1.5 technical report (Microsoft Research, 2023)](https://arxiv.org/pdf/2309.05463)
- [SmolLM - blazingly fast and remarkably powerful (HuggingFace blog, 2024)](https://huggingface.co/blog/smollm)
- [Cosmopedia: how to create large-scale synthetic data for pre-training (HuggingFace blog)](https://huggingface.co/blog/cosmopedia)
- [Introducing Gemma 3 270M (Google Developers Blog, 2025)](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [google/gemma-3-270m (model card)](https://huggingface.co/google/gemma-3-270m)
- [BabyLM Challenge — sample-efficient pretraining](https://babylm.github.io/)
- [Findings of the Second BabyLM Challenge (2024)](https://arxiv.org/html/2412.05149v1)
- [Small Language Models (SLMs) Can Still Pack a Punch: A survey (2025)](https://arxiv.org/html/2501.05465v1)
- [Demystifying Synthetic Data in LLM Pre-training: A Systematic Study (2025)](https://arxiv.org/html/2510.01631v1)

---

## 7. 한 줄 요약

> pikogpt가 의미 학습 못 하는 이유 — 알고리즘이나 capacity가 본질이 아니라,
> **TinyStories 학습량의 1/25, Gemma 270M의 1/300,000인 데이터 노출량**이 분포 가설이 작동할 임계 미만이기 때문.

---

## 8. 진행 — v4 TinyStories 재생성 (2026-05-08 기준)

5순위 처방 중 1순위(데이터 5~10×)를 cefr-kb 인프라(deepseek-v4-pro / OpenRouter)로 실행. TinyStories(Eldan & Li, 2023) 정통 방법론을 CCMC 1825 lemma 풀 안에서 재현.

### 8.1 파이프라인 설계

```
[anchor curation]                       [stories generation]
  noun anchor 1094개  ──→  자연 (verb,noun,adj) tuple  ──→  short children's story
  + verb pool 161 (in-prompt)            5932 tuples              N편/batch
  + adj  pool 123 (in-prompt)            (avg 6.16/anchor)
```

**큐레이션 (anchor-based + pool injection, "v2 mode")**:
- noun을 anchor로, verb/adj는 풀을 통째로 프롬프트에 박아 LLM이 자연스러운 조합만 골라 N tuples/anchor 생성.
- 무작위 21M 조합(verb 161 × noun 1094 × adj 123) 중 strain 없는 자연 조합만 통과 → 결과적으로 약 5932 tuples가 stories generator의 input이 됨.
- 핵심: 풀 외 단어 사용 금지 + 풀 내에서도 anchor와 자연스럽지 않으면 drop. forced 조합("drink/linguistics/interesting")을 사전 차단.

### 8.2 큐레이션 결과

| 항목 | 값 |
|---|---|
| anchor (noun) 처리 | 1094 → 963 yielded ≥1 tuple |
| 총 tuples | **5932** (avg 6.16/anchor, target 15의 41%) |
| LLM 호출 수 | 43 (anchors-per-call 26) |
| 소요 시간 | 1497s (~25min) |
| 사용된 verb / adj | 132 / 111 (풀 161 / 123 중) |
| 빈도 상위 verb | see(297) · remember(278) · learn(276) |
| 빈도 상위 adj | interesting(347) · new(334) · important(308) |

산출물: `data/processed/ccmc_v4_tinystories/curated_tuples.jsonl` (5932 lines, 553KB).

샘플:
```
(learn, swahili, new) (help, child, small) (love, argentina, wonderful)
(enjoy, folklore, funny) (cook, fowl, delicious)
```

### 8.3 다음 단계

1. `synth-tinystories --tuples-file curated_tuples.jsonl` 실행 — tuple당 1편씩 짧은 children's story 생성. 목표 토큰량 100M+.
2. pikogpt 측 `runCcmcV4TinyStoriesPrep` (`src/main/kotlin/data/CcmcV4TinyStoriesPrep.kt`)으로 stage1 BPE meta로 인코딩 → `data/ccmc-v4-tinystories/{train,val}.bin` 생성.
3. 5순위 처방 후속 — embedding dim 96 → 128, model capacity ~2~3M로 분포 가설 임계 위 학습.
> **합성 데이터 10× 확장 + dim 1.5× 확장**이 검증된 공식.
