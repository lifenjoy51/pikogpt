# CCMC v2-pro 의미 학습 실패 — Root Cause Analysis (v2)

## Context

이전 (5/5 데이터, val 2.19 best) / 현재 (5/6 데이터, val 2.37 best) 두 차례의 ~1M params Stage1 학습 모두 **lemma binding (의미 학습)이 약하게만 발현**됐다. 사용자가 "왜 의미학습이 안되는지" 깊이 검토 요청.

**v2 갱신 (sampling 실험 후)**: 첫 분석 후 best v0050 ckpt에 leading-space prompt (` cat is`, ` cat`)로 재샘플링한 결과 — cat/water/run는 활성화, tree/eat/happy/big은 활성화 약함. 이 편차의 정량 원인을 추적해 **진단을 수정**한다: "binding 학습 안 됨"이 부정확하고, **"정의 binding은 약하고 일화/analogy binding은 강함"이 정확**.

이 문서는 **원인 분석 결과**와 **다음 실험 옵션**을 정리한다. 본 plan은 즉시 실행할 코드 변경 plan이 아니라 — 사용자 결정 후 옵션 중 하나를 채택해 별도 실행 plan을 작성한다.

---

## 근본 원인 (중요도 순)

### 원인 1 — Vocab/평가 prompt mismatch (immediate, fix 가능)

평가에 사용한 10개 prompt의 단어가 모두 **vocab에 single token으로 존재하지 않는다**:

```
Cat   → vocab에 없음, ' cat' (id=380)만 존재
Tree  → vocab에 없음, ' tree' (id=899)만 존재
Run   → vocab에 없음, ' run' (id=521)만 존재
Eat   → vocab에 없음, ' eat' (id=394)만 존재
Happy → vocab에 없음, ' happy' (id=603)만 존재
Big   → vocab에 없음, ' big' (id=183)만 존재
Above → vocab에 없음, ' above'도 없음 (모두 sub-token 분해)
Quickly → vocab에 없음, ' quickly'도 없음
And   → vocab에 없음, ' and' (id=110), 'and' (id=315)
```

원인: `useWordPreTokenize=true`인 BPE는 단어 앞 공백을 붙여 토큰화. 학습 데이터 대부분은 `the cat`, `a big tree` 형태 → ` cat` (공백+cat)이 single token. 그러나 평가 prompt는 `<|bos|><|sep|>Cat is` (대문자, 공백 없음, 문장 시작) → `Cat`은 vocab 미존재 → C, a, t 각 char로 분해.

**효과**: 모델이 학습한 lemma binding을 평가에서 활성화하지 못함. "Cat is..." 입력 시 모델은 ` cat`이 아닌 character-level fragments를 보므로 lemma 토큰 임베딩이 활성화 안 됨.

검증된 정량 fact:
- "Cat is" exact match: stage1 train에 **1번**
- "cat is" exact match: **138번** ← 학습 분포의 실체
- 모델은 거의 모든 학습에서 ` cat`을 봤지 `Cat`을 본 적이 없음

### 원인 2 — 평가 prompt 형식이 학습 데이터에 거의 없음

```
"Run means"     0번
"Eat means"     0번 (대문자), 1번 (소문자)
"Happy means"   0번
"Big means"     0번
"Above means"   0번
"Quickly means" 0번
"And means"     0번
```

`X means Y` 형식은 **학습 데이터의 lemma 정의 패턴이 아님**. 데이터의 정의 패턴은 `X is Y` (예: "A cat is an animal", "above is the opposite of below"). "means"라는 단어 자체는 데이터에 등장하지만, "[Lemma] means [Definition]"의 record-start 패턴은 거의 zero.

**효과**: 평가 prompt가 in-distribution이 아니므로 **out-of-distribution generation** — 모델이 본 적 없는 형식을 받아 학습된 분포에서 가장 그럴듯한 후속어만 생성.

### 원인 3 — 평가 lemma 자체가 데이터에 희소

```
        prev (val 2.19)  curr (val 2.37)
Cat       1                1
Tree      1                1
And       0                1
Quickly   4                0    ← 현재 데이터에 0번!
Run       6                8
Eat       4                4
Above    96                92
```

대문자 형태 (sentence-start position) 기준. 예: "Quickly" 대문자가 현재 stage1 train에 단 한 번도 없음. 이는 외부 데이터 생성 단계에서 sentence-start 변환이 변경된 결과.

**효과**: 모델이 평가 lemma를 sentence-start position에서 본 적이 거의 없으므로, 평가 시 그 position의 적절한 후속 분포를 학습하지 못함.

### 원인 4 — 데이터 record 구조 변경 (5/5 → 5/6)

이전 (val 2.19) vs 현재 (val 2.37) 데이터의 결정적 차이:

| 항목 | 이전 (5/5) | 현재 (5/6) |
|---|---|---|
| record 수 (stage1 train) | 1,824 | 3,313 (split 후) |
| 평균 chars/record | 1,143 | 1,685 (+47%) |
| record 응집도 | lemma centered short | mixed long |
| record 1 예시 | `Abraham has...<\|sep\|>This is Abraham's sheep.<\|sep\|>The story is about Abraham.<\|sep\|>Abraham is my dad.<\|sep\|>` (모든 sentence가 lemma 직접 언급) | `In my bedroom at night, I open...Cool air comes inside.<\|sep\|>Air can be cold...<\|sep\|>` (일화 묘사 + lemma 묘사 혼합) |

이전: record = "단일 lemma + 짧은 정의/예시 응집". 한 record를 보는 동안 모델은 그 lemma만 반복 노출 → 강한 binding 신호.

현재: record = "단일 lemma + 일화 묘사 + 다양한 sentence 형식". 같은 lemma가 record 안에 있더라도 "I open the window..." 같은 일화로 시작 → lemma 정의 신호 희석.

**효과**: 같은 epoch 수에서도 lemma binding 신호가 더 약하게 흘러간다.

### 원인 5 — 모델 capacity (~1M)의 본질적 한계

이전 학습조차 binding이 명확히 정착되지는 않았다. `docs/ccmc-v2-pro-stage1.md`의 가장 좋은 사례:
- "Happy means → **Happy is good, but now is a sad. Running helps you feel happy.**" (good↔sad contrast 발현, Running 연관) — **약한 의미적 신호**
- "Cat is → a thing you can feel and bumpy. Cats look at the big building" — **Cats 활성화는 됐지만 정의는 X**

1M params로 1,825 lemma의 정확한 의미 binding을 학습하는 건 본질적으로 어렵다. TinyHelen vec (1.05M, 4L) val 4.88, TwoStageBase (768K, 6L) val 2.97 — 이전 분야 1M-class 실험 모두 binding은 약했다. 본 setup의 val 2.19/2.37은 1M-class에서 **상위권** 수치이지만 binding 자체가 개념적으로 안정될 capacity는 아니다.

### 원인 6 — 학습 setup의 보조 요인

- **정규화 과다**: `dropout=0.05, weight_decay=0.05, label_smoothing=0.05` 동시. 비교: TinyHelenTrainVec (1M)은 dropout=0.0, weight_decay=0.02, label_smoothing 없음 → 본 setup이 훨씬 강한 정규화. lemma binding은 본질적으로 *memorization*인데 정규화가 logit 극단을 평탄화 → binding 약화.
- **blockSize 64 vs 평균 294 tokens/record**: 한 batch slice가 record 1/5만 본다. record 시작의 정의문이 context 창 밖이면 정의-후속 sentence 학습 못함. RecordAwareDataLoader는 record 경계는 보존하지만 random offset uniform → 정의문 위치 가중 X.

이 둘은 1, 2, 3, 4의 영향에 비해 부차적 — fix해도 1~3을 안 고치면 의미 학습은 여전히 안 됨.

---

### 원인 7 — 학습 batch에서 binding 신호 자체가 매우 약함 (v2 신규, 정량 추정)

RecordAwareDataLoader (`src/main/kotlin/train/DataLoader.kt`) 동작:
- random record 균등 + record 내 random offset uniform [0, recordLen-65]
- batch: 2 × 32 (gradAccum) = 64 seq/iter, 10000 iter → 640,000 batch sequences

정량 추정:
- record당 starting record로 선택되는 횟수: 640,000 / 3,313 ≈ **193회** per training run
- record당 가능한 starting offset: 평균 record 294 tokens − 65 ≈ **229개**
- **한 record의 한 offset이 학습 batch에 들어오는 평균 빈도: 193 / 229 ≈ 0.84회 per run**

즉, 한 record의 한 위치가 평균 1회도 학습 안 됨. 정의 sentence (~5 tokens)가 record 시작 (offset 0~4)에 있더라도 학습 batch에 들어오는 횟수: ≈ **20회** (offset 0~4가 모두 cover되는 batch slice 수). strong binding 학습은 일반적으로 **50+ 회 노출** 필요. 따라서 본 setup은 **학습 신호 부족**.

추가: blockSize 64에서 position 0~15는 context < 32 tokens라 신호 약함. 실효 학습 위치는 position 30~63 (batch slice의 마지막 절반). 즉 정의 5-tok span이 강한 신호로 학습되려면 batch slice의 후반부에 들어와야 함 → 추가 1/2 가중치 손실 → 실효 학습 ≈ 10회.

### 원인 8 — Lemma별 next-token 분포의 entropy 차이가 활성화 편차의 직접 원인 (v2 신규, sampling 결과 설명)

`grep -oE " <lemma> is [a-zA-Z]+" stage1/train.txt`로 다음 word top-3 분석:

| Lemma | "X is" 등장 | Top-3 next-word | Entropy | 분포 특성 | sampling 활성화 |
|---|---|---|---|---|---|
| cat | 132 | sleeping(11), on(11), a(10) | **5.09** | 평탄 | ✅ |
| water | 156 | to(18), cold(13), not(11) | **5.06** | 평탄 | ✅ |
| tree | 45 | tall(9), to(8), small(3) | 3.97 | 집중 | ❌ |
| run | 8 | to(4), for(1), fast(1) | 2.00 | 매우 집중 | △ run 단독에서만 |
| eat | 10 | to(8), food(1), different(1) | 0.92 | 극집중 | ❌ |
| happy | 6 | to(5), a(1) | **0.65** | 극집중 | ❌ |
| big | 13 | to(11), small(1), not(1) | 0.77 | 극집중 | ❌ |

해석:
- **cat/water (entropy 5.0+)**: "X is" 직후 분포가 평탄 → 다양한 일화/속성/관계가 균등하게 학습 → prompt에서 모델이 cat/water를 보면 "다양한 후속" 중 하나를 자신감 있게 선택 → 응답에 lemma 자기 자신 / 변형 등장
- **tree (entropy 3.97)**: 분포가 "tall"에 20% 집중 → "tree is" prompt에 모델은 "tall"을 가장 강하게 예측해야 하지만, 다른 활성화 경로 (여러 lemma의 일화)가 더 확률 높을 수 있음 → 결과적으로 tree token 활성화 약함, kitten 등 다른 lemma로 흘러감
- **run/eat/happy/big (entropy < 2)**: "X is" 다음 거의 항상 "to" → 모델은 정의 학습 신호 거의 없고 analogy 패턴 ("X is to Y as A is to B")만 학습 → 평가에서 "X is" prompt 받으면 "to" 출력 후 analogy로 흘러감, lemma 자체 binding은 약함

또한 record 시작 비율 (`<|sep|>` 직후 sentence가 "X is"인 비율):
- cat 9%, water 6%, tree 4%, big 30%
- 즉 tree는 record 시작에 거의 없음 → 강한 학습 위치를 못 차지

### 원인 9 — 모델이 학습한 binding의 본질이 "정의"가 아닌 "일화/analogy" (v2 신규, 진단 갱신)

`grep "lemma is"` 138 / 167 / 49 sentence를 정의/속성/일화/관계로 분류:

| Lemma | 정의 (a/an/to/the) | 속성 (adj) | 일화 (loc/verbing) | 관계 (for/like) |
|---|---|---|---|---|
| cat | **49%** | 7% | 24% | 9% |
| water | **35%** | 28% | 6% | 17% |
| tree | 33% | 산재 (tall 등) | 54% | — |
| run/eat/happy/big | **70~100% "to"** (analogy) | 0% | 0% | — |

해석:
- cat은 정의 비율 비교적 높음 (49%) → 이전 학습 응답 "Cat is a thing you can feel and bumpy"가 정의-like인 이유
- water는 정의+속성 합쳐 63% → 이번 응답 "I drink the cool water"가 cool=속성으로 활성화
- tree는 일화 비율 압도적 (54%) → 응답이 일화 묘사로 흘러가지만 tree token 자체는 안 나옴 (tree binding이 일화 token에 분산)
- run/eat/happy/big은 거의 100% "X is to Y as A is to B" analogy 패턴 → 응답이 analogy 형식 ("Tree is to think as a book is to read")으로 자주 나옴

**즉 모델은 데이터 분포를 정확히 학습**. 우리가 평가에서 기대한 "X is [정의]" 형식이 데이터에 없는 lemma에 대해선 binding이 일화/analogy로만 학습됨. 이전/현재 학습 둘 다 같은 본질 — capacity 한계 + 데이터 분포 + 평가 형식 불일치의 3중 문제.

---

## 핵심 진단 (v2 갱신)

진단을 수정한다. *"왜 의미학습이 안 됐나?"* → **"정의 binding은 약하고, 일화/analogy binding은 학습 데이터 분포대로 정확히 학습됐다."**

분해:

1. **모델은 lemma token을 인식하고 활성화함** (sampling 검증). ` cat is` → "cat" 등장, ` water is` → "water" 등장, ` run` → "running/runs" 등장
2. **lemma별 binding의 *형태*가 데이터 분포대로 학습됨**:
   - cat (정의 49%, entropy 5) → 정의-like 응답 일부 가능
   - water (정의 35% + 속성 28%, entropy 5) → 속성 응답 ("cool")
   - tree (일화 54%, entropy 3.97) → 일화로 흘러감, lemma 활성화 약함
   - run/eat/happy/big (analogy 70~100%) → "X is to Y" 응답
3. **본 setup의 본질적 문제 (이번/이전 모두)**:
   - **학습 신호 부족** (원인 7): 한 record의 한 offset이 평균 0.84회만 학습 → strong binding 못 만듦
   - **데이터 분포가 정의 부족** (원인 9): 데이터에 정의 sentence 자체가 적음. 모델은 *없는 신호를 학습 못 함*
   - **vocab/prompt mismatch** (원인 1~3): 평가가 학습 분포를 못 활성화
   - **모델 capacity 1M** (원인 5): 위 모든 약한 신호를 메우기엔 부족

**즉 fix해야 할 우선순위 (v2 갱신)**:
1. **데이터 정의 sentence 비율 ↑** (원인 9 fix) — 외부 데이터 파이프라인 수정 또는 정의-augmentation
2. **학습 신호 강화** (원인 7 fix) — record 시작 가중 sampling 또는 blockSize ↑
3. **평가 prompt를 데이터 분포에 맞춤** (원인 1~3 fix, 즉시) — 이미 sampling 실험으로 부분 검증됨
4. **모델 capacity ↑** (원인 5 fix) — 위 fix 후 ceiling 도달 시 적용

---

## 다음 실험 옵션 (사용자 결정 사항)

각 옵션은 가설 검증 + 비용 함께 표기. 사용자가 선택하면 별도 plan으로 분리해 실행.

### 옵션 A — 평가 prompt를 학습 분포에 맞게 (가장 싼 검증)
**비용**: 코드/학습 0, 샘플링만 다시
**검증 목표**: vocab/prompt mismatch가 binding "보이지 않음"의 주범인지

- prompt를 sentence-mid position 분포에 맞춰 변경:
  - `<|bos|><|sep|>The cat is` (lowercase + space + 일반 article)
  - `<|bos|><|sep|>A tree is`
  - `<|bos|><|sep|>I run when`
  - `<|bos|><|sep|>To eat is to`
  - 등 학습 데이터에서 빈도가 ≥10인 패턴
- 기존 best ckpt v0050에 새 prompt로 `runSamplePromptsFromFile` 실행
- 만약 binding 신호가 갑자기 좋아지면 → 원인 1+2가 main culprit. 모델은 binding을 학습했으나 평가가 잘못된 것.

### 옵션 B — Vocab 키우고 cased single token 강화 (중간 비용)
**비용**: BPE 재학습 + Stage1 재학습 ~10시간
**검증 목표**: vocab 2000 부족이 lemma 활성화의 본질 장벽인지

- vocab `2000 → 4000` 또는 `8000`. cased + useWordPreTokenize 유지
- 효과: `Cat`, `Tree`, `Run`, `Eat`, `Happy`, `Big`, `Above`, `Quickly`, `And`가 모두 single token화될 가능성 ↑
- 단, vocab 늘면 1M params에서 token embedding 차지 비중 ↑ (vocab 4000 × 96 = 384K, 모델의 ~1/3) — capacity 압박

### 옵션 C — 데이터 형식을 이전 응집도로 복원 (외부 의존)
**비용**: 외부 코드 수정 + 데이터 재생성 + 재학습
**검증 목표**: record 응집도 ↓가 binding 약화의 main 원인인지

- llm-playground 데이터 생성 코드에서:
  - record당 sentence 수 ↓ (현재 길어진 일화 묘사 제거)
  - lemma centered 응집 (모든 sentence가 lemma 직접 mention) 강제
- 예전 record 1 형식 복원: "Lemma A. Lemma B is C. Lemma D ..." 같은 short binding 묶음
- 본 repo에서 직접 처리는 어려움 — 외부 데이터 파이프라인 변경 필요

### 옵션 D — 모델 capacity 증가 (binding 본질 해결, 큰 비용)
**비용**: 모델 크기 2~5×, 학습 시간 2~5×, 메모리 ↑
**검증 목표**: 1M의 본질 한계가 binding 정착의 ceiling인지

- 8L × 96D × 3H × ~1M → 8L × 192D × 6H × ~3M 또는 12L × 256D × 8H × ~8M
- vocab 같이 키워도 OK (옵션 B와 결합)
- 위험: pikogpt 본 repo는 1M-class 검증된 setup. 큰 모델은 학습 시간/메모리 미검증

### 옵션 E — 정규화 약화 (가장 빠른 ablation)
**비용**: Stage1 재학습 ~10시간
**검증 목표**: 강한 정규화가 binding을 막는지

- `dropout=0.0, weight_decay=0.02, labelSmoothing=0.0` (TinyHelen 1M-class 설정)
- 다른 설정 동일하게 두고 binding 변화 비교
- 작은 ablation — 결과로 정규화의 영향만 격리해 확인

### 옵션 F — 옵션 A + E 병행 (실용적)
**비용**: 옵션 A는 즉시, 옵션 E는 ~10시간
**전략**:
1. 먼저 옵션 A로 v0050 ckpt에 in-distribution prompt 샘플링 — 하루 안에 결과
2. 결과에 따라:
   - binding이 잘 보이면 → 원인 1+2 확정. 모델은 OK, 평가만 고치면 됨
   - 여전히 약하면 → 원인 4+5 영향 큼. 옵션 E로 정규화 ablation 시도
   - 그래도 약하면 → 옵션 B 또는 C (데이터/vocab 근본 변경)

### 옵션 G — 이전 best v0072 (val 2.19)에 같은 leading-space prompt (v2 신규, 즉시)
**비용**: 0 (sampling만 다시)
**검증 목표**: 데이터 형식 변경 (응집도 ↓)이 실제로 binding 약화의 원인인지 격리 비교

이전 ckpt 경로: `model/stage1/vec/1087936/_archive/run_20260506_v01/v0072/`
같은 prompt (` cat is`, ` water is`, ...)로 sampling → 이전이 더 정의-like 응답이면 데이터 형식이 main culprit. 비슷하면 본질적 1M 한계 (원인 5 + 9).

### 옵션 H — Lemma 분포 기반 평가 prompt 재선정 (v2 신규, 즉시)
**비용**: 0 (sampling만)
**검증 목표**: 데이터에 정의 비율 높은 lemma만 평가 → 모델이 어디까지 binding 학습했는지 ceiling 측정

학습 데이터에서 "X is" 다음 분포 entropy 5.0+ 인 lemma + 정의 비율 ≥40% 인 lemma를 직접 추출 (cat 외 ~10~20개). 이 lemma로 ` X is` prompt 평가 → 활성화 비율 측정. 이게 ≥80%면 모델은 binding 학습 잘 한 것, 평가 형식만 잘못됐던 것.

### 옵션 I — Stride-chunk anchored sampling (v3 갱신, 코드 변경)
**비용**: RecordAwareDataLoader 확장 + Stage1 재학습 ~10시간
**검증 목표**: 원인 7 (학습 신호 부족, 한 offset 평균 0.84회 학습) fix

직전 v2의 "record 시작 가중"은 약함 — 정의 sentence가 record 전체에 균등 분산 (0-20% 24%, 20-40% 18%, ..., 80-100% 22%) 확인됨. 따라서 모든 위치를 빈틈없이 cover해야 함.

**v3 설계 (사용자 제안 통합)**:
```
on load:
  for each record:
    chunk_starts = [0, blockSize, 2*blockSize, ..., recordLen-blockSize]
                    # stride = blockSize (no overlap), 마지막 anchor는 record 끝 보장

on getBatch:
  record = random record
  base = random choice from chunk_starts        # chunk anchor random
  jitter = random offset in [0, blockSize)      # 매 batch 시작점 약간 흔듦
  pos = clamp(base + jitter, 0, recordLen - blockSize)
  return tokens[pos : pos + blockSize+1]
```

데이터 정량 검증:
- record 평균 294 tokens, blockSize 64 → record당 5 chunk anchor, 총 ~16,565 anchor
- 한 anchor 평균 학습: 640,000 / 16,565 = **38.6회** (현재 0.84회 대비 **46×** 강화)
- jitter로 anchor 사이 위치도 cover → 빈틈 없음 + 일반화

### 옵션 J — blockSize 줄이기 (v3 신규, hyperparam 변경)
**비용**: hyperparam 1줄 수정 + Stage1 재학습 (32: ~5h, 48: ~7h vs 현재 64: 10h)
**검증 목표**: blockSize가 binding 학습의 sweet spot인지

데이터 정량 (직전 검증):
| blockSize | 단일 sentence cover | 2-sentence pair (contrast) cover | 학습 시간 |
|---|---|---|---|
| 32 | 89.2% | **64.7%** ← contrast 35% 잘림 | ~50% |
| 48 | 91.1% | 79.3% | ~70% |
| 64 | 95.7% | 85.1% | 100% (현재) |

**Trade-off**: 데이터에 "X is to Y as A is to B" / "X is A, but Y is B" 같은 multi-sentence contrast/analogy 패턴이 많음 (2-sentence 묶음 평균 35 tokens). blockSize 32는 단일 sentence 학습은 충분하나 contrast 학습 35% 잘림 → binding 표현력 약화.

**선택지**:
- 32: 가장 빠름, 단일 sentence 학습 안전, contrast 약함
- 48: sweet spot 후보 — 단일 91% + pair 79%, 시간 70%
- 64 유지: contrast 안전, 시간 100%

옵션 I와 결합 가능 — blockSize와 stride를 같은 값으로 두면 chunk 수가 늘거나 줄어 한 anchor 학습 빈도 변화. 예: blockSize 48 + stride 48 → record당 chunk ~6개, 한 anchor 학습 27회.

---

## 권장 순서 (v3 갱신)

**완료**:
1. ✅ **옵션 G** — 이전 v0072 비교 sampling: 이전도 binding 약함, 데이터 응집도 변화는 main culprit 아님
2. ✅ **옵션 H** — 정의 비율 높은 lemma 재평가: dog/cat/book만 활성화 (빈도 ≥ 900). 1M ceiling 보임
3. ✅ **옵션 I + J 결합 실행 (v0053)** — chunk-anchored sampling + blockSize 32 + worker 10 + parallel sampler
   - 학습 시간 9h25m → **1h47m** (5.3× 단축)
   - 한 anchor 학습 빈도 0.84회 → **20.2회** (24× 강화)
   - 활성화 lemma 수 4/10 → **9~10/10** ✅ 큰 진전
   - 카테고리 binding 첫 등장: "the water and the sun are all liquids" 🌟
   - val avg 2.37 → 2.68 (약간 악화 — blockSize 32 contrast 35% 잘림 trade-off)

**다음 단계 (정의 binding 강화)**:
이번 결과로 *활성화*는 해결, 남은 과제는 *정의 binding* (cat is an animal 같은 명시적 정의). 다음 옵션 검토 — 새로 추가된 v4 옵션들.

---

## 실행 결정 (사용자 선택)

옵션 **I + J 결합** 채택. blockSize **32** 선택.

### 변경 사항

**1. 신규 sampling mode — `ChunkAnchoredDataLoader`**

`src/main/kotlin/train/DataLoader.kt`에 신규 클래스 추가 (기존 `RecordAwareDataLoader` 패턴 따름):
```
class ChunkAnchoredDataLoader(dataPath, batchSize, blockSize, bosId, jitter):
  on load:
    tokenData = read train.bin
    recordStarts = scan bosId positions
    chunkAnchors = for each record:
      [base, base+blockSize, base+2*blockSize, ..., recordEnd-blockSize]
      # stride = blockSize, 마지막 anchor는 record 끝 보장
    flatten chunkAnchors into single list

  getBatch():
    for each batch slot:
      anchor = random pick from chunkAnchors
      offset = anchor + Random.nextInt(0, blockSize)   # jitter
      offset = clamp(offset, recordStart, recordEnd - blockSize)
      x = tokenData[offset : offset + blockSize]
      y = tokenData[offset+1 : offset + blockSize+1]
```

**2. `TrainConfig.kt` — 신규 field**

```
val chunkAnchoredSampling: Boolean = false
```
`recordAwareSampling`과 mutually exclusive. 둘 다 true면 `chunkAnchoredSampling` 우선.

**3. `VecTrainer.kt` — 분기 추가**

기존 `recordAwareSampling` 분기 옆에 `chunkAnchoredSampling` 분기 추가. train + val 양쪽 동일 적용.

**4. `CcmcV2ProStage1TrainVec.kt` — config 변경**

```
blockSize = 32                     # 64 → 32 (binding 학습 신호 강화 검증)
chunkAnchoredSampling = true       # 신규
recordAwareSampling = false        # 비활성
maxIters = 10000                   # 동일 (학습 시간 ~5h 예상)
```
나머지 hyperparams 동일.

### 정량 기대 효과

- record당 chunk anchor: ceil((294-32)/32) ≈ **9개** (이전 stride 64에서 5개보다 더 세밀)
- 총 anchor: 3313 × 9 ≈ **29,800**
- 한 anchor 평균 학습 (640K batches): 640,000 / 29,800 = **21.5회/run** (현재 0.84회 대비 **26×**)
- jitter (0~31)로 anchor 사이 위치도 cover → 빈틈 없음
- 학습 시간: blockSize 32라 attention O(n²) 1/4, FFN O(n) 1/2 → 전체 ~50% (~5h)

### 검증 절차

1. 학습 완료 후 best ckpt를 `model/stage1/vec/<paramCount>/v00XX/`에 저장
2. 같은 leading-space prompt 셋 (`/tmp/ccmc-prompts-leading-space.txt`, `/tmp/ccmc-prompts-defrich.txt`)으로 `runSamplePromptsFromFile` 실행
3. 비교 baseline:
   - 현재 v0050 (blockSize 64, random offset): cat/water/dog/book 활성화 ✅, tree/sun/moon/apple 활성화 ❌
   - 새 ckpt: 같은 lemma 셋의 활성화 비율 측정
4. binding 강화 신호:
   - 활성화 lemma 수 ↑ (이전 4/10 → 새 N/10)
   - 정의 binding 응답 (e.g. "cat is an animal") 등장 빈도 ↑
   - contrast/analogy 패턴 약화 여부 (blockSize 32라 약해질 우려)
5. 만약 binding 향상이 크면 → sampling/blockSize fix가 main bottleneck이었음. 옵션 D (모델 키우기) 불필요.
   향상 없으면 → 1M capacity 진짜 한계. 옵션 D 진행.

### 영향받는 파일

- `src/main/kotlin/train/DataLoader.kt` — `ChunkAnchoredDataLoader` 클래스 추가
- `src/main/kotlin/train/TrainConfig.kt` — `chunkAnchoredSampling` field 추가
- `src/main/kotlin/vec/VecTrainer.kt` — 분기 추가 (train + val loader 생성 부분) + parallel sampler
- `src/main/kotlin/train/experiments/CcmcV2ProStage1TrainVec.kt` — blockSize 32, chunkAnchoredSampling=true

---

## v4 옵션 — 정의 binding 강화 (v0053 결과 기반)

v0053 학습 결과 분석:
- 활성화 lemma 수 큰 진전 (4 → 9~10/10) — chunk-anchored sampling fix 검증
- val 2.68 (이전 2.37보다 0.3 높음) — blockSize 32에서 contrast/analogy 35% 잘림 trade-off
- 정의 binding은 여전히 약함 — 일화/카테고리 binding만 정착 ("water → liquids" 정도)
- 학습 1h47m로 매우 빠름 → 다양한 ablation 가능

남은 진단:
- **데이터 정의 sentence 비율 부족** (원인 9): "cat is" 138건 중 정의 49%, "happy is" 6건 중 100% "to" — 데이터 자체가 정의 신호 적음. 학습으로는 못 메움
- **모델 capacity 1M** (원인 5): TinyHelen vec val 4.88, TwoStageBase 2.97 — 1M-class에서 binding은 본질적으로 약함

### 옵션 K — blockSize 64 + chunk-anchored 유지 (v4 신규)
**비용**: blockSize 32 → 64 변경. 학습 시간 ~3-4h.
**검증 목표**: blockSize 32의 val 악화가 contrast 학습 손실 때문인지 / 64로 회복되면서 활성화도 유지되는지

- 모든 다른 hyperparams 동일, blockSize만 64로
- chunk-anchored sampling 유지: record당 chunk anchor 5개 → 한 anchor ~38회 (현재 20회보다 강함)
- 기대: val 2.37 회복 + 활성화 9/10 유지 (best of both)

### 옵션 L — 정규화 약화 (v4 갱신, 옵션 E 구체화)
**비용**: hyperparam 변경 + 학습 ~2h (chunk32 setup 그대로).
**검증 목표**: 정규화가 정의 binding 정착을 막는지

설정:
- `dropout = 0.0` (현재 0.05)
- `weightDecay = 0.02` (현재 0.05)
- `labelSmoothing = 0.0` (현재 0.05)

비교: TinyHelenTrainVec (1M-class) 검증된 hyperparams. lemma binding은 본질적으로 memorization → 정규화가 logit 극단 평탄화 → binding 약화. 1h47m 빠르니 즉시 ablation.

### 옵션 M — iter 2× 증가 (v4 신규)
**비용**: maxIters 10000 → 20000, 학습 ~3.5h.
**검증 목표**: chunk-anchored 효과 극대화 — 한 anchor 평균 학습 ~40회로 strong binding (>50회 기준 근접)

- 다른 모든 설정 동일, maxIters만 20000
- 기대: val 2.5 정도, 정의 binding 정착 일부

### 옵션 N — 모델 capacity 2× (v4 갱신, 옵션 D 구체화)
**비용**: 모델 변경 (1M → ~3M), 학습 ~5-7h.
**검증 목표**: 1M의 본질 한계 검증

설정:
- 8L × 96D × 3H (~1.09M) → 8L × 192D × 6H (~3.4M) 또는 12L × 144D × 4H (~2M)
- chunk-anchored + blockSize 32 그대로
- 기대: 정의 binding 정착, val 2.0 영역

### 옵션 O — Stage 2 진행 (v4 신규)
**비용**: 0 (이미 entry 준비됨, ~30분 학습)
**검증 목표**: Stage1 v0053을 base로 instruction tuning이 binding 강화하는지

- `runCcmcV2ProStage2TrainVec --args="model/stage1/vec/1087936/v0053"`
- Stage 2 형식 ("What is a cat?\nA cat is...") 학습 → instruction prompt 활성화
- replay 0.25로 Stage1 binding 보존
- 빠른 검증 — Stage1 약함을 stage 2가 메울 수 있는지

### 옵션 P — 결합 (v4 신규, 가장 적극적)
**비용**: 다 합치면 ~5h.
**검증 목표**: 모든 fix 동시 적용

- blockSize 64 (옵션 K)
- 정규화 약화 (옵션 L)
- iter 20000 (옵션 M)
- chunk-anchored 유지

기대: val 2.2 부근 + 활성화 + 정의 binding 정착 가능성

---

## v4 권장 순서

**즉시 검증 (빠름)**:
1. **옵션 O** (Stage 2) — 30분, instruction binding 활성화 효과 측정 — Stage 2가 정의 패턴 학습할지
2. **옵션 L** (정규화 약화) — 2h, memorization 정착 효과 격리

**중기 ablation**:
3. **옵션 K** (blockSize 64) — 3-4h, val/활성화 best of both
4. **옵션 M** (iter 2×) — 3.5h, binding 신호 극대화

**근본 변경 (1M 한계 확인 후)**:
5. **옵션 N** (모델 2×) — 5-7h, capacity 한계 fix
6. **옵션 P** (모두 결합) — 5h, ablation 결과로 best config 확정 후 진행

---

## 실행 결정 (v4) — 옵션 O 채택

Stage 2 instruction tuning을 v0053 위에 진행. ~30분 학습.

### 변경 사항

**`src/main/kotlin/train/experiments/CcmcV2ProStage2TrainVec.kt`**:
- `blockSize = 64` → `32` (Stage1 v0053과 통일, Stage2 record 평균 75 tokens라 32도 충분)
- 나머지 그대로 (recordAwareSampling=true, replayRatio=0.25, learningRate=1e-4 등)

### 실행 명령

```bash
./gradlew runCcmcV2ProStage2TrainVec --args="model/stage1/vec/1087936/v0053"
```

### Stage 2 setup 요약 (이미 정의됨)

```
dataPath = "data/ccmc-v2-pro/stage2"
pretrainCheckpointDir = "model/stage1/vec/1087936/v0053"
initFrom = "pretrain_weights"  (가중치 로드, optimizer reset, iter=0)
replayDataPath = "data/ccmc-v2-pro/stage1/train.bin"
replayRatio = 0.25 (Bernoulli per minibatch)
batchSize = 2, gradientAccumulationSteps = 32
blockSize = 32 (변경)
learningRate = 1e-4 (Stage1의 1/3 — finetune 보수적)
warmupRatio = 0.05 (pretrained 보호)
maxIters = 3000
recordAwareSampling = true (chunk-anchored는 MixedDataLoader replay와 conflict라 적용 안 함)
samplePrompts = ["<|bos|>What is a cat?<|turn|>", ...]  (instruction 형식)
```

### 검증 절차

1. 학습 첫 eval: val loss가 Stage1 best 직후 수치 (~2.7)에서 시작 → "pretrain_weights" 정상 적용 확인
2. 자동 sampling 응답이 Q/A 형식으로 자연스러운지: "What is a cat?<|turn|> A cat is..." 형태
3. binding 강화 신호:
   - 정의-like 응답 등장 빈도 ↑ (Stage1 v0053 카테고리 binding "water → liquids" 수준 이상)
   - replay 0.25로 Stage1 lemma 활성화 보존
4. 만약 Stage 2가 정의 binding을 명확히 형성하면 → instruction tuning이 정의 약점 보완. Stage 1 자체 fix 불필요.
   약하면 → 옵션 L (정규화 약화) 또는 옵션 K (blockSize 64) 진행.

---

## 핵심 참고 파일 / 데이터

- `data/ccmc-v2-pro/shared/meta.json` — 현재 vocab (2000, cased, 5 special)
- `data/ccmc-v2-pro/stage1/train.txt` — 현재 학습 데이터 (3,313 records)
- `data/ccmc-v2-pro.20260506_v01/stage1/train.txt` — 이전 학습 데이터 (1,824 records, archive)
- `model/stage1/vec/1087936/v0050/` — 현재 best ckpt (val avg 2.374)
- `model/stage1/vec/1087936/_archive/run_20260506_v01/v0072/` — 이전 best ckpt (val 2.19)
- `docs/ccmc-v2-pro-stage1.md` — 이전 학습 결과 기록 (binding 진화 표 포함)
- `src/main/kotlin/sample/SamplePromptsFromFile.kt` — 옵션 A/G/H에서 사용 (현재 trim → trimEnd로 수정됨)
- `src/main/kotlin/data/SimpleBPE.kt` — `useWordPreTokenize`, `lowercase` 동작 위치
- `src/main/kotlin/train/experiments/CcmcV2ProStage1TrainVec.kt` — 옵션 E에서 hyperparams 변경 대상
- `src/main/kotlin/train/DataLoader.kt` — RecordAwareDataLoader (옵션 I 수정 대상, line 151~181)
- `/tmp/ccmc-prompts-leading-space.txt`, `/tmp/ccmc-prompts-lemma-only.txt` — v2 sampling 실험 prompt 파일
