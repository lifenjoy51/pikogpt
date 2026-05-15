# CCMC-all 모델 진화 — 전체 통합 (2026-05-08 → 2026-05-15)

`data/ccmc-all/` 합본 코퍼스 위에서 점진적으로 학습한 **10개 모델**의 통합 보고. vocab 크기 ablation부터 1M 모델 + vocab 확장까지 cycle별 변화를 한 자리에 정리.

기존 문서 [ccmc-all-scaling-2026-05-11.md](ccmc-all-scaling-2026-05-11.md)는 Phase 1(v256~v2048 계획)까지의 스냅샷. 이 문서는 그 이후의 Phase 2/3을 합친 **전체 진화 기록**.

## 1. 명명 체계

ccmc-all 디렉토리 트리는 두 단계 명명이 섞여 있다:

```
model/ccmc-all-v{vocab}              ─ Phase 1: vocab ablation (작은 모델)
model/ccmc-all-v2048-v{cycle}        ─ Phase 2: vocab=2048 고정 + 모델/데이터 진화
model/ccmc-all-v{vocab}-v{cycle}     ─ Phase 3: vocab 확장 (4096) 새 cycle
```

| Phase | 시기 | 디렉토리 | 의미 |
|---|---|---|---|
| 1 | 2026-05-08 ~ 05-11 | `ccmc-all-v{256,800,1024,2048}` | vocab 크기별 baseline (작은 모델) |
| 2 | 2026-05-12 ~ 05-13 | `ccmc-all-v2048-v{6,7,8}` | vocab 2048 고정, 모델·데이터 변경 cycle |
| 3 | 2026-05-14 ~ (진행 중) | `ccmc-all-v4096-v1` | vocab 4096으로 첫 cycle |

대화 중 편의상 "v9"로 부른 모델은 디렉토리상 `ccmc-all-v4096-v1`이다.

## 2. 데이터셋 진화

```
data/ccmc-all/{train,val}.txt  ←  스크립트 `scripts/build_ccmc_all.py`로 매번 재생성
                                  raw 입력 디렉토리: data/ccmc-all-raw/
```

| Phase | 파일 구성 | 특수 처리 | tokens (train) |
|---|---|---|---:|
| 1 — v256~v1024 | lemma_sentences + stories + dialogues (3파일) | — | varies (vocab별) |
| 1 — v2048 | + wiki.txt (4파일) | wiki `\n\n` → `. ` 자연화 (옵션 D) | 5,029,658 (vocab 2048) |
| 2 — v6 | + cause_seq.txt + chained.txt + counting.txt (7파일) | lemma_sentences 문장 단위 분리, hapax 5세 단순화 | 6,398,923 (vocab 2048) |
| 2 — v7 / v8 | 7파일 (v6과 동일) | + `<\|bos\|> ... <\|eos\|>` record 래핑 | 6,977,153 (vocab 2048) |
| 3 — v4096-v1 | 7파일 (v7/v8과 동일) | BPE 재학습 vocab=4096 | 6,258,591 (vocab 4096) |

특수 토큰 슬롯: `0 = <\|eos\|>`, `1 = <\|unk\|>`, `2 = <\|bos\|>`, `3 = <\|turn\|>`, `4 = <\|sep\|>`. Phase 2부터 BOS/EOS 적극 활용.

## 3. 통합 비교표 — 10개 모델 best ckpt

| Phase | 모델 | embd | L | H | vocab | tied | params | best iter | val | **bpc** | random bpc | wall-clock |
|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | v256 | 20 | 2 | 1 | 256 | ✓ | 16,560 | 192,000 | 3.160 | **2.081** | 3.652 | — |
| 1 | v800 | 24 | 6 | 1 | 800 | ✓ | 64,128 | 87,000 | 3.379 | **1.534** | 3.033 | — |
| 1 | v1024 | 32 | 6 | 1 | 1024 | ✓ | 111,104 | 231,000 | 2.996 | **1.273** | 2.945 | ~5h |
| **1** | **v2048** | 48 | 7 | 1 | 2048 | ✓ | 299,376 | 249,000 | 2.813 | **1.035** | 2.806 | ~13h |
| 2 | v2048-v6 | 64 | 7 | 1 | 2048 | ✓ | 485,184 | 150,000 | 2.852 | **1.066** | 2.806 | 11h 51m |
| 2 | v2048-v7 | 64 | 7 | 2 | 2048 | ✓ | 485,184 | 183,000 | 2.603 | **0.960** | 2.806 | ~6h |
| 2 | v2048-v8 | 96 | 7 | 3 | 2048 | ✓ | 985,824 | 142,400 | 2.099 | **0.775** | 2.806 | 17h 23m |
| 3 | **v4096-v1** | 96 | 6 | 3 | **4096** | ✓ | 1,070,592 | 73,200 | 2.330 | **0.771** | 2.752 | ~12.2h (best) / 15.3h (총) |

v4096-v1은 사용자 중단(iter 87,500, 종료 시점 평균 2.42)으로 종료. Best ckpt는 **v0063 (iter 73,200, val 2.3302, bpc 0.771)** — v8 best(0.775)에 미세 우위. patience=20 도달 직전(12 eval 미갱신)에 다음 cycle(v10) 준비를 위해 종료.

공통 설정 (별도 언급 없으면): `blockSize=64`, `batchSize=8`, `evalIntervalRatio=0.01`, `warmupRatio=0.05`, `weightDecay=0.01`, `gradClip=1.0`, `beta1=0.9`, `minimumLearningRate=1e-5`, `learningRateDecayRatio=0.95`.

Cycle별 핵심 hyperparam 차이:

| 모델 | lr | β₂ | dropout | gradAccum | maxIters | patience |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 (v256~v2048) | 3e-4 | 0.95 | 0.1 | 4 (eff batch 32) | 282k~300k | 20~30 |
| v2048-v6 | 3e-4 | 0.95 | 0.1 | 4 | 300k | 20 |
| v2048-v7 | 3e-4 | 0.95 | 0.1 | 4 | 300k | 20 |
| v2048-v8 | **2e-4** | **0.99** | **0.05** | **8** (eff batch 64) | 160k | 20 |
| v4096-v1 | 2e-4 | 0.99 | 0.05 | 8 | 120k | 20 (12/20 도달 시 중단) |

## 4. bpc 진화 추이

bpc = `val_loss / ln(2) / chars_per_token`. vocab 영향이 정규화된 절대 압축률 척도 — cross-model 비교의 유일한 fair metric.

```
random vocab=256       3.652 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
random vocab=2048      2.806 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
random vocab=4096      2.752 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
───────────────────────────────
v256  (16k params)     2.081 │ ▓▓▓▓▓▓▓▓▓▓
v800  (64k params)     1.534 │ ▓▓▓▓▓▓▓
v1024 (111k params)    1.273 │ ▓▓▓▓▓▓
v2048-v6 (485k, 7파일)  1.066 │ ▓▓▓▓▓
v2048   (300k baseline) 1.035 │ ▓▓▓▓▓
v2048-v7 (485k +BOS/EOS) 0.960 │ ▓▓▓▓▓
v2048-v8 (986k 1M)      0.775 │ ▓▓▓▓
v4096-v1 (1M, vocab×2)  0.771 │ ▓▓▓▓
```

Phase별로 가장 의미 있는 단일 변경:

| 단계 | 변경 | bpc Δ | 효과 |
|---|---|---:|---|
| v256 → v1024 | params 16k → 111k (7×) + vocab 256 → 1024 | -0.808 | "글자 조합" → "실제 단어 + 짧은 문장" |
| v1024 → v2048 | + wiki.txt + params 111k → 300k | -0.238 | wiki 톤 학습, 어휘 풍부화 |
| v2048 → v2048-v6 | + 3 axes + lemma split + width 48→64 | **+0.031 (후퇴)** | 데이터/모델 단순 확장은 효과 없음 |
| v6 → v2048-v7 | + BOS/EOS 래핑 + heads 1→2 | **-0.106** | record boundary 학습 결정적 |
| v7 → v2048-v8 | embd 64→96, heads 2→3, lr/dropout 튜닝, gradAccum 4→8 | **-0.185** | 1M scale + HP 정교화 |
| v8 → v4096-v1 | vocab 2048 → 4096, depth 7→6 | **-0.004 (실질 동등)** | vocab 확장은 정량 효과 미미, 정성 entity 표현은 강화 |

## 5. Phase별 정성 평가 (T=0.5, best ckpt)

각 모델의 best ckpt에서 동일한 prompt 3개 응답 비교. 전체 15-prompt 결과는 `logs/sample-{model}-best-T05.log` 참조.

### Prompt: `the cat`

| 모델 | best ckpt | 생성 (앞부분) |
|---|---|---|
| v256 | v0065 | `eople to visite is freation. we did youry after. the cards. it is alesse beautifult, i know the start and wormo.` |
| v800 | v0030 | `. it is a best day. the mal has apple. i find a big halcubm, but i am so we use the fark in the garden.` |
| v1024 | v0078 | `last week, my friend says it is a new holide. she shows me to buy anymore. he was very easy to buy it.` |
| v2048 | v0086 | `s are called a secret. the church of colombia is like 20, and it is very big. i think colombia is an interesting country.` |
| v6 | v0052 | `s are called a mouse. they have more than thick clays, and some have the same number. in total, there are three bones that make it very strong.` |
| v7 | v0064 | `s are also a big elephant.` |
| v8 | v0090 | `s are both small, but some are big.` |
| v9 | v0063 | `is very smart. i like to use my smart brain every day.` |

### Prompt: `i went to the`

| 모델 | best ckpt | 생성 |
|---|---|---|
| v256 | v0065 | `blany a dad. my it is very canderstose, i after. i see the creentleaste a comethy to wall comp.` |
| v800 | v0030 | `cupht for the park. i like to buy a new bedy. the teacher said it is a big mine. i like to learn about the garden.` |
| v1024 | v0078 | `big museum. i see a bark. we can see many city with my family. i like to go to the shop and store.` |
| v2048 | v0086 | `park and saw a big tree. i could see my dad's toy car on the floor. he said we can go to the bus, but it is very big. now she wants to visit him.` |
| v6 | v0052 | `park and saw a big mountain. it was very tall and bright. i like to see how birds fly in the park. my friend said we can see the green trees.` |
| v7 | v0064 | `park to see a big city. i saw a new city with many trees and green trees. my dad said we can see the old calendar in the park.` |
| v8 | v0090 | `park and saw a big red ball. he wore it every day after school.` |
| v9 | v0063 | `park and saw a big river. it was a very old river in the country of china. he built tall buildings with big stone buildings. the river was so long that he could not see many houses.` |

### Prompt: `do you like`

| 모델 | best ckpt | 생성 |
|---|---|---|
| v256 | v0065 | `to buy a good today. i see the grury to outseople to learn about the books is skton.` |
| v800 | v0030 | `a picture. i imagine the favorite new things. the bestion is a beautiful place. i remember the teacher says the dear.` |
| v1024 | v0078 | `a nice dream. i love to drawing about the saturday. my mom says it is special because we have a good day. what do you want to buy?` |
| v2048 | v0086 | `a funny story with you. my friend also likes to talk about the funny story. i love to listen to her stories.` |
| v6 | v0052 | `to write a new song. first, i write the letter on my paper. then, she writes a story about her cat. last, we both think it is nice.` |
| v7 | v0064 | `to cook a big meal for me.` |
| v8 | v0090 | `to ride a bike for fun.` |
| v9 | v0063 | `to swim in the pond. first, it walks around the water. then, it swam away. last, it swims away from its home.` |

### 정성 진화 요약 (척도별)

| 척도 | v256 | v800 | v1024 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|---|---|
| 단어 인식 | 깨진 글자 다수 | 일부 깨짐 | 거의 영어 | 깨끗 | 깨끗 | 깨끗 | 깨끗 |
| 문법 | 단편 phrase | 짧은 문장 | 자연 문장 | 자연 문장 | 자연 문장 | 자연 영어 | 자연 영어 |
| 응답 길이 | 길고 nonsense | 중간 | 중간~긴 | 긴 wiki 톤 | 짧고 정확 | 매우 짧음 (1 sentence) | **양극화 (1문장 ↔ wiki 장문)** |
| EOS 학습 | 약함 | 부분 | 부분 | 부분 | **강함** | 강함 | **과도** (lemma 영향, 단문 prior) |
| 의미 일관성 | 없음 | 약함 | 부분 | 부분 | 양호 | **양호** | 부분 (entity drift 있음) |
| 사실 정확도 | — | 무관 | 무관 | 부분 | 부분 | **부분 + 일부 오류** | 부분 + 환각 (`khalabra`, `pajabo`) |

## 6. Phase별 핵심 발견

### Phase 1 (vocab ablation)

- params 16k → 111k → 300k에서 bpc 2.08 → 1.27 → 1.04 (단조 감소).
- **v1024(111k) → v2048(300k)에서 wiki 코퍼스 추가**가 어휘 풍부화 견인. vocab 확장만으로는 한계.

### Phase 2 (cycle 진화, vocab 2048 고정)

#### v6 wider (485k, 7파일, lemma split, no BOS/EOS)
- 데이터 양 +35% (5.0M → 6.4M tokens) + width 1.6× (48 → 64)
- **bpc 후퇴: 1.035 → 1.066** (+3%)
- 가설: lemma_sentences 문장 단위 분리로 record당 평균 토큰 수 감소 → 짧은 context 학습 비중 ↑ → wiki/stories의 긴 흐름 학습 손실 가능성. 또는 단순 noise.

#### v7 (485k 동일, +BOS/EOS, heads 2)
- 동일 params에서 **bpc 1.066 → 0.960 (-0.106)**. 단일 변경 중 가장 큰 도약.
- BOS/EOS 래핑 → 모델이 record 시작/종료를 명시적 토큰으로 학습 → EOS 확률 분포 학습 → 자연 종료 학습 → 전체 token 분포 모델링 ↑.
- heads 1 → 2: head_dim 64 → 32. attention representation 분할로 multi-head 효과.

#### v8 (986k, 1M scale, HP 정교화)
- embd 64 → 96, heads 2 → 3 (head_dim 32 유지), lr 3e-4 → 2e-4, dropout 0.1 → 0.05, gradAccum 4 → 8 (eff batch 64), β₂ 0.95 → 0.99.
- **bpc 0.960 → 0.775 (-0.185)**.
- depth 7 유지하며 width 1.5×. params 2배(485k → 986k).
- 더 작은 lr + 더 큰 effective batch + 더 약한 dropout = 큰 모델 + 충분한 데이터에 맞춘 안정화. β₂ 0.99로 Adam 2차 모멘트 smoothing 강화.

### Phase 3 (vocab 확장)

#### v9 = v4096-v1 (1.07M, vocab 4096, L=6)
- vocab 2배(2048 → 4096). depth 1 축소(7 → 6)로 params 유지.
- token emb 비중 19.9% → 36.7%로 회복 (v8은 1M 모델 대비 vocab 작아 transformer 비중 과대).
- best: **iter 73,200, val 2.3302, bpc 0.771** (v0063). v8 best(0.775)와 사실상 동일 — vocab 확장의 정량 효과 미미.
- 학습 곡선: iter 16.8k에서 이미 bpc 0.902 (v7 final 추월), iter 49.2k에서 bpc 0.778, iter 73.2k에서 최종 best. 그 후 plateau, 12 eval 미갱신.
- 정성 강화: wiki 톤 자연 문장 형성 ("hello → a small city in the southern europe..."), entity 묘사 풍부 ("china/colombia/europe" 등장). 하지만 단답 prior 강함 (`do you like` → 한 문장으로 끝), entity drift도 잔존 (cat/dog → 동일 wiki 묘사 템플릿 재활용).
- **bpc 정체의 진단**: 단순 vocab 확장은 EOS 학습 신호의 비대칭(lemma 라인 EOS/chunk 6.5× vs 다른 소스 1×)을 해결 못 함. 다음 cycle에서 lemma stream 가중치 축소로 시도.

## 7. 학습 곡선 — Phase 2/3 상세

### v6 학습 진척
- best 갱신: v0036 → v0044 → **v0052 (iter 150,000, val 2.852)**.
- 이후 plateau, 22 eval 지속 후 patience 도달 가능했으나 maxIters 300k 자연 종료 (실제 best 후 1.5× 더 학습).
- wall-clock 11h 51m.

### v7 학습 진척
- best 갱신: v0021 → v0030 → ... → **v0064 (iter 183,000, val 2.603)**.
- v0064 이후 best 갱신 1회 더 가능했으나 patience 발동 또는 사용자 중단. 마지막 v0066.
- BOS/EOS 효과 즉시 가시: iter 5k부터 짧은 prompt에 자연 EOS.

### v8 학습 진척
- best 갱신: v0008 → ... → **v0090 (iter 142,400, val 2.099)**.
- 이후 plateau, patience=20 도달로 종료. v0101이 마지막 ckpt (iter 160k).
- 17h 23m. 1M 모델 + eff batch 64 + lr 2e-4의 가장 안정적인 학습 곡선.

### v9 학습 진척 (2026-05-14 → 05-15)
```
iter   1,200  val 6.210  bpc 2.054  v0002
iter   2,400  val 4.778  bpc 1.581  v0003
iter   4,800  val 3.887  bpc 1.286  v0006
iter   7,200  val 3.400  bpc 1.125  v0008
iter  10,800  val 2.979  bpc 0.986  v0011
iter  16,800  val 2.725  bpc 0.902  v0016
iter  49,200  val 2.345  bpc 0.776  v0043
iter  73,200  val 2.330  bpc 0.771  v0063 ← final best
iter  87,500  (사용자 중단, plateau 12/20)
```
- 6시간 안에 v7 final 추월, 12시간에 best 도달.
- iter rate 0.61 s/iter (8 워커) → 0.59 s/iter (4 워커 resume 후).
- best 갱신: v0008 → ... → v0043 → v0063 (총 20+ 차례). v0043 이후 17 eval 정체 후 v0063에서 1회 추가 갱신, 그 후 다시 12 eval plateau.
- 최종 bpc 0.771이 v8 0.775 대비 미미한 개선 — vocab 확장은 추가 cycle 없이는 saturation 도달.

## 8. 비용 (wall-clock, ForkJoinPool 8 workers, JDK 21)

| 모델 | params | iter rate | best iter | 누적 wall-clock |
|---|---:|---:|---:|---:|
| v1024 | 111k | ~0.08 s/iter | 231k | ~5h |
| v2048 | 300k | ~0.15 s/iter | 249k | ~13h |
| v6 | 485k | ~0.14 s/iter | 150k | 11h 51m |
| v7 | 485k | ~0.14 s/iter | 183k | ~6h (best까지) |
| v8 | 986k | ~0.39 s/iter | 142k | 17h 23m |
| v9 | 1,070k | ~0.61 s/iter | 73,200 | 12.2h (best) / 15.3h (총) |

vocab 2배 + params 8% 증가 + depth 1 감소가 iter rate ~56% 증가로 나타남. softmax + tied lm head matmul 비용이 vocab에 선형으로 들어간다.

## 9. 핵심 결론

1. **BOS/EOS 래핑은 동일 params에서 가장 cost-effective한 단일 변경**. 데이터 prep만 바꾸면 됨. -0.106 bpc.
2. **v2048-v6의 후퇴 사례**가 보여주듯 "데이터 더 + 모델 폭 더"의 단순 확장은 효과 없을 수 있음. **변경 변수 isolation** 필수.
3. **1M scale + lr 2e-4 + eff batch 64 + dropout 0.05**가 본 코퍼스에서 안정 동작하는 sweet spot. v8 종료 시점에서 train(2.04) vs val(2.10) 격차 0.06 — overfit 없음.
4. **vocab 4096은 동일 params에서 token emb 비중을 30%대로 회복**. 정성 효과는 entity 표현력 ↑ (`the cat → has a long, thin, flat tail`).
5. **bpc 0.77 ↔ 1.07M params**가 본 코퍼스의 capacity 부근. Chinchilla 비율(20:1)로 보면 1M 모델에 ≥20M tokens 필요하나 현재 ~6.3M — under-data 상태.

## 10. 다음 실험 방향

v9 종료 분석으로부터 도출된 다음 cycle:

| 방향 | 변경 | 기대 |
|---|---|---|
| **v10 (LemmaW10, 준비됨)** | v9 동일 모델 + lemma stream을 weighted source loader로 secondaryProb=0.1 sampling. val도 동일 분포. | EOS/chunk 1.75 → ~1.18로 균형 → 단답 prior 완화 + 장문 generation 강화. bpc는 동등 or 약간 개선 예상 |
| b) 데이터 확장 | ccmc-all-raw에 신규 axes 추가 → tokens 10M+ 목표 | under-data 해소, bpc 0.7 미만 |
| c) depth 늘리기 | embd 96 + L 8 + vocab 2048 (~1.1M, depth ↑) | depth vs width trade-off 측정 |
| d) RoPE / SwiGLU / GQA | turbo 옵션 활용 (v2048 동일 코퍼스) | architecture ablation |
| e) Chinchilla 가까운 모델 | params 300k + tokens 6M (= 20:1) → bpc 비교 | 모델 vs 데이터 균형 검증 |

권장 순서: **v10 → b → c**. d/e는 별도 ablation track.

### v10 핵심 변경 (`CcmcAllV4096M1LemmaW10TrainTurbo`)
- `data/ccmc-all-v4096-v2/` — lemma/other 분리 BPE (train_lemma/train_other/val_lemma/val_other 4-stream)
- `train.WeightedSourceDataLoader` — batch sequence별 베르누이 draw (p_lemma=0.1)
- `TurboTrainConfig.lemmaSamplingRatio: Float?` 옵션 추가
- v9와 모든 hyperparam 동일 (model 1.07M, lr 2e-4, β₂ 0.99, dropout 0.05, gradAccum 8, blockSize 64, maxIters 120k)

## 진입점 및 재현

```bash
# 데이터 prep (Phase 2/3 공통)
python3 scripts/build_ccmc_all.py

# Phase별 학습
./gradlew runCcmcAllV2048TrainTurbo                   # v2048 baseline
./gradlew runCcmcAllV2048WiderTrainTurbo              # v6
./gradlew runCcmcAllV2048WiderH2TrainTurbo            # v7
./gradlew runCcmcAllV2048M1TrainTurbo                 # v8
./gradlew runCcmcAllV4096M1TrainTurbo                 # v9
./gradlew runCcmcAllV4096M1LemmaW10TrainTurbo         # v10 (lemma stream weight 0.1)

# Resume
./gradlew runCcmcAllV4096M1TrainTurbo --args="resume"
./gradlew runCcmcAllV4096M1LemmaW10TrainTurbo --args="resume"
```

데이터 prep:
```bash
python3 scripts/build_ccmc_all.py        # 단일 합본 (v9 이전)
python3 scripts/build_ccmc_all_split.py  # lemma/other 분리 (v10)
```

체크포인트: `model/ccmc-all-{vocab}{-cycle}/main/v{NNNN}/`. 매 evalIntervalRatio(=0.01)마다 저장.

샘플링:
```bash
./gradlew runSamplePromptsFromFile \
    --args="model/ccmc-all-v4096-v1/main/v0063 prompts/ccmc_15.txt"
```

## 참고

- 백엔드: turbo (`TurboTrainer`, `TurboPikoGPT`). 모두 SIMD + ForkJoinPool 8 workers.
- 모든 모델 `alwaysSaveCheckpoint=true` — best 외에도 매 eval 저장.
- bpc 계산은 `val_loss / ln(2) / chars_per_token`. `chars_per_token`은 각 데이터셋의 `train.txt` 크기 / `train.bin` 토큰 수.
- 기존 Phase 1 상세: [ccmc-all-scaling-2026-05-11.md](ccmc-all-scaling-2026-05-11.md)
- Hapax 단순화: [ccmc-hapax-refinement.md](ccmc-hapax-refinement.md)
- v6 multi-axis 계획: [ccmc-v6-multi-axis-plan.md](ccmc-v6-multi-axis-plan.md)
