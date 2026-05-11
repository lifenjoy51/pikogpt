# CCMC-all 모델 크기 scaling 실험 (2026-05-11)

vocab/모델 크기를 단계적으로 키우며 best val loss 추이를 측정. turbo 백엔드, tied weights (v512 제외) 기본.

## 데이터셋

- **`data/ccmc-all/`** — CCMC v2/v3/v4/v5 합본
  - v256~v1024: 3파일 (lemma_sentences + stories + dialogues), train ~18MB chars
  - **v2048부터**: 4파일 (+ wiki.txt), train ~19.7MB chars. wiki는 `\n\n` → `. ` 자연화 (옵션 D)
- Split 95:5, shuffle seed=51 (v2048부터 명시), `<|turn|>` 대화 경계 토큰 보존.

## 실험 결과

| 실험 | vocab | params | L | H | embd | tied | dropout | best val | @ iter | last iter | 비고 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| v256  |  256 |  16,560 | 2 | 1 | 20 | T | 0.1 | **3.1602** | 192k | 282k | Early stop |
| v800  |  800 |  64,128 | 6 | 1 | 24 | T | 0.1 | **3.3788** |  87k | 108k | 100k에서 사용자 종료 (수렴 전) |
| v1024 | 1024 | 111,104 | 6 | 1 | 32 | T | 0.1 | **2.9955** | 231k | 300k | maxIters 자연 종료, patience=30 (실제 plateau ~22 eval) |
| **v2048 (계획)** | 2048 | **299,376** | 7 | 1 | 48 | T | 0.1 | — | — | 300k | wiki 포함, patience=20 |

공통: blockSize=64, batchSize=8, gradientAccumulationSteps=4 (effective=32), lr=3e-4, beta2=0.95, evalIntervalRatio=0.01.

### 파라미터 분포

| 실험 | token emb | pos emb | transformer | LN+head | transformer 비중 |
|---|---:|---:|---:|---:|---:|
| v256  |  5,120 (30.9%) |  1,280 |  10,120 (61.1%) | 40 | 61.1% |
| v800  | 19,200 (29.9%) |  1,536 |  43,344 (67.6%) | 48 | 67.6% |
| v1024 | 32,768 (29.5%) |  2,048 |  76,224 (68.6%) | 64 | 68.6% |
| v2048 | 98,304 (32.8%) |  3,072 | 197,904 (66.1%) | 96 | 66.1% |

모두 tied weights (token embedding ↔ lm head 공유, lm head params 0).

## bpc 정규화 비교

val loss는 vocab에 의존 (vocab 크면 loss 작아 보임). char 기준 정보량으로 정규화:

```
bpc = val_loss × (tokens/char) × log₂(e)
random_bpc = log₂(vocab) × (tokens/char)
```

| 실험 | val | tokens/char | bpc | random bpc | bpc/random |
|---|---:|---:|---:|---:|---:|
| v256  | 3.1602 | 0.456 | **2.08** | 3.65 | 57.0% |
| v800  | 3.3788 | 0.315 | **1.53** | 3.04 | 50.4% |
| v1024 | 2.9955 | 0.295 | **1.27** | 2.95 | 43.1% |

**관찰**:
- params 16k → 111k에서 bpc 2.08 → 1.27 (40% 개선).
- bpc/random은 단조 감소 (vocab 영향 정규화한 절대 척도). v1024는 random의 43.1%까지 도달.
- v800은 조기 종료(108k iter)로 미수렴 — 더 학습했으면 v1024 수준 도달 가능 추정.

## 정성적 샘플링 결과 (T=0.5, 동일 prompt 세트 15개)

각 모델의 **best ckpt와 last ckpt 모두** 샘플링. 전체 로그는 `logs/sample-{v256,v800,v1024}-{best,last}-T05.log` (6개 파일). 대표 prompt 3개 비교:

| 모델 | best ckpt | last ckpt | val (best=last) |
|---|---|---|---:|
| v256  | v0065 (iter 192k) | v0095 (iter 282k) | 3.1602 |
| v800  | v0030 (iter 87k)  | v0037 (iter 108k) | 3.3788 |
| v1024 | v0078 (iter 231k) | v0101 (iter 300k) | 2.9955 |

(best 갱신 후 plateau에서 학습 계속되므로 best val = last val. last ckpt는 plateau 이후 drift 시기.)

### Prompt: `the cat`

| 모델 | ckpt | 생성 (앞부분) |
|---|---|---|
| v256  | best | `eople to visite is freation. we did youry after. the cards. it is alesse beautifult, i know the start and wormo.` |
| v256  | last | `e of pre. the old?` (조기 EOS/turn stop) |
| v800  | best | `. it is a best day. the mal has apple. i find a big halcubm, but i am so we use the fark in the garden.` |
| v800  | last | `. it is a felicicive and a small place. the pet is a big mine. i am happy to see my hands and i am happy today.` |
| v1024 | best | `last week, my friend says it is a new holide. she shows me to buy anymore. he was very easy to buy it.` |
| v1024 | last | `ch in the park. i want to buy a new prime number. i see a big milk dreweller for my friends.` |

### Prompt: `i went to the`

| 모델 | ckpt | 생성 |
|---|---|---|
| v256  | best | `blany a dad. my it is very canderstose, i after. i see the creentleaste a comethy to wall comp.` |
| v256  | last | `blany a look is quice is enjoy about day, i taked. the brounders are not ales to learnings to the teacher says it.` |
| v800  | best | `cupht for the park. i like to buy a new bedy. the teacher said it is a big mine. i like to learn about the garden.` |
| v800  | last | `cup. i am happy and i like to buy a new voes. the teacher said it is a big mine. i like to learn about the park.` |
| v1024 | best | `big museum. i see a bark. we can see many city with my family. i like to go to the shop and store.` |
| v1024 | last | `new museum. i want to see a big vacation. is in your town?` |

### Prompt: `do you like`

| 모델 | ckpt | 생성 |
|---|---|---|
| v256  | best | `to buy a good today. i see the grury to outseople to learn about the books is skton.` |
| v256  | last | `to buy time. i want to many flater. i love a new greensele. i see the share the clownt is interesting...` |
| v800  | best | `a picture. i imagine the favorite new things. the bestion is a beautiful place. i remember the teacher says the dear.` |
| v800  | last | `a long age because it is good to remember them. i want to learn about the old book. what do you understand?` |
| v1024 | best | `a nice dream. i love to drawing about the saturday. my mom says it is special because we have a good day. what do you want to buy?` |
| v1024 | last | `a nice dem. i see the new vacation in my room. i love to watch it with him. we saw at the small dinosaurs on her hands.` |

### 정성적 비교

| 척도 | v256 (16k) | v800 (64k) | v1024 (111k) |
|---|---|---|---|
| 단어 인식 | 깨진 글자 조합 다수 (`freation`, `clunder`, `cardent`) | 깨진 단어 일부 (`halcubm`, `fictell`, `snhia`) | 거의 실제 영어, 드물게 깨짐 (`vacuction`, `dreweller`) |
| 문법 구조 | 단편적 phrase, 종종 의미 단절 | 짧은 문장 단위 의미 형성 | 자연 문장 흐름, multi-paragraph |
| Q-A 패턴 | 미형성 | 부분 형성 ("can you can buy a games?") | 형성 ("what is your" → "favorite vacuction?", "where are you" → "go to the park?") |
| 주제 일관성 | 없음 | "learn about" 패턴 반복 (overfit 신호) | 문맥 다양 (museum/Brazil/dinosaurs/prime number 등 주제어 풍부) |

### best vs last 차이 관찰

- **v256**: last가 짧게 EOS로 끊기는 경향 (`the cat` → `e of pre. the old?`). plateau 이후 noise drift로 stop token 확률 약간 증가한 듯.
- **v800**: best/last 모두 비슷한 길이/품질. 어휘는 약간씩 다름 (`halcubm` ↔ `felicicive`).
- **v1024**: best와 last 모두 자연 문장. last가 더 다양한 명사 등장 (`prime number`, `dinosaurs`, `vacuum`). plateau 이후에도 surface form은 계속 변동.

**핵심 관찰**: val loss는 동일해도 last ckpt는 plateau에서 계속 학습되며 표현 다양성이 살짝 증가. 그러나 핵심 quality 척도(단어 인식, 문법, Q-A)는 best와 last가 거의 동일. → best ckpt가 deployment용으로 충분.

**모델 scaling jump**: 64k → 111k에서 "글자 조합" → "실제 단어 + 문장 흐름"으로 질적 변화. bpc 1.53 → 1.27의 0.26 감소가 가시적 품질 차이로 나타남.

## 학습 곡선 관찰

### v1024 (300k iter)
- best val 갱신 이력: v0004 → v0049 → v0058 → **v0078 (iter 231k, val 2.9955)**.
- v0078 이후 22 eval 동안 plateau (best 미갱신).
- earlyStopPatience=30이 헐겁게 잡혀 maxIters로 자연 종료. → **v2048부터 patience=20** 으로 축소.

### v256 (vs v1024)
- 16k tiny 모델이 val 3.16 도달 (iter 192k). bpc=2.08.
- 동일 코퍼스, vocab만 4배 키우고 모델 7배 키운 v1024(bpc 1.27)와 절대 척도(bpc)로 비교하면 약 40% gap.

## 모델 크기 단계 추이

```
v256 (16k)  ──→  v800 (64k)  ──→  v1024 (111k)  ──→  v2048 (300k 계획)
  bpc 2.08         bpc 1.53          bpc 1.27          (다음)

  params×7 (16k→111k) → bpc 절반(2.08→1.27). vocab+모델 동시 scale.
```

## 다음 실험 (v2048)

- **목적**: params 300k 도달 + wiki 코퍼스 포함 효과 측정.
- **변경 사항** (v1024 대비):
  - vocab 1024 → **2048** (sub-word coverage ↑, tokens/char 0.295 → 0.255)
  - embd 32 → **48** (1.5×)
  - L 6 → **7** (transformer 비중 66.1% 유지)
  - earlyStopPatience 30 → **20** (v1024 plateau 22 eval 관찰 반영)
  - 코퍼스 +wiki.txt (3,940 records, `\n\n` → `. ` 자연화)
- **예상**: 학습 시간 ~10시간 (v1024 5시간의 2배), bpc 1.0 근처 도달 시 성공.

## 진입점 및 재현

```bash
# 1. raw 합본 → 95:5 split (seed=51)
python3 scripts/build_ccmc_all.py

# 2. BPE 인코딩 (vocab=2048)
mkdir -p data/ccmc-all-v2048
cp data/ccmc-all/{train,val}.txt data/ccmc-all-v2048/
./gradlew runBpe --args="data/ccmc-all-v2048 2048"

# 3. 학습 (v2048 config는 WordStart4PrefixTrainTurbo.kt에 정의)
./gradlew runWordStart4PrefixTrainTurbo 2>&1 | tee logs/v2048-300k-tied-300k.log
```

체크포인트: `model/ccmc-all-v2048/main/v0001/`부터 매 3000 iter 저장.

## 참고

- 백엔드: turbo (`TurboTrainer`, `TurboTrainConfig`). scalar 백엔드는 1M 미만 빠른 prototyping 가능하지만, 이 scaling 실험은 모두 turbo.
- micro-batch 효율: turbo 1M params iter당 ~2.6초 기준, v2048 300k는 ~12k 코어 초 ≈ 10시간 (8코어 ForkJoinPool 병렬).
- 모든 ckpt는 `alwaysSaveCheckpoint=true`로 저장 (best 갱신 외에도 매 eval).
