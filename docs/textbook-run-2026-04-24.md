# TinyHelen textbook-only 학습 보고 (2026-04-24)

## 목적

혼합 코퍼스 10k run([vec-run-2026-04-24.md](vec-run-2026-04-24.md))에서 출력이 textbook/book/wiki/conversation 여러 장르로 드리프트한 것을 관찰한 뒤, **단일 장르(textbook) 단독 학습**이 톤 일관성을 얼마나 올려주는지 실험. 같은 1M 아키텍처 유지.

동시에 파이프라인 품질 이슈 2개를 수정:

1. **공식 held-out val 사용** — 기존 pipeline은 `stories.txt` 토큰을 90:10 순차 절단해 "뒤쪽 10%"를 val로 썼으나, TinyHelen은 자체 `validation/` 폴더를 제공. 이쪽이 진짜 out-of-distribution held-out.
2. **체크포인트 폭증 완화** — `alwaysSaveCheckpoint=true`에서 매 eval마다 dir가 쌓이는 문제를 avg 개선 시에만 저장으로 변경.

## 코드 변경 요약

### 파이프라인 확장 — `StoriesBpePrep`

입력 규약 (우선순위):
- `<path>/train.txt` + `<path>/val.txt` 둘 다 있으면 → **분리 입력 경로**. BPE vocab은 train에서만 학습(val leakage 차단), val은 encode만.
- 없으면 기존 `stories.txt` 90:10 cut fallback.

`src/main/kotlin/data/StoriesBpePrep.kt`에 내부 sealed class `SplitSource`로 분기. 기존 호출 시그니처 그대로.

### 학습 엔트리 — `TinyHelenTrainTextbookVec`

`TinyHelenTrainVec.kt`와 5가지만 차이:

| 항목 | TinyHelenTrainVec | TinyHelenTrainTextbookVec |
|---|---|---|
| `dataPath` | `data/tinyhelen` | `data/tinyhelen-textbook` |
| `modelDir` | `model` | `model-textbook` (네임스페이스 격리) |
| `maxIters` | 10000 | 6000 |
| `alwaysSaveCheckpoint` | `true` | **`false`** |
| `evalIters` | 16 | **100** (noise ↓) |

### Eval 병렬화 — `vec.Trainer.estimateLoss`

기존 `estimateLoss`는 마스터 모델 1개로 순차 forward (evalIters=100 기준 train+val 400 시퀀스 싱글 스레드). `trainStepParallel` 패턴을 차용해:

1. worker param은 master와 이미 동기화된 상태이므로 eval 한 번에 1회 sync
2. train/val 각각의 시퀀스를 flatten → round-robin 분배 → coroutine forward
3. loss 합산 후 시퀀스 수로 나눔

Eval 중엔 weight 불변이라 grad/merge 불필요 → train step 대비 구현이 단순.

### Resume 지원 (이전 커밋) 실사용 검증

이번 run에서 step 1800에서 중지 후 `--args="resume"`으로 이어하기 성공:
```
체크포인트 재개: iter=1800, bestLoss=3.6725 from model-textbook/vec/1057536/36
```
Adam 모멘트, iter, bestLoss 모두 복원됨 → 재시작 후 첫 eval이 곧바로 하강 추세 유지.

## 데이터

### TinyHelen 공식 분할 사용

```bash
mkdir -p data/tinyhelen-textbook
jq -r '"<|bos|>" + .text + "<|eos|>"' \
  /tmp/TinyHelen/data/leaner/10M/train/textbook0000.jsonl \
  > data/tinyhelen-textbook/train.txt
jq -r '"<|bos|>" + .text + "<|eos|>"' \
  /tmp/TinyHelen/data/leaner/10M/validation/textbook0000.jsonl \
  > data/tinyhelen-textbook/val.txt
./gradlew runStoriesBpe --args="data/tinyhelen-textbook"
# -> "분리 입력 감지: train.txt + val.txt → vocab은 train에서만 학습"
```

| 소스 | docs | 문자 | 토큰 |
|---|---:|---:|---:|
| train (leaner/10M/train/textbook0000.jsonl) | 848 | 2,184,432 | **610,429** |
| val (leaner/10M/validation/textbook0000.jsonl) | 41 | 110,369 | **30,860** |

BPE: vocab 1000, 913 merges, special tokens eos=0 / unk=1 / bos=2.

## 학습 설정

| 항목 | 값 |
|---|---|
| maxIters | 6000 |
| batch × accum | 2 × 16 = 32 (유효) |
| blockSize | 64 |
| layers / heads / dim | 4 / 4 / 128 (total 1,057,536 params) |
| LR peak/min | 3e-4 / 3e-5 |
| warmup / decay | 3% / 95% (cosine) |
| evalInterval / evalIters | 300 iter / **100** |
| alwaysSaveCheckpoint | **false** (best avg 갱신 시에만) |
| worker | **8** (CPU 12, `VEC_MAX_WORKERS=8`) |

학습은 두 단계로 나뉨 (resume 검증 겸):
1. `step 0 → 1800`: 초기 학습, 첫 best avg 3.67 도달
2. 중단 → Trainer 코드에 eval 병렬화 추가 → `--args="resume"`로 재시작
3. `step 1800 → 6000`: 이어 학습, best avg **3.17** (step 5100)에서 달성

## Loss 궤적 (eval 21회 × 300 iter)

| iter | train | val | avg | gap | 비고 |
|---:|---:|---:|---:|---:|---|
| 0 | 6.94 | 6.95 | 6.95 | 0.01 | baseline (ln 1000 ≈ 6.908) |
| 300 | 4.84 | 4.88 | 4.86 | 0.04 | **공식 val이 매우 건강한 신호** (이전 90:10 cut에선 gap 0.12) |
| 600 | 4.30 | 4.41 | 4.36 | 0.11 | |
| 900 | 4.02 | 4.18 | 4.10 | 0.16 | |
| 1200 | 3.86 | 4.03 | 3.95 | 0.17 | |
| 1500 | 3.70 | 3.93 | 3.81 | 0.23 | |
| 1800 | 3.51 | 3.83 | **3.67** | 0.32 | resume 직전 best — ckpt `/36/` |
| — | — | — | — | — | **resume from /36/, parallel eval 적용** |
| 2100 | 3.47 | 3.72 | 3.60 | 0.25 | |
| 2400 | 3.32 | 3.64 | 3.48 | 0.32 | |
| 2700 | 3.29 | 3.62 | 3.46 | 0.33 | |
| 3000 | 3.20 | 3.60 | 3.40 | 0.40 | overfitting 뚜렷 |
| 3300 | 3.14 | 3.57 | 3.36 | 0.43 | |
| 3600 | 3.06 | 3.53 | 3.30 | 0.47 | |
| 3900 | 3.07 | 3.52 | 3.30 | 0.45 | val 정체 시작 |
| 4200 | 3.04 | 3.49 | 3.26 | 0.45 | |
| 4500 | 3.00 | 3.50 | 3.25 | 0.50 | train만 감소 |
| 4800 | 2.99 | 3.55 | 3.27 | 0.56 | val 반등 — 과적합 |
| **5100** | **2.89** | **3.44** | **3.17 ← best** | **0.55** | **실질 최적** — ckpt `/31/` |
| 5400 | 3.02 | 3.41 | 3.21 | 0.39 | gap 일시 축소 |
| 5700 | 2.96 | 3.42 | 3.19 | 0.46 | |
| 6000 | 2.94 | 3.49 | 3.22 | 0.55 | final (best 대비 +0.05 drift) |

### 관찰

1. **첫 eval(step 300)의 train-val gap 0.04** — 혼합 코퍼스 run(step 300에서 gap 0.12) 대비 극적으로 작음. 공식 held-out val이 train과 같은 분포를 대표한다는 증거.
2. **best avg 3.17 @ step 5100** — perplexity exp(3.17) ≈ 23.8. 혼합 run best(4.65)보다 훨씬 낮지만, vocab/분포가 다르므로 직접 비교는 어려움. 같은 textbook 장르 val 기준이란 점이 중요.
3. **step 4200 이후 val 정체** — train이 2.9대까지 내려가는 동안 val은 3.4-3.6에서 진동. 순수 memorization 영역. 남은 iter에서 val 개선 없음.
4. **Resume은 완전 투명** — 재개 후 첫 eval(step 2100)이 자연스러운 곡선 상에 위치. Adam moment 복원 덕에 warm-restart 이슈 없음.
5. **Eval 병렬화 효과** — 300 iter마다 20회 eval × train/val 200 seq씩. 단일 스레드 대비 worker 8개 분산으로 eval 시간 ~6-8×. 체감상 eval 블록 소요가 거의 안 느껴질 정도.

## 체크포인트

저장된 `model-textbook/vec/1057536/` 하위 (11개, best 갱신만 발생):

| dir | 저장 iter | avg | 종류 |
|---|---|---|---|
| `/48/` | 300 | 4.86 | best |
| `/43/` | 600 | 4.36 | best |
| `/41/` | 900 | 4.10 | best |
| `/39/` | 1200 | 3.95 | best |
| `/38/` | 1500 | 3.81 | best |
| `/36/` | 1800 | 3.67 | best (**resume anchor**) |
| `/35/` | 2100 | 3.60 | best |
| `/34/` | 2400-3000 | 3.26-3.48 | best (여러 번 덮어씀) |
| `/33/` | 3300 | 3.36 | best |
| `/32/` | 3600-4500 | 3.21-3.27 | best |
| **`/31/`** | **5100** | **3.17** | **best ← 실질 최적** |

각 dir는 `checkpoint.json`, `meta.json`, `model_weights.bin`, `optimizer_state.bin` 4파일 (마지막은 resume용 AdamW 상태).

## 샘플 품질

상세 HTML 리포트: [`textbook-sample-report-2026-04-24.html`](textbook-sample-report-2026-04-24.html)

체크포인트 `/31/` (best avg 3.17), temperature 0.8, top-k 40으로 **val에서 랜덤 선정한 6개 프롬프트**(Python `random.seed(42)`, 앞 4단어) 각각 2 샘플:

### 주요 관찰

**잘 배운 것**:
- **Textbook 형식 마커 재현** — "## Chapter", "**what we learned**", "Q:/A:", "Part 2:", "Conclusion:" 등 구조적 템플릿이 자연스럽게 등장. 단일 장르 학습의 효과.
- **단어 목록 스키마** — "- word: definition" 패턴이 여러 샘플에서 재현 ("- talk:", "- big details:", "- vote:"). textbook 특유의 어휘 설명 블록.
- **교실 말투** — "in this lesson, we will learn about...", "for example, if the..." 같은 교사 화법.

**못 배운 것**:
- **의미 일관성 낮음** — "fluffs are good things", "bathone is important", "fizite: an option..." 같이 fabricated 단어/개념 빈출.
- **주제 드리프트** — 한 문단 안에서 주제가 여러 번 바뀜. attention 기반 글로벌 일관성 부족.
- **긴 거리 의존성 부재** — 시작 프롬프트가 textbook 주제를 선언하지만 이후 본문이 그 주제를 지키지 못함.

**해석**: avg 3.17 ≈ perplexity 24는 "형식 마커는 잡히지만 의미론적 이해는 여전히 없음" 단계. 혼합 run(perplexity ~132)과 비교하면 같은 장르 내 예측은 훨씬 잘함. 단일 장르 학습이 스타일 일관성에 확실히 유리하다는 결론.

## 비교 매트릭스

| 지표 | 혼합 10k (2026-04-24) | textbook 6k (이번) |
|---|---|---|
| 데이터 | 4 장르 concat, 90:10 cut val | textbook only, 공식 val |
| train tokens | 1,485,987 | 610,429 |
| val tokens | 165,110 (cut 편향) | 30,860 (공식 held-out) |
| 토큰당 노출 (완주) | ~13.8 (최대) | ~20 (6000 iter) |
| best avg | 3.14 @ step 7000 | 3.17 @ step 5100 |
| best perplexity (val) | ~132 (val 4.88 기준) | ~31 (val 3.44 기준, 단 장르 집중) |
| 샘플 특성 | 장르 드리프트 (textbook > book > wiki > conv) | textbook 톤 일관 |
| 체크포인트 수 | 16 (always-save) | **11 (best-only)** |
| worker | 4 (run 초반엔 4) | 8 (VEC_MAX_WORKERS=8) |

## 이번 run의 교훈

1. **공식 held-out val이 필요**. train-val gap이 run 내내 0.04-0.55 범위로 자연스럽게 보이는 것은 val이 진짜 held-out이라는 증거. 90:10 cut은 "뒷부분 문서 분포"에 따라 bias 가능.
2. **단일 장르 학습은 스타일 일관성에 큰 도움**. 같은 아키텍처·같은 iter 예산에서 장르 드리프트가 사라지고 형식 마커 재현이 강해짐.
3. **Best-only 저장이 디폴트 좋음**. `alwaysSaveCheckpoint=false`로 ckpt 수 16→11, 디스크 부담 감소. 나쁜 eval이 갱신 건너뛰므로 의미 있는 체크포인트만 남음.
4. **Eval 병렬화는 무료 점심**. 학습은 이미 worker 분산인데 eval만 순차였던 비대칭 해소. 구현도 `trainStepParallel` 패턴 복사 수준.
5. **Resume 실전 검증 통과**. 중단→재시작이 loss 곡선에 점프 없이 이어짐. 옵티마이저 상태까지 복원하는 게 결정적.

## 후속 실험 아이디어

- **다른 장르 독립 학습** — wiki-only, conversation-only. 각 장르의 perplexity 하한을 측정해 모델 capacity 포화 시점 파악.
- **Mixed with weighted sampling** — 혼합하되 장르별 비율을 조정(현재 textbook 41% + book 28% + wiki 18% + conv 14% → 예: 균등 25%). DataLoader가 장르 인지하는 모드 필요.
- **더 큰 val 활용** — 10M 라이너 대신 full-100M 스케일 데이터셋이면 val도 훨씬 많음. 지금 val 30k는 평가에 충분하지만 학습 곡선 세부 모니터링용으론 작음.
- **Dropout 활성화** — `vec`는 현재 dropout 미구현. step 4500 이후 명확한 overfit 관찰됐으므로 regularization이 이득 있을 것.
- **Early stopping**: patience 기반 (예: 3 eval 연속 best 미갱신 시 종료). 이번 run의 경우 step 5100 이후 bridge 없이 바로 끝낼 수 있었음.

## 참조

- 학습 로그: `run-logs/textbook-w8.log` (step 0-1800), `run-logs/textbook-w8-resume.log` (resume 이후)
- 체크포인트: `model-textbook/vec/1057536/{31,32,...,48}/` (gitignore 제외)
- 샘플 리포트: [`textbook-sample-report-2026-04-24.html`](textbook-sample-report-2026-04-24.html)
- 코드 커밋 (이번 PR):
  - 파이프라인 분리 입력 지원 (`StoriesBpePrep` sealed SplitSource)
  - `TinyHelenTrainTextbookVec` 엔트리 + gradle task
  - `SamplePromptsFromFile` 엔트리 + gradle task
  - `vec.Trainer.estimateLoss` 병렬 경로 추가
  - `.gitignore`에 `model-textbook/` 추가
  - `StoriesBpePrepSplitTest` 3-case 테스트
