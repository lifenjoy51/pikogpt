# 대화 LM 실험 시리즈 — conv-mix → 773k tied (2026-04-25 ~ 04-26)

## 목적

벡터 백엔드 1M 클래스 모델을 **자연스러운 대화 응답**까지 올리는 일련의 실험. 시작점은 v3 1M (val 3.76, 0.83M 토큰)로 데이터 부족·미세한 표현력 부재 상태. 한 단계씩 변경하며 perplexity와 샘플 품질 모두 개선.

## 실험 시리즈 (chronological)

| run | data | tok | model | iter | best avg | val | perplexity | 주요 변경 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v3 1M | tinyhelen-conv | 0.83M | 1.06M | 12000 | 3.30 | 3.76 | 43 | 베이스라인 (heavy overfit) |
| conv-mix | TH+age-5 quote+speaker strip | 8.97M | 432k | 8000 | 3.16 | 3.19 | 24 | Chinchilla 정렬 (×11 데이터) |
| conv-mix-turn | + `<|turn|>` 토큰 | 9.5M | 432k | 8000 | 2.99 | 2.95 | 19.1 | turn boundary 명시 |
| conv-mix-turn-noq | + outer 따옴표 제거 | 9.5M | 432k | 8000 | 2.99 | 2.96 | 19.3 | 베이스 |
| (b128) | block 64→128 | 9.5M | 436k | 4000+resume | 3.04 | 3.00 | 20.1 | long context 효과 미미 |
| a510 432k | + age-10 추가 (untied) | 18.9M | 432k | 16000 | 3.08 | 3.06 | 21.3 | 데이터 2× — 모델 부족 신호 |
| **a510 773k tied** | a510 데이터 | 18.9M | **773k** | 12000 | 2.89 | 2.89 | 18.0 | 모델 1.8× + tied |
| (a510 773k b128) | block 128 | 18.9M | 779k | 8000(중단) | — | — | — | 큰 모델 + 긴 context (실험 중단) |
| clean-a510 773k tied | + emphasis 마커 정리 | 18.9M | **773k** | 12000 | 2.87 | 2.87 | 17.6 | GELU 베이스 |
| **clean-a510 773k SwiGLU** | + Llama 스타일 SwiGLU MLP | 18.9M | **774k** | 12000 | **2.81** | **2.81** | **16.6** | **베스트** (params ≈ 동일) |

## 핵심 인사이트

### 1. Chinchilla 정렬이 1순위
- v3 1M(0.79× tok/param) → conv-mix 432k(20.7× tok/param)에서 **모델 1/3 + 데이터 11×만으로 perplexity 43→24** (1.8배 개선)
- gap 1.5(memorize) → ~0(generalize) 극적 변화
- **모델 키우기는 데이터 비율 맞춘 후에만 의미**

### 2. 구조적 토큰이 중요
- `<|turn|>` 추가만으로 val 3.19 → 2.96 (perplexity 24 → 19)
- 모델이 turn-taking 패턴 명시적으로 학습
- 따옴표 제거(noq)로 BPE merge 깨끗·출력 더 깔끔

### 3. blockSize 64로 충분
- 대화 turn 평균 25 tokens, 13 turns/doc
- block 64→128 효과 미미 (작은 모델), 큰 모델에서도 marginal
- O(block²) 비용 대비 ROI 낮음

### 4. 모델 크기 vs 데이터 — 균형이 정답
- 데이터 18M에 432k 모델 → capacity 부족 (a510 432k val 3.06)
- 같은 데이터에 773k tied → val 2.89 (Chinchilla 24×, 정확)
- Tied weights = vocab×dim 절약 + 양방향 grad signal (Press & Wolf 2017)

### 5. 데이터 정제 영향
- 본문 안 emphasis (`**Child**` 콜론 없음) 3,474개 누락된 채 학습 → 응답에 마커 그대로 출력
- 정규식 보강 (콜론 무관 매칭, emphasis는 안 텍스트만 보존) 후 재학습 → val 2.89 → **2.87** + 출력 깨끗

### 6. SwiGLU MLP — 같은 params로 6% perplexity 개선
- Llama 표준: `MLP(x) = down(SiLU(gate(x)) ⊙ up(x))`. hidden = ⌈8/3 × dim⌉ = 256 (vs GELU 384) — params는 거의 동일 (774k vs 773k)
- val 2.87 → **2.81**, perplexity 17.6 → **16.6**
- 응답 quality도 명확히 향상: multi-clause, 조건/이유 절, listener question, 부모-아이 톤 분리
- **activation 한 줄 변경으로 가장 큰 ROI** — 검증된 modern transformer 권장사항

## 데이터 처리 파이프라인 (clean-a510 기준)

```bash
# 1) raw 추출 (TinyHelen + TinyDialogues age 5/10)
jq '...' /tmp/TinyHelen/...                                  # TinyHelen conv (191 docs)
unzip individual_age_data.zip                                  # TinyDialogues age-{5,10}

# 2) 정규식 정제 (Python)
re.sub(r'\*\*[^*]+\*\*:\s*', '<|turn|>', text)                # speaker → turn
re.sub(r'\*\*([^*]+)\*\*', r'\1', text)                        # emphasis → 안 텍스트
re.sub(r'"\s+"', '"<|turn|>"', th_text)                        # TinyHelen 따옴표 사이에 turn
text.strip().strip('"').strip()                                # outer 따옴표 제거 per turn

# 3) BPE prep (12g heap 필요, 30MB+ 데이터)
./gradlew runStoriesBpe --args="data/conv-mix-clean-a510"

# 4) 학습
VEC_MAX_WORKERS=10 ./gradlew runConvMixCleanA510M773TrainVec

# 5) 샘플링 / chat
./gradlew runChatVec --args="model/conv-mix-clean-a510/vec/773376/28 0.7 40" --console=plain
```

## 베스트 모델 사용

체크포인트 force-commit: `model/conv-mix-clean-a510/vec/773376/28/` (3.2 MB)
- `model_weights.bin` — 773,376 floats
- `meta.json` — vocab 1000 + BPE merges + special tokens
- `checkpoint.json` — iter, bestLoss, modelArgs

**ChatVec 사용**:
```bash
./gradlew runChatVec --args="model/conv-mix-clean-a510/vec/773376/28 0.7 40" --console=plain
```

권장 temperature 0.7 (다양성 vs topic relevance 균형).

## 한계 및 다음 단계

**여전한 약점**:
- **fabricated 단어** (`heig`, `homemond` 등 BPE 합성 noise) — vocab 1000 + 모델 표현력 한계
- **topic relevance 부족** — instruction-tuned 데이터 부재로 "응답이 질문에 답해야" 학습 안 됨
- **긴 거리 의존성** — block 64에선 한계, 128도 marginal

**향후 시도**:
1. **Vocab 2000-3000** — fabricated 단어 감소, 단 BPE 학습 시간 1.5×
2. **Instruction-tuned 데이터** — Q→A 페어. 외부 (Alpaca, FLAN) 또는 내부에서 재구성
3. **Top-p (nucleus) sampling** — top-k 대체, fine-grained 다양성
4. **Repetition penalty** — sampler 단, 반복 어구 차단
5. **Few-shot prompting** — REPL 시작 시 예시 대화 1-2개 미리 주입
6. **모델 1-2M로 확장** + 데이터 30M+ (age 2,5,10,15 모두 + DailyDialog)

## 참조

- 코드 커밋: `29aafe7` (실험 시리즈), `a8f2f2b` (773k ckpt 공유), `f33ffa2` (ChatVec quote fix), 이번 (clean ckpt + 정제)
- 학습 로그: `run-logs/clean-a510-773k.log`
- 샘플 비교 (이전): `docs/conv-mix-432k-report-2026-04-25.html`
