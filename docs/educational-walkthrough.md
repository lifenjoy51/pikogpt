# Scalar 백엔드 educational walkthrough

PikoGPT 스칼라 백엔드는 GPT를 처음부터 손으로 만들면서 autodiff와 transformer를 이해하기 위한 교육용 코드입니다. 실제 학습 실험은 turbo 백엔드(`turbo/`)에서 하고, 스칼라는 "각 수식이 코드로 어떻게 그대로 쓰여 있는지"를 읽기 위한 곳입니다.

이 문서는 **읽는 순서**와 **각 파일에서 무엇을 가져가야 할지**를 안내합니다. 각 단계 끝에 다음 단계로 넘어가기 전에 스스로에게 던질 점검 질문을 두었습니다.

---

## 0. 전제

- Kotlin 기본 문법 (operator overloading, data class, lambda).
- 기본 미분 — chain rule이 무엇인지.
- 기본 신경망 — perceptron, MLP, gradient descent.

PyTorch 경험은 도움이 되지만 필수는 아닙니다.

---

## 1. `Value.kt` — autodiff 코어 (가장 중요)

**파일**: `src/main/kotlin/Value.kt`

PyTorch의 `torch.Tensor`가 텐서 단위로 그래프를 만드는 것과 달리, 여기서는 **하나의 스칼라마다 `Value` 객체 하나**가 만들어집니다. Karpathy의 micrograd 직계 포팅.

핵심 메커니즘:
- `+`, `*`, `pow`, `relu`, `exp`, `sigmoid`, `gelu` 같은 연산을 호출하면 결과 `Value`가 생기고, 그 안에 **부모 노드 집합**과 **backward 클로저**가 저장됨 (`_parentNodes`, `backwardFunction`).
- `backward()`가 호출되면 출력 노드부터 DFS로 위상정렬한 다음, 역순으로 각 `backwardFunction`을 호출 → chain rule로 모든 부모의 `gradient`에 기여분이 누적.

각 연산자 메서드 위에는 표준 헤더가 있습니다:
```kotlin
/**
 * forward:  out = a * b
 * backward: ∂out/∂a = b, ∂out/∂b = a
 */
```
이 두 줄이 곧 그 연산의 미분 정의.

**점검 질문**:
1. `c = a + b`로 그래프를 만들고 `c.backward()`를 부르면 `a.gradient`, `b.gradient`는 각각 얼마가 되어야 하는지 손으로 풀어보고, 코드의 `plus` backward 클로저가 같은 결과를 주는지 확인.
2. `GradContext.noGrad { ... }` 블록은 왜 필요한가? (힌트: 추론 시 그래프 빌드 안 함)
3. `backward(clearGraph = true)`는 언제 의미가 있는가? (힌트: 학습 루프의 메모리)

---

## 2. `grad/Micrograd*.kt` — 가장 단순한 학습 예제

**파일들**:
- `src/main/kotlin/grad/MicrogradNeuron.kt`
- `src/main/kotlin/grad/MicrogradLayer.kt`
- `src/main/kotlin/grad/MicrogradMLP.kt`
- `src/main/kotlin/grad/MicrogradLossCalculator.kt`

`Value`만으로 MLP가 어떻게 만들어지는지 보여주는 micrograd 직계 포팅. Layer는 Neuron의 리스트, MLP는 Layer의 스택. 모든 연산이 `Value` 위에서 일어나므로 `Value.backward()`만 부르면 모든 weight의 gradient가 채워짐.

**점검 질문**:
1. Neuron 한 개의 forward 식은 `output = relu(Σ w_i * x_i + b)` 입니다. 이 식의 모든 sub-식이 `Value` 연산자라는 것을 코드에서 확인.
2. `MicrogradLayer.parameters()`가 모든 neuron의 모든 weight + bias를 평면 리스트로 반환하는 이유는? (힌트: optimizer가 단일 루프로 update)

---

## 3. `MlpTest.kt` / `LossTest.kt` — 학습 사이클의 4단계

**파일들**:
- `src/test/kotlin/grad/MlpTest.kt::trainExample`
- `src/test/kotlin/grad/LossTest.kt::test1, test2`

테스트 코드 자체가 튜토리얼입니다. 한 epoch에서 다음 4단계가 반복됩니다:

```
Step 1 — forward pass:        예측 생성
Step 2 — loss computation:    예측과 정답의 차이로 loss 노드 생성
Step 3 — backward propagation: chain rule로 모든 파라미터의 gradient 계산
Step 4 — parameter update:     gradient 반대 방향으로 한 step (SGD)
```

`zeroGrad`는 매 step 시작에 이전 step의 gradient를 청소하기 위함 — backward는 항상 누적이므로.

**점검 질문**:
1. `zeroGrad`를 빠뜨리면 어떻게 되는가? (힌트: gradient가 epoch마다 누적되어 학습이 망가짐)
2. `LossTest::test1`의 learning rate decay (`1.0 → 0.1`)는 무엇을 의도하는가? (힌트: coarse-to-fine)
3. test1의 full-batch와 test2의 mini-batch의 차이는 학습 신호에 어떤 영향을 주는가?

---

## 4. `gpt/Matrix.kt` — 2D Value 추상화

**파일**: `src/main/kotlin/gpt/Matrix.kt`

GPT부터는 토큰 시퀀스가 들어가므로 2D 배열이 필요합니다. `Matrix`는 `Array<Array<Value>>` 위의 얇은 wrapper:
- `mapRows`: 각 행에 함수 적용 (e.g. 토큰별 LayerNorm)
- `zipWith`: 두 행렬 원소별 결합 (e.g. 잔여 연결 `x + attn(x)`)
- `softmaxRows` / `argMaxRows` / `lastRow`: 의미적 helper

추상화는 일부러 얇게. 학습자가 필요하면 raw `Array<Array<Value>>`로 쉽게 옮겨갈 수 있도록.

**점검 질문**:
1. `mapRows`와 `zipWith`의 결과는 새 `Matrix`인가? `data` 참조를 공유하는가? (힌트: `map { ... }.toTypedArray()`로 새 배열 생성)
2. `softmaxRows`의 max-trick은 왜 필요한가? (힌트: exp(big number) overflow)

---

## 5. `gpt/ScalarLinear.kt`, `ScalarEmbeddingTable.kt`, `ScalarLayerNorm.kt`, `ScalarDropout.kt` — building blocks

**파일들**: `src/main/kotlin/gpt/Scalar{Linear,EmbeddingTable,LayerNorm,Dropout}.kt`

- **`ScalarLinear`**: `output[i] = Σ_j w[i][j] * input[j] + b[i]`. forward 식 그대로.
- **`ScalarEmbeddingTable`**: 학습 가능한 lookup table. `lookup(indices)`가 토큰 ID 배열을 받아 [N, embed_dim] 행렬로 만듦.
- **`ScalarLayerNorm`**: 토큰 단위로 평균을 빼고 분산으로 나눠 정규화. `gain`/`bias`로 학습된 affine 변환.
- **`ScalarDropout`**: 학습 시 랜덤한 일부 뉴런을 0으로, 나머지를 `1/(1-p)`로 스케일.

각 클래스의 `parameters()`가 학습 가능한 모든 `Value`를 반환 — optimizer가 모를 수 있게.

**점검 질문**:
1. `ScalarEmbeddingTable`이 `Matrix`를 상속한다는 사실의 의미는? 행이 곧 vocab의 한 토큰이라는 해석.
2. LayerNorm의 affine 부분(`gain`/`bias`)이 없으면 학습이 어떻게 달라지는가?

---

## 6. `gpt/ScalarFeedForward.kt` — Transformer FFN

**파일**: `src/main/kotlin/gpt/ScalarFeedForward.kt`

Transformer block의 두 번째 sub-layer. 4단계:
1. 확장: `embed_dim → 4 * embed_dim`
2. GELU 비선형
3. 수축: `4 * embed_dim → embed_dim`
4. dropout

4× expansion은 GPT-2 표준. "더 많은 표현력"을 한 번 가졌다가 다시 좁힌다.

**점검 질문**:
1. 만약 GELU 대신 ReLU를 쓰면 무엇이 달라지는가?
2. expansion ratio를 4x가 아니라 2x로 줄이면 파라미터 수는 어떻게 변하는가?

---

## 7. `gpt/ScalarCausalSelfAttention.kt` — Self-Attention의 5단계

**파일**: `src/main/kotlin/gpt/ScalarCausalSelfAttention.kt`

이 파일은 의도적으로 5개 helper로 분해되어 있습니다:

```
Step 1 — projectQkv:           Q, K, V 사영
Step 2 — attentionScores:      Q·K^T / √d_k + causal mask
Step 3 — softmaxRows (Matrix): row-wise softmax
Step 4 — weightedSum:          weights · V
Step 5 — outputAndDropout:     출력 사영 + dropout
```

`forward()` 본문은 5줄입니다:
```kotlin
fun forward(input: Matrix): Matrix {
    val (queries, keys, values) = projectQkv(input)
    val scores = attentionScores(queries, keys)
    val weights = scores.softmaxRows()
    val context = weightedSum(weights, values)
    return outputAndDropout(context)
}
```

각 helper에 forward 수식이 docstring으로 있어 코드와 식이 1:1입니다. causal mask는 미래 위치에 `Value.MIN`(매우 큰 음수)을 넣어 softmax 후 사실상 0이 되도록.

**점검 질문**:
1. `√d_k` 스케일링이 없으면 softmax는 어떻게 되는가? (힌트: dot product 분산이 d_k에 비례 → 한 점에 거의 다 몰림)
2. causal mask가 없으면 자기회귀 학습에 어떤 leakage가 발생하는가?
3. `weightedSum`이 사실상 "모든 토큰의 V 벡터에 대한 가중 평균"인 이유는?

---

## 8. `gpt/ScalarTransformerBlock.kt` — Pre-LN block

**파일**: `src/main/kotlin/gpt/ScalarTransformerBlock.kt`

```
h1 = x  + attn(ln1(x))
y  = h1 + mlp(ln2(h1))
```

각 sub-layer 앞에 LayerNorm을 두는 Pre-Norm 구조. 잔여 연결 + Pre-Norm이 깊은 stack에서도 학습 안정성을 보장.

**점검 질문**:
1. Pre-Norm vs Post-Norm: 잔여 연결 안에 LN이 들어가는가, 밖에 있는가?
2. 만약 잔여 연결이 없다면 깊은 stack에서 무엇이 깨지는가? (힌트: vanishing gradient)

---

## 9. `gpt/ScalarPikoGPT.kt` — full model

**파일**: `src/main/kotlin/gpt/ScalarPikoGPT.kt`

```
tokenEmbedding(ids) + positionEmbedding(positions)
    → (TransformerBlock) × N
    → finalLayerNorm
    → lmHead → logits
```

`forward(tokenIds: IntArray): Matrix`가 [seqLen, vocabSize] logits를 반환. 자기회귀 학습은 이 logits의 각 위치를 "다음 토큰"으로 학습.

**점검 질문**:
1. position embedding이 없으면 어떻게 되는가? (힌트: attention은 위치 불변 → 순서를 잃음)
2. `lmHead`의 출력 차원이 `vocabSize`인 이유는?

---

## 10. `train/ScalarAdamW.kt`, `train/ScalarTrainer.kt` — 학습 메커니즘

**파일들**: `src/main/kotlin/train/Scalar{AdamW,Trainer}.kt`

- **`ScalarAdamW`**: Adam + weight decay. 각 파라미터마다 1차/2차 모멘트를 저장하고 bias correction 적용. micrograd의 단순 SGD와 비교해보면 차이가 명확.
- **`ScalarTrainer`**: LR schedule, gradient clipping, 평가, 체크포인트. 매 step의 `backward(clearGraph = true)`로 메모리 압력을 관리.

**점검 질문**:
1. AdamW가 SGD보다 잘 작동하는 직관적 이유는? (힌트: 파라미터별 적응형 lr + weight decay)
2. `clearGraph = true`를 빠뜨리면 무엇이 누적되는가? (힌트: 그래프 노드의 부모 참조 + 클로저)

---

## 11. 직접 돌려보기 — quickstart

이 문서는 **코드 읽기** 가이드입니다. **실제로 학습→샘플링을 한 번 돌려보고** 싶다면
[scalar-quickstart.md](scalar-quickstart.md)를 참고하세요.

요약: `runAlphabetPrep` → `runMiniTrainer` (~10분) → `runSampler`. 알파벳 a-z 텍스트로
가장 작은 모델을 학습해 음절·단어 패턴을 눈으로 확인할 수 있습니다.

---

## 다음 단계 — 실제 학습 실험

스칼라는 71K 파라미터 모델로도 iter당 ~16초가 걸려 실험에는 너무 느립니다. 실험 단계로 넘어가려면:

- **`turbo/`** 백엔드를 읽어보세요. 같은 transformer 식이 SIMD + KV cache로 ~1000× 빠르게 구현되어 있습니다. RMSNorm/SwiGLU/RoPE/GQA/qk-norm/fused QKV/z-loss 같은 modern variants도 포함.
- 실제 학습은 `./gradlew runTinyHelenTrainTurbo` 같은 터보 진입점으로.

스칼라에서 익힌 식과 코드의 1:1 대응이 turbo 백엔드를 읽을 때도 그대로 적용됩니다.
