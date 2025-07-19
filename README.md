# PikoGPT: Kotlin으로 구현된 미니 GPT 모델

이 프로젝트는 학습 목적으로 [nanoGPT](https://github.com/karpathy/nanoGPT)와 [micrograd](https://github.com/karpathy/micrograd)를 Python에서 Kotlin으로 변환한 것입니다.

## 원본 프로젝트

이 프로젝트는 Andrej Karpathy의 다음 원본 작업을 기반으로 합니다.

*   **nanoGPT:** https://github.com/karpathy/nanoGPT
*   **micrograd:** https://github.com/karpathy/micrograd

⚠️ 이 프로젝트는 주로 **Claude Code**와 **Gemini**에 의해 작업되었습니다. 사람이 직접 작성한 코드나 텍스트의 양은 매우 적습니다.


PikoGPT는 Kotlin으로 구현된 경량화된 GPT(Generative Pre-trained Transformer) 모델입니다. 이 프로젝트는 Transformer 아키텍처의 핵심 개념을 이해하고, 자동 미분(autograd) 엔진을 직접 구현하여 신경망 훈련 과정을 투명하게 보여주는 것을 목표로 합니다. 텍스트 생성, 훈련, 데이터 전처리 기능을 포함하며, 교육 및 연구 목적으로 활용될 수 있습니다.

## 주요 기능

*   **경량 GPT 모델:** Transformer 블록, Multi-Head Self-Attention, MLP, Layer Normalization 등 GPT의 핵심 구성 요소를 Kotlin으로 직접 구현했습니다.
*   **자동 미분 엔진:** `Value` 클래스를 통해 순전파 중 계산 그래프를 동적으로 구축하고, 역전파 시 연쇄 법칙을 적용하여 그래디언트를 자동으로 계산합니다.
*   **AdamW 옵티마이저:** 최신 딥러닝 모델 훈련에 널리 사용되는 AdamW 옵티마이저를 구현하여 효율적인 파라미터 업데이트를 지원합니다.
*   **BPE 토크나이저:** Byte Pair Encoding (BPE) 알고리즘을 사용하여 텍스트 데이터를 효율적인 토큰 시퀀스로 변환합니다.
*   **데이터 파이프라인:** 텍스트 데이터 로딩, 훈련/검증 데이터 분할, 바이너리 형식 저장 등 전체 데이터 전처리 과정을 관리합니다.
*   **체크포인트 시스템:** 훈련 중 모델의 상태(가중치, 옵티마이저 상태)를 저장하고 로드하여 훈련을 재개하거나 사전 훈련된 모델을 활용할 수 있습니다.
*   **텍스트 생성 및 품질 검증:** 학습된 모델을 사용하여 새로운 텍스트를 생성하고, 생성된 텍스트가 특정 품질 기준(예: 아동용 이야기 적합성)을 만족하는지 외부 LLM(LM Studio)을 통해 검증하는 기능을 포함합니다.

## 아키텍처 개요

PikoGPT는 다음과 같은 주요 모듈로 구성됩니다.

### 1. GPT 모델 (`gpt` 패키지)

*   **`PikoGPT.kt`**: 전체 GPT 모델의 메인 클래스입니다. 토큰 임베딩, 위치 임베딩, 여러 개의 `TransformerBlock` 및 최종 언어 모델 헤드(`lmHead`)를 포함합니다.
*   **`TransformerBlock.kt`**: GPT 모델의 핵심 빌딩 블록입니다. `SimpleSelfAttention`과 `MLP` 레이어를 포함하며, 잔여 연결(Residual Connection)과 Layer Normalization을 적용합니다.
*   **`SimpleSelfAttention.kt`**: Multi-Head Self-Attention 메커니즘을 단순화하여 구현합니다. Query, Key, Value 프로젝션 및 인과 마스킹(Causal Masking)을 처리합니다.
*   **`Linear.kt`**: 신경망의 기본 선형 변환(완전 연결) 레이어입니다. 가중치와 편향을 관리하고 순전파를 수행합니다.
*   **`MLP.kt`**: Transformer 블록 내의 Feed-Forward Network를 구현합니다. 확장-GELU 활성화-수축 구조를 가집니다.
*   **`LayerNorm.kt`**: Layer Normalization을 구현하여 신경망 훈련의 안정성을 높입니다.
*   **`Dropout.kt`**: 과적합 방지를 위한 Dropout 정규화 기법을 구현합니다.

### 2. 훈련 (`train` 패키지)

*   **`Trainer.kt`**: 모델 훈련의 전체 과정을 관리하는 메인 클래스입니다. 학습률 스케줄링, 그래디언트 클리핑, 평가 및 체크포인트 저장 로직을 포함합니다.
*   **`AdamW.kt`**: AdamW 옵티마이저를 구현하여 모델 파라미터를 효율적으로 업데이트합니다.
*   **`DataLoader.kt`**: 훈련 및 검증 데이터를 로드하고 미니배치를 생성합니다.
*   **`TrainConfig.kt`**: 훈련 과정에 필요한 모든 하이퍼파라미터(배치 크기, 학습률, 모델 크기 등)를 정의합니다.
*   **`Checkpoint.kt`**: 모델의 가중치, 옵티마이저 상태, 훈련 설정 등을 저장하고 로드하기 위한 데이터 구조입니다.
*   **`States.kt`**: 모델의 다양한 레이어(Attention, Block, FeedForward, Linear, LayerNorm)의 상태를 직렬화 가능한 형태로 정의합니다.

### 3. 데이터 처리 (`data` 패키지)

*   **`SimpleBPE.kt`**: 텍스트를 토큰화하는 BPE(Byte Pair Encoding) 알고리즘의 간단한 구현체입니다. 훈련 및 인코딩 기능을 제공합니다.
*   **`StoriesBpePrep.kt`**: `SimpleBPE`를 사용하여 원본 텍스트 데이터를 전처리하고, 훈련 및 검증을 위한 바이너리 파일(`train.bin`, `val.bin`)과 어휘 사전(`meta.json`)을 생성합니다.
*   **`MetaInfo.kt`**: 어휘 사전의 메타데이터(어휘 크기, 토큰-ID 매핑)를 저장하는 데이터 클래스입니다.
*   **`StoryGenerator.kt`**: 외부 LLM(LM Studio)과 연동하여 새로운 이야기를 생성하고, 생성된 이야기의 품질을 검증하는 기능을 제공합니다.

### 4. 핵심 유틸리티

*   **`Value.kt`**: 자동 미분 엔진의 핵심 클래스입니다. 모든 스칼라 값에 대한 연산을 오버로딩하여 계산 그래프를 구축하고 역전파를 통해 그래디언트를 계산합니다.
*   **`RandomGaussian.kt`**: 표준 정규 분포를 따르는 난수를 생성하는 유틸리티입니다.
*   **`Funtions.kt`**: `sumOf`와 같은 유용한 확장 함수들을 포함합니다.

## 시작하기

### 빌드 및 실행

이 프로젝트는 Gradle을 사용하여 빌드됩니다.

```bash
# 프로젝트 빌드
./gradlew build
```

### 실행 흐름

PikoGPT 프로젝트의 일반적인 실행 순서는 다음과 같습니다.


0.  **이야기 생성 (`StoryGenerator.kt`)**
    *   (선택 사항) 외부 LLM(LM Studio)과 연동하여 어린이를 위한 이야기를 생성하고, 생성된 이야기의 품질을 검증합니다.
    *   `data/StoryGenerator.kt` 클래스가 LM Studio API 호출 및 응답 처리를 담당합니다.
    *   이 기능은 `StoryGenerator` 클래스의 `generateStory()` 및 `validateStoryQuality()` 메서드를 통해 사용됩니다.
    * 
1.  **데이터 준비 (`StoriesBpePrep.kt`)**
    *   원시 텍스트 데이터를 BPE 토큰화하고, 훈련 및 검증에 필요한 바이너리 파일(`train.bin`, `val.bin`)과 어휘 사전(`meta.json`)을 생성합니다.
    *   이 과정은 `data/StoriesBpePrep.kt` 파일의 `main` 함수를 통해 실행됩니다.
    ```bash
    ./gradlew run --args="prepare_data data/1k"
    ```

2.  **모델 훈련 (`Trainer.kt`)**
    *   준비된 데이터를 사용하여 GPT 모델을 훈련합니다.
    *   `train/Trainer.kt` 클래스가 훈련 루프, 옵티마이저(`AdamW.kt`), 데이터 로더(`DataLoader.kt`), 체크포인트 저장(`Checkpoint.kt`) 등을 관리합니다.
    *   훈련 설정은 `train/TrainConfig.kt`에서 정의됩니다.
    *   훈련 실행 시 `Trainer` 클래스의 `train()` 메서드가 호출됩니다.
    ```bash
    ./gradlew run --args="train"
    ```

3.  **텍스트 생성 (`Sampler.kt`)**
    *   훈련된 모델을 로드하여 새로운 텍스트를 생성합니다.
    *   `sample/Sampler.kt` 클래스가 모델 로드, 토큰화 설정, 샘플링 전략(온도, Top-K) 적용 및 텍스트 생성을 담당합니다.
    *   `Sampler` 클래스의 `generateText()` 또는 `sample()` 메서드를 통해 텍스트 생성을 시작할 수 있습니다.
    ```bash
    ./gradlew run --args="sample"
    ```

### 데이터 준비

훈련을 위해서는 `data` 디렉토리에 텍스트 파일이 필요합니다. `StoriesBpePrep`를 사용하여 텍스트 파일을 BPE 토큰화하고 훈련 가능한 형식으로 변환할 수 있습니다.

### LM Studio 연동 (StoryGenerator 사용 시)

`StoryGenerator`를 사용하려면 로컬에 LM Studio가 설치되어 있고, `google/gemma-3-1b` 모델이 로드되어 `http://127.0.0.1:1234`에서 API 서버가 실행 중이어야 합니다.

## 기여

이 프로젝트는 교육 및 연구 목적으로 시작되었습니다. 기여를 환영합니다.

## 라이선스

MIT License