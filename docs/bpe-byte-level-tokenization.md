# BPE와 Byte-level Tokenization — GPT-2 트릭의 이해

작성: 2026-05-08

pikogpt 학습 중 GPT-2 BPE의 동작을 파헤쳐본 기록. 핵심 의문은 단순했다 —
**vocab 50,257개로 모든 다국어 문자(한글, 이모지, 한자, 러시아어, ...)를 어떻게 다 커버하는가?**
답을 추적하다 보니 BPE의 byte-level 설계가 UTF-8과 만나 만든 우아한 트릭이 보였다.

---

## 1. 토크나이저가 풀어야 하는 두 가지 문제

LM 입력은 결국 정수 시퀀스. 그래서 텍스트 → 정수 매핑이 필요하다. 두 극단:

### Char-level
- 영문 알파벳 + 구두점 + 숫자 → 100개 정도. 매우 작은 vocab.
- **장점**: 모든 영어 문자 cover, OOV 없음.
- **단점**:
  - sequence 매우 길어짐 ("hello" = 5 token).
  - 모델이 단어 의미를 char level에서 조립해야 함. 학습 부담↑.
  - **다국어** (한글/이모지 등) 들어오면 vocab 폭발. 한글 음절 11,172개 + 한자 수만 자.

### Word-level
- 단어를 통째 token. "hello" = 1 token.
- **장점**: sequence 짧음, 의미 단위 명확.
- **단점**:
  - vocab 폭발 (수십만~수백만).
  - **OOV 심각**: 학습에 없던 단어("xyzword") 인코딩 불가능.
  - typo, 합성어, 신조어 모두 OOV.

→ 둘 다 실용 어려움. 그래서 **BPE (Byte Pair Encoding)** 등장.

---

## 2. BPE — 빈도 기반 sub-word 합성

기본 아이디어:
1. 모든 char를 base vocab에 넣음 (`a`, `b`, ..., `z`, ` `, ...).
2. 학습 코퍼스에서 가장 빈번한 인접 char pair를 찾아 합쳐서 새 token 만듦.
3. vocab 크기 도달까지 반복.

예 (영어 코퍼스):
- 초기: `t`, `h`, `e`, `,`, ` `, ...
- 1번째 merge: ` t` (단어 시작 "t"가 빈번) → 새 token
- 2번째: `he` (영어 흔한 bigram)
- 3번째: `in` (-ing, infinity 등에 자주)
- ...
- N번째: ` the` (너무 흔한 단어)

자주 등장하는 단어/sub-word는 점차 단일 token으로 합쳐짐. 자주 안 나오는 조합은 char 단위로 남음.

**OOV 처리**: 학습에 없던 단어가 와도 char-level fallback이 있어 항상 인코딩 가능.

---

## 3. Char-level BPE의 한계 — 우리 CharBPE 사례

pikogpt의 `CharBPE` (이전 이름 `SimpleBPE`, vocab 2K)로 v4-merged 코퍼스를 학습한 결과, 출력 sample에 다음과 같은 OOV가 빈번:

```
livee, spoit, surp, dace, rereat, mip, toar, cererciderer,
bumercad, gapercess, molemleted, moogal, bilnation,
steohaatics, prombures, teleibiurn
```

이들은 단어가 아니라 **sub-word fragment 모음**. 학습 vocab(2,000)에 들어 있는 sub-word 토큰들이 자연스럽지 않게 결합되어 출력됐다.

근본 원인:
- vocab 2K는 너무 작아서 흔한 단어 일부만 단일 토큰화
- 어떤 단어들은 학습 시 분해된 채로 학습 → 출력 시 다시 분해된 형태로 등장
- 모델이 char-level fragment를 "단어처럼" 조립하지 못함

→ 실용 LLM은 vocab을 더 크게(GPT-2: 50K, Llama 3: 128K) 잡거나, 다른 트릭이 필요.

---

## 4. GPT-2의 천재성 — byte-level BPE

GPT-2(2019)는 char-level이 아닌 **byte-level**에서 BPE를 시작했다.

### 핵심 아이디어
1. **base vocab = 256 byte** (0x00 ~ 0xFF)
2. 어떤 텍스트도 UTF-8로 인코딩하면 **byte 시퀀스** 됨
3. 256개 base token이 **모든 가능한 byte 값**을 cover → **OOV 원천 불가**
4. 자주 등장하는 byte 시퀀스만 BPE merge로 합침 (token 256+)

### 표시 트릭 — bytes_to_unicode

256 byte 중 일부는 비출력 문자(NUL, TAB, SPACE 같은 제어 문자, byte 128~160 등). vocab JSON 파일에 저장하려면 모두 printable Unicode char로 매핑 필요. GPT-2는 다음 함수로 매핑:

```python
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))   # 33~126 (printable ASCII)
        + list(range(ord("¡"), ord("¬")+1))  # 161~172 (Latin-1 일부)
        + list(range(ord("®"), ord("ÿ")+1))  # 174~255 (Latin-1 일부)
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)              # U+0100+로 shift
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
```

요약:
- byte 33~126 (printable ASCII `!`~`~`) → 그대로 char
- byte 161~172, 174~255 (printable Latin-1) → 그대로 char
- 나머지 byte (0~32, 127~160, 173) → U+0100+ Unicode char로 shift

이렇게 매핑된 256 char를 vocab에 넣으면 모든 byte가 표현 가능.

### 실측: token 0~255 분포

```
token id 범위 | 표현하는 byte
0~93         | byte 33~126 (printable ASCII !-~)
94~187       | byte 161~172, 174~255 (Latin-1 supplement)
188~221      | byte 0~32, 127 (control + SPACE!)
222~254      | byte 128~160 (additional control)
255          | byte 173 (soft hyphen)
```

흥미로운 점: SPACE(byte 32)는 token 220. printable ASCII이지만 GPT-2 pre-tokenization
규칙(단어 시작에 space prefix 붙임 — ` t`, ` a` 같은 형태로 BPE merge)으로 **standalone space는 별도 처리**.

---

## 5. UTF-8과의 자연스러운 결합

byte-level이 다국어를 어떻게 커버하는가? 답: **UTF-8 design 자체가 ASCII byte와 multi-byte 시퀀스를 분리**해주기 때문.

### UTF-8 인코딩 규칙

| 문자 byte 길이 | 첫 byte 패턴 | 첫 byte 범위 | 후속 byte 범위 |
|---|---|---|---|
| 1 byte (ASCII) | `0xxxxxxx` | 0~127 | — |
| 2 byte | `110xxxxx` | 192~223 | 128~191 |
| 3 byte | `1110xxxx` | 224~239 | 128~191 |
| 4 byte | `11110xxx` | 240~247 | 128~191 |

핵심:
- byte 0~127 (printable ASCII 포함)은 **오직 1-byte ASCII 문자**에만 등장
- multi-byte 문자(한글/이모지/한자/...)는 **byte 128 이상만 사용**
- 두 영역이 **절대 겹치지 않음**

### 실측

```
'cat'    bytes=[99, 97, 116]                — 모두 ASCII range
'가'      bytes=[234, 176, 128]              — 모두 non-ASCII
'안녕'    bytes=[236, 149, 136, 235, 133, 149]  — 모두 non-ASCII
'🐱'     bytes=[240, 159, 144, 177]          — 모두 non-ASCII
'café'   bytes=[99, 97, 102, 195, 169]       — c,a,f는 ASCII / é는 non-ASCII
'안녕 cat' bytes=[236,149,136,235,133,149,32,99,97,116]  — 섞임. 충돌 없음.
```

→ token 0~93은 ASCII text 처리, token 94~255는 multi-byte 문자 처리. UTF-8 덕에 자연스럽게 분리.

---

## 6. token 256+ — BPE merge 결과

학습 코퍼스(WebText)에서 자주 등장하는 byte 시퀀스가 단일 token으로 합쳐진 것. 처음 20개 (merge 순서 = 빈도 순):

```
256 ' t'    258 'he'   260 're'   262 ' the'  264 ' s'   266 ' w'   268 'en'   270 'it'
257 ' a'    259 'in'   261 'on'   263 'er'    265 'at'   267 ' o'   269 ' c'   271 'is'
                                                                    272 'an'   273 'or'
                                                                    274 'es'   275 ' b'
```

영어에서 `' t'`(단어가 t로 시작), `' a'`(단어 a-), `'he'`, `' the'`가 가장 빈번한 byte pair.

긴 단어/구문은 더 후순위 merge — `' the'`(262), `' to'`(284), `' and'`(290), `' you'`(345). 그래서 영어 텍스트는 token 256~10000 범위 token이 압도적으로 자주 나옴.

비-영어 multi-byte 문자도 학습 코퍼스에 자주 등장하면 BPE merge로 합쳐질 수 있음.
예: GPT-2가 학습한 WebText는 영어 위주라 한글 byte 시퀀스 merge가 거의 없어, 한글 1글자 = 3 token이 됨. (Llama 등 다국어 학습 모델은 한국어 byte도 merge되어 token 수 줄어듦.)

---

## 7. 실측: 다양한 입력의 token 수

| 입력 | UTF-8 byte 수 | GPT-2 token 수 | 비율 |
|---|---|---|---|
| `cat` | 3 | 1 | 0.33 (`cat` 단일 token으로 merge) |
| `café` | 5 | 3 | 0.6 (`c`, `af`, `é`) |
| `가` | 3 | 3 | 1.0 (한글 byte별로 분해) |
| `안녕` | 6 | 6 | 1.0 |
| `🐱` | 4 | 3 | 0.75 (일부 byte pair merge) |
| `привет` | 12 | 7 | 0.58 (러시아어 byte 일부 merge) |

→ 영어가 가장 효율적, 한글/일본어/한자 등 비영어는 byte 단위 분해로 token 수↑.
이게 다국어 LLM이 영어 외 언어에서 context length 손해 보는 이유.

---

## 8. 우리 CharBPE vs GPT-2 BPE 비교

| | 우리 CharBPE | GPT-2 BPE |
|---|---|---|
| 학습 출발 | char 단위 (코퍼스 unique chars) | byte 단위 (256개 fixed) |
| OOV 처리 | char로 fallback. 코퍼스에 없는 char(한글)은 UNK. | byte로 fallback. 모든 입력 인코딩 가능. |
| Vocab 크기 | 작음 (2K) | 큼 (50K) |
| 영어 효율 | char/sub-word 단위 → 단어 평균 ~3-4 token | 단어 단위 → 단어 평균 1-2 token |
| 다국어 지원 | char 학습 안 됐으면 X | byte 단위 보장 |
| 학습 방식 | 코퍼스에서 from scratch | 보통 사전학습 토크나이저 사용 |

**우리 학습 결과 OOV의 진짜 원인**:
- 우리는 vocab 2K + char-level 시작
- 자주 등장하지 않은 단어는 char/sub-word 조각 채로 학습
- 출력 시 그 조각들이 자연스럽게 안 모여 `prombures`, `gapercess` 같은 fragment 모음 발생
- 만약 GPT-2 BPE(50K)로 같은 데이터 학습했다면 `prombures` 같은 fragment 안 나옴 (단어가 단일 토큰)

---

## 9. 시사점

### 작은 모델에 byte-level BPE를 그대로 가져오면?
- vocab 50K → embedding 50K × 96 = 4.8M params (1M 모델이 6M으로 폭발)
- byte-level의 장점은 살아있지만 모델 capacity 균형 깨짐

### TinyStories 방식의 의미
- TinyStories: GPT-Neo BPE(50K) + 데이터 어휘를 제한
- 즉 toolkit은 큰 vocab BPE, 데이터는 좁은 vocab만 사용
- effective vocab 2K, fallback BPE 50K → OOV 거의 없음

### 우리 corpus의 개선 방향
1. **데이터 어휘 좁히기** (TinyStories 방식): proper nouns, foreign words 제거
2. **vocab 적당히 늘림** (4K~8K): trade-off
3. **byte-level fallback 추가**: CharBPE를 byte-level로 재설계

---

## 10. 정리

GPT-2 BPE의 우아함은 세 요소의 결합:

1. **byte-level base vocab**: 256개로 모든 가능한 byte 표현
2. **bytes_to_unicode 매핑**: byte를 printable Unicode로 옮겨 vocab JSON에 안전 저장
3. **UTF-8 design**: ASCII byte와 multi-byte가 절대 겹치지 않아 토큰 영역 자연 분리

결과:
- 영어/숫자 → token 0~~~ (ASCII 영역)
- 한글/이모지/한자 → token 94~~~ + multi-byte sequences
- 어떤 입력도 인코딩 가능, OOV 원천 불가
- BPE merge로 자주 등장하는 byte 시퀀스 자동 압축

우리 CharBPE는 char-level이라 OOV 가능. 다국어 학습이나 robust handling이 필요하면 byte-level로 전환하는 게 현대적 접근.

---

## 참고

- Sennrich et al. (2015), "Neural Machine Translation of Rare Words with Subword Units" — 원조 BPE
- Radford et al. (2019), "Language Models are Unsupervised Multitask Learners" (GPT-2) — byte-level BPE
- Kudo (2018), "Subword Regularization" — SentencePiece (BPE 변형)
- UTF-8 RFC 3629
