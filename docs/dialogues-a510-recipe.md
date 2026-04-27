# dialogues-a510 데이터셋 레시피

`data/{train,val}.txt.gz`(git에 추적되는 정제본)을 처음부터 만들거나 검증할 때 쓰는 절차서. TinyDialogues age-5 + age-10만 사용하는 dialogues-only 코퍼스. 베스트 모델(`model/best/`, val 2.72, conv-mix-clean-a510 기반)의 후속 실험 베이스라인.

## 1. 출처

- 원본 리포: <https://github.com/styfeng/tinydialogues> (EMNLP 2024, MIT 라이선스)
- 데이터: HuggingFace `styfeng/TinyDialogues` — `individual_age_data.zip`을 받아 압축 해제
- 학습용으로는 age-5와 age-10 두 개만 사용 (age-2, age-15 등은 제외)

## 2. 입력 파일 (raw)

압축 해제 후 4개 .txt 파일:

| 파일 | 줄 수 | `<|endoftext|>` 수 |
|---|---:|---:|
| `tinydialogue_age-5_train.txt` | 28,187 | 28,188 |
| `tinydialogue_age-5_val.txt` | 5,061 | 5,061 |
| `tinydialogue_age-10_train.txt` | 23,040 | 23,041 |
| `tinydialogue_age-10_val.txt` | 4,158 | 4,158 |

각 줄 = 한 대화. 한 줄 안에 `**Speaker**: "발화"` 형식이 ` \n\n ` 구분자로 이어지고, 줄 끝(또는 직전)에 `<|endoftext|>` 하나가 conversation 종료를 표시.

raw 발화 형식 예시:

```
**Dad**: "And they all lived happily..." \n\n **Child**: "Yes! I love..." ... <|endoftext|>
```

## 3. 처리 절차

### 3.1 concat (age-5 → age-10 순서)

```bash
cat tinydialogue_age-5_train.txt  tinydialogue_age-10_train.txt > train.raw.txt
cat tinydialogue_age-5_val.txt    tinydialogue_age-10_val.txt   > val.raw.txt
```

train: 51,229 conversations / val: 9,219 conversations.

### 3.2 정규식 정제 (Python)

각 conversation(= 한 줄)마다 다음 변환:

```python
# (a) speaker 마커 → turn 토큰
text = re.sub(r'\*\*[^*]+\*\*:\s*', '<|turn|>', text)

# (b) emphasis 마커 제거 (안 텍스트만 보존)
text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)

# (c) 발화별 outer 따옴표 제거
text = re.sub(r'"([^"]*)"', r'\1', text)

# (d) raw의 종료 마커 → eos
text = text.replace('<|endoftext|>', '').rstrip()

# (e) 첫 turn 토큰 제거 → BOS가 첫 turn boundary 겸함
if text.startswith('<|turn|>'):
    text = text[len('<|turn|>'):]

# (f) BOS / EOS 래핑
text = f'<|bos|>{text}<|eos|>'
```

마지막에 줄바꿈으로 join해서 `train.txt` / `val.txt` 저장.

### 3.3 결과 형식

conversation 단위로 `<|bos|>...<|eos|>` 래핑 + 발화 구분자 `<|turn|>`:

```
<|bos|>발화1<|turn|>발화2<|turn|>...<|turn|>발화N<|eos|>
```

따옴표 / `**Speaker**:` / `<|endoftext|>` / `\n\n` 구분자 모두 제거.

## 4. 검증 카운트

| 항목 | train | val |
|---|---:|---:|
| 줄 수 (= conversation 수) | 51,229 | 9,219 |
| `<|bos|>` 등장 | 51,229 | 9,219 |
| `<|eos|>` 등장 | 51,229 | 9,219 |
| `<|turn|>` 등장 | 651,340 | 117,097 |
| 파일 크기 (평문) | 60,867,198 B | 10,979,472 B |
| 파일 크기 (gzip) | 19,595,565 B | 3,536,832 B |

raw `<|endoftext|>` 합산(train: 28,188 + 23,041 = **51,229**, val: 5,061 + 4,158 = **9,219**)과 정제 결과의 `<|bos|>` 카운트가 정확히 일치 — conversation 단위 1:1 대응 검증됨.

## 5. BPE 토큰화

git에 추적되는 형태는 `data/train.txt.gz` + `data/val.txt.gz` 두 압축본이므로, prep 직전 임시 디렉터리에 풀어서 사용한다.

```bash
mkdir -p data/dialogues-a510
gunzip -kc data/train.txt.gz > data/dialogues-a510/train.txt
gunzip -kc data/val.txt.gz   > data/dialogues-a510/val.txt
./gradlew runStoriesBpe --args="data/dialogues-a510"
```

vocab 1000, special tokens(`<|bos|>`, `<|eos|>`, `<|turn|>`)는 BPE 학습 단계에서 단일 토큰으로 등록됨. 산출물: `meta.json`, `train.bin`, `val.bin`, `unique_words.txt`.

## 6. 안전망 — gzip 보존본

raw 출처가 휘발(`/tmp/`)되거나 정제 절차가 어긋날 위험을 대비해, 정제된 train.txt + val.txt를 각각 gzip으로 묶어 git에 추적:

- 위치: `data/train.txt.gz` (19.6 MB), `data/val.txt.gz` (3.5 MB)
- 복원: `gunzip -kc data/train.txt.gz > path/to/train.txt` (`-k`로 원본 .gz 보존)

위 절차(2 → 3)로 재생성한 결과가 gzip 보존본을 푼 파일과 byte-identical이면 재현 성공.

## 7. 디자인 메모

- **첫 turn 처리 (3.2 (e))**: 정규식 (a)를 그대로 적용하면 conversation 시작이 `<|bos|><|turn|>발화1...`이 됨. 첫 `<|turn|>`을 제거해 `<|bos|>발화1<|turn|>발화2...` 형식으로 정리 — BOS가 첫 turn boundary를 겸함. 토큰 절약(51k개) + 의미 중복 제거.
- **age-5 → age-10 concat 순서**: shuffle 안 함. DataLoader가 minibatch 단위로 무작위 추출하므로 순서 자체는 학습에 영향 없음. 다만 동일 결과 재생성하려면 순서 고정 필요.
- **train/val split**: raw에 이미 분리되어 있음(약 85:15). 재split 안 함.
- **이전 버전(`conv-mix-clean-a510`)과 차이**: TinyHelen 191개 doc 제거. 결과 코퍼스가 dialogue-only로 통일되어 동화체 noise가 빠짐.
