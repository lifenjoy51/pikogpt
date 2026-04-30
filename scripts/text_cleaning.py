#!/usr/bin/env python3
"""텍스트 코퍼스 ASCII 정제 라이브러리.

BPE 학습 전 train.txt / val.txt 같은 plain-text 코퍼스에 적용하는 정제 함수 모음.
이전 `clean_v2_data.py` (base-v2/it-v2 paths 하드코딩) 의 로직을 재사용 가능한
모듈로 분리. main 처리(어떤 파일을 정제할지)는 호출자 빌더 스크립트가 담당.

핵심 함수:
  clean_strict(text)
      smart-quote / dash / ellipsis → ASCII 등가 + non-ASCII 제거 + 공백 정규화
  build_allowed_chars(combined_text)
      pivot char (`_`) 빈도 + 1을 임계값으로 동적 결정해 보존 char set 산출
  filter_low_freq_chars(text, allowed)
      allowed에 없는 char를 공백으로 치환 (단어 경계 보존)

사용 예:
  from text_cleaning import clean_strict, build_allowed_chars, filter_low_freq_chars

  raw = open(src).read()
  cleaned = clean_strict(raw)
  # 합본으로 임계값 결정
  allowed, removed, threshold = build_allowed_chars(train_text + it_train_text)
  # 모든 분할 파일에 동일 allowed 적용
  filtered = filter_low_freq_chars(cleaned, allowed)
"""

import re
from collections import Counter

SMART = str.maketrans({
    '‘': "'",   # left single
    '’': "'",   # right single
    '“': '"',   # left double
    '”': '"',   # right double
    '–': '-',   # en dash
    '—': '-',   # em dash
    '…': '...', # ellipsis
})

NON_ASCII = re.compile(r'[^\x20-\x7e\n]')
WS_RE = re.compile(r'[ \t]+')

# 항상 보존되는 chars — 학습에 필수.
# - 알파벳·숫자
# - 공백·newline
# - special token wrap chars (`<|>`) — `<|bos|>` 등 단일 토큰 분리
# - 자주 등장 punctuation
ALWAYS_KEEP: set[str] = (
    set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    | set(" \n")
    | set("<|>")
    | set(".,'\"?!:;-*()")
)

# 임계값은 pivot char (`_`) 빈도 + 1 기준으로 동적 결정.
# 즉 `_`보다 자주 등장한 char만 보존 (`_` 자체도 제거).
# 데이터 분포에 따라 자동 적응. `_`는 자연어 코퍼스에서 일관되게 "낮은 빈도 cut-off"
# 위치를 차지해 기준점으로 적합.
THRESHOLD_PIVOT_CHAR = '_'


def clean_strict(text: str) -> str:
    """smart-quote/dash/ellipsis → ASCII 등가 + non-ASCII 제거 + 공백 정규화.
    paragraph 구분(\\n\\n)은 모두 단일 공백으로 평탄화 (1라인 1doc 형식용)."""
    text = text.translate(SMART)
    text = NON_ASCII.sub('', text)
    text = WS_RE.sub(' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    return '\n'.join(ln for ln in lines if ln)


def clean_preserve_paragraphs(text: str) -> str:
    """clean_strict와 동일하되 paragraph 구분(\\n\\n)을 보존.

    - smart-quote → ASCII 등가
    - non-ASCII 제거
    - 라인 안의 multi-space/tab → 단일 공백
    - 라인 trim (단, 빈 줄은 살려서 paragraph 구분 유지)
    - 3+ 연속 newline → 2개 (단락 간 한 줄만)
    """
    text = text.translate(SMART)
    text = NON_ASCII.sub('', text)
    text = WS_RE.sub(' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = '\n'.join(lines)
    # 3+ newline → 2 (단락 간 빈 줄 1개로 정규화)
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text.strip()


def filter_low_freq_chars(text: str, allowed: set) -> str:
    """allowed에 없는 char를 공백으로 치환 (단어 경계 보존)."""
    return ''.join(c if c in allowed else ' ' for c in text)


def build_allowed_chars(combined_text: str) -> tuple[set, dict, int]:
    """combined_text에서 char 빈도 측정.

    - 임계값 = THRESHOLD_PIVOT_CHAR(`_`) 빈도 + 1
      → `_`보다 자주 등장한 char만 보존 (`_` 자체도 제거)
    - ALWAYS_KEEP은 항상 포함

    Returns:
        (allowed_chars, removed_chars, computed_threshold)
    """
    counter = Counter(combined_text)
    pivot_freq = counter.get(THRESHOLD_PIVOT_CHAR, 0)
    threshold = pivot_freq + 1
    allowed = set(ALWAYS_KEEP)
    for c, freq in counter.items():
        if freq >= threshold:
            allowed.add(c)
    removed = {c: freq for c, freq in counter.items() if c not in allowed}
    return allowed, removed, threshold


def normalize_after_filter(text: str) -> str:
    """filter_low_freq_chars 결과의 다중 공백 squeeze + 라인 trim."""
    text = WS_RE.sub(' ', text)
    return '\n'.join(ln.strip() for ln in text.splitlines() if ln.strip())
