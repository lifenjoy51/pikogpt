#!/usr/bin/env python3
"""외부 다운로드 데이터셋 합본 — train+val 합쳐 raw text only.

⚠️ **라이선스 주의**: 본 스크립트의 산출물(.txt)은 license 의무 동반 source에서 파생.
   data/external-all-raw/*.txt는 git에 포함하지 않음 (.gitignore의 data/* 패턴 유지).
   각자 본인 책임 하에 재생성/사용. 학습용 활용은 fair use 범주 일반적이나
   재배포·commit 시 원본 라이선스 준수 필수.

원천 라이선스:
- dict (Simple English Dict + NLTK WordNet):
    * NLTK WordNet 3.0 — Princeton WordNet license (자유 사용, copyright notice 필수)
    * Simple Dict — 출처에 따라 Wiktionary CC BY-SA 또는 Webster's 1913 PD
- wiki (SimpleWiki XML dump → vital articles):
    * **Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)**
    * Attribution: Wikipedia + contributors. Share-Alike: 파생물도 CC BY-SA로 배포
- conv (HuggingFace styfeng/TinyDialogues age-5 + age-10):
    * dataset card에서 라이선스 확인 필요

원천 처리본: `data/three-stage-v4/{dict,wiki,conv}/{train,val}.txt`
- dict: Simple English Dict + NLTK WordNet 병합 → 자연어 문단
- wiki: SimpleWiki vital articles (L1-L4) 1 line=1 doc
- conv: TinyDialogues age-5 + age-10

Skip (중복):
- data/two-stage-v3/it == three-stage-v4/conv (byte-identical)
- data/dialogues-a510 ≈ three-stage-v4/conv (~30KB 차이, 사실상 같음)
- data/two-stage-v3/base — wiki와 같은 SimpleWiki source의 sentence-line 처리본 (wiki가 더 깨끗)

출력: data/external-all-raw/{dict,wiki,conv}.txt + README.md (모두 gitignored)
"""
import re
from pathlib import Path

OUT = Path("data/external-all-raw")
OUT.mkdir(parents=True, exist_ok=True)

# bos/eos는 인접한 리터럴 \n까지 함께 제거 (시각 padding)
_BOS_AND_NL = re.compile(r"<\|bos\|>(?:\\n)*")
_EOS_AND_NL = re.compile(r"(?:\\n)*<\|eos\|>")
_SEP = re.compile(r"<\|sep\|>")
# 각 \n-segment의 선두 `# ` markdown 마커
_MD_HEADER_LEAD = re.compile(r"^#\s+")


def normalize_line(s: str) -> str:
    """`<|bos|>`/`<|eos|>`/`<|sep|>` 제거 + 인접 \\n padding 흡수.
    `# ` markdown 마커 제거. **리터럴 `\\n`은 보존** (title→content / paragraph 경계).
    `<|turn|>`은 양쪽 공백 ` <|turn|> ` 형태로 통일."""
    s = s.replace("\r", "").rstrip("\n")
    s = _BOS_AND_NL.sub("", s)
    s = _EOS_AND_NL.sub("", s)
    s = _SEP.sub("", s)
    s = s.replace("<|turn|>", " <|turn|> ")
    # 리터럴 \n 단위로 split → segment별로 markdown 제거 + 공백 정규화 → \n으로 재결합
    parts = s.split("\\n")
    parts = [_MD_HEADER_LEAD.sub("", p) for p in parts]
    parts = [" ".join(p.split()) for p in parts]
    # 선두/말미 빈 segment만 제거 (내부 빈 segment는 paragraph break로 유지)
    while parts and not parts[0]:
        parts.pop(0)
    while parts and not parts[-1]:
        parts.pop()
    return "\\n".join(parts)

SOURCES = {
    "dict": ("data/three-stage-v4/dict", "Simple English Dict + NLTK WordNet 병합 → 자연어 문단"),
    "wiki": ("data/three-stage-v4/wiki", "SimpleWiki XML dump → vital articles L1-L4 (1 line=1 doc)"),
    "conv": ("data/three-stage-v4/conv", "HuggingFace styfeng/TinyDialogues age-5 + age-10"),
}


def merge(src_dir: Path, out_file: Path) -> tuple[int, int]:
    lines = 0
    bytes_out = 0
    with out_file.open("w") as fout:
        for split in ("train.txt", "val.txt"):
            p = src_dir / split
            if not p.exists():
                continue
            with p.open() as fin:
                for raw in fin:
                    norm = normalize_line(raw)
                    if not norm:
                        continue
                    fout.write(norm + "\n")
                    lines += 1
                    bytes_out += len(norm.encode("utf-8")) + 1
    return lines, bytes_out


def main():
    stats = {}
    for name, (src, desc) in SOURCES.items():
        out = OUT / f"{name}.txt"
        n, b = merge(Path(src), out)
        stats[name] = (n, b, desc, src)
        print(f"[{name}] {n:,} lines, {b:,} bytes -> {out}")

    readme = OUT / "README.md"
    rows = []
    for name, (n, b, desc, src) in stats.items():
        rows.append(f"| `{name}.txt` | {n:,} | {b:,} | `{src}/{{train,val}}.txt` | {desc} |")
    readme.write_text(f"""# External all raw — 외부 다운로드 데이터셋 합본 (텍스트 전용)

생성: `scripts/build_external_all_raw.py`

각 파일은 해당 source의 train+val을 단순 concat (학습 prep 단계에서 재분할).

## 파일

| 파일 | lines | bytes | 원천 | 출처 |
|---|---:|---:|---|---|
{chr(10).join(rows)}

## Dedup 메모 (포함 안 한 source)

- `data/two-stage-v3/it/{{train,val}}.txt` — `three-stage-v4/conv`와 byte-identical
- `data/dialogues-a510/{{train,val}}.txt` — `three-stage-v4/conv`와 첫/끝 동일, ~30KB 차이뿐 (실질 동일)
- `data/two-stage-v3/base/{{train,val}}.txt` — `three-stage-v4/wiki`와 같은 SimpleWiki source의 sentence-line 처리본. wiki가 doc-per-line로 더 깨끗해 채택.

## 형식

- 모든 파일 1 line = 1 record (doc)
- 정규화/dedup은 적용 안 함 (이미 source에서 처리됨)
- 토큰화 안 함 — 학습용 BPE는 별도 단계
""")
    print(f"\nREADME: {readme}")


if __name__ == "__main__":
    main()
