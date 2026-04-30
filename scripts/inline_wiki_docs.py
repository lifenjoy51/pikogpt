"""
wiki/train.txt val.txt 변환: 다단락 본문 → 1 line = 1 doc + 리터럴 \\n.

doc 분리: `<|bos|>...<|eos|>` 명시적 매칭(DOTALL).
doc 안 paragraph 구분 `\\n\\n`도 보존되어야 하므로 split("\\n\\n") 방식은 부적합 —
doc 안 단락까지 자르는 버그.
"""

from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "three-stage-v4" / "wiki"

DOC_RE = re.compile(r"<\|bos\|>(.*?)<\|eos\|>", re.DOTALL)


def convert(src: Path, dst: Path):
    text = src.read_text()
    docs: list[str] = []
    for m in DOC_RE.finditer(text):
        body = m.group(1).strip("\n")
        body_inline = body.replace("\n", "\\n")
        docs.append(f"<|bos|>\\n{body_inline}\\n<|eos|>")
    dst.write_text("\n".join(docs) + "\n")
    print(f"{src.name}: {len(docs):,} docs → {dst.stat().st_size:,} bytes")


def main():
    for name in ("train.txt", "val.txt"):
        src = ROOT / name
        convert(src, src)  # in-place


if __name__ == "__main__":
    main()
