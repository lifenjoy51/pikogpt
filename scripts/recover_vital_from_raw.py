#!/usr/bin/env python3
"""
vital_titles_resolved.json 의 missing 항목을 wikiextractor raw 출력
(data/simplewiki/extracted/*/wiki_*) 에서 찾아 복구.

clean_simplewiki.py 의 cleaning 함수를 그대로 적용하되, vital 항목은
길이 임계값을 완화 (200 → 50 chars) — 위계상 중요한 표제어가
stub이라는 이유로 누락되는 것을 막기 위함.

산출:
  data/simplewiki/simplewiki_vital_extras.jsonl   — clean format
  data/external/vital_articles/vital_recovery_report.json
"""

import hashlib
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

VITAL_DIR = Path("data/external/vital_articles")
RAW_DIR = "data/simplewiki/extracted"
OUT_JSONL = Path("data/simplewiki/simplewiki_vital_extras.jsonl")
REPORT = VITAL_DIR / "vital_recovery_report.json"

# clean_simplewiki.py 의 함수와 동일 (복제)
DROP_SECTIONS = re.compile(
    r'^\s*(references|external links?|see also|notes|further reading|bibliography|sources|citations)\s*$',
    re.IGNORECASE,
)
RE_IMAGE_FILE     = re.compile(r'\[\[(?:File|Image):[^\]]*\]\]', re.IGNORECASE)
RE_DOUBLE_BRACKET = re.compile(r'\[\[([^\]]*?)\]\]')
RE_DOUBLE_BRACE   = re.compile(r'\{\{[^{}]*\}\}')
RE_HTML_TAG       = re.compile(r'<[^>]+>')
RE_CITATION       = re.compile(r'\[\d+\]')
RE_NUM_ENTITY     = re.compile(r'&#(\d+);')
RE_MULTI_NL       = re.compile(r'\n{3,}')
RE_MULTI_SPACE    = re.compile(r'[ \t]+')
RE_PAREN_SENT     = re.compile(r'\([^()]{80,}\)')
RE_REDIRECT       = re.compile(r'^\s*#?\s*redirect', re.IGNORECASE)
HTML_ENTITY = {
    '&amp;': '&', '&nbsp;': ' ', '&ndash;': '-', '&mdash;': '-',
    '&quot;': '"', '&apos;': "'", '&lt;': '<', '&gt;': '>',
}


def is_header_like(line: str) -> bool:
    s = line.strip()
    if not s or not s[0].isalpha():
        return False
    if len(s) > 60:
        return False
    if s.endswith(('.', '!', '?', ':', ',', ';', '"', "'")):
        return False
    if len(s.split()) > 6:
        return False
    return True


def cut_drop_sections(text: str) -> str:
    out, skipping = [], False
    for ln in text.split('\n'):
        if is_header_like(ln):
            if DROP_SECTIONS.match(ln.strip()):
                skipping = True
                continue
            skipping = False
        if not skipping:
            out.append(ln)
    return '\n'.join(out)


def strip_markup(text: str) -> str:
    text = RE_IMAGE_FILE.sub('', text)
    def link_repl(m):
        inner = m.group(1)
        return inner.split('|')[-1] if '|' in inner else inner
    for _ in range(2):
        text = RE_DOUBLE_BRACKET.sub(link_repl, text)
        text = RE_DOUBLE_BRACE.sub('', text)
    text = RE_HTML_TAG.sub('', text)
    for ent, ch in HTML_ENTITY.items():
        text = text.replace(ent, ch)
    text = RE_NUM_ENTITY.sub(lambda m: chr(int(m.group(1))), text)
    text = RE_CITATION.sub('', text)
    text = RE_PAREN_SENT.sub('', text)
    return text


def normalize_ws(text: str) -> str:
    text = RE_MULTI_SPACE.sub(' ', text)
    text = '\n'.join(ln.rstrip() for ln in text.split('\n'))
    text = RE_MULTI_NL.sub('\n\n', text)
    return text.strip()


def is_list_only(text: str) -> bool:
    """완화된 list-only 판정: bullet으로 시작하는 라인 비율로만 판단.
    short-paragraph(sub-header) 다수인 백과사전 페이지의 false positive 방지.
    """
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return True
    bullet_lines = sum(1 for ln in lines if ln.startswith(('*', '-', '#', '•', '·')))
    return bullet_lines / len(lines) > 0.6


def clean_lenient(obj, min_chars=50, min_para_words=20):
    """clean_simplewiki.py와 같지만 min_chars=50, paragraph filter 완화."""
    title = obj.get('title', '').strip()
    text = obj.get('text', '')
    if not title or not text:
        return None, "empty"
    if RE_REDIRECT.match(text):
        return None, "redirect"
    if text.lstrip().startswith(title):
        text = text.lstrip()[len(title):].lstrip()
    text = cut_drop_sections(text)
    text = strip_markup(text)
    text = normalize_ws(text)
    paras = text.split('\n\n')
    keep = [p for p in paras if len(p.split()) >= min_para_words]
    text = '\n\n'.join(keep)
    if len(text) < min_chars:
        return None, f"too_short_{len(text)}c"
    if is_list_only(text):
        return None, "list_only"
    if 'may refer to:' in text.lower()[:200] or 'may refer to' in title.lower():
        return None, "disambig"
    return {'title': title, 'text': text}, "ok"


def main():
    resolved = json.load(open(VITAL_DIR / "vital_titles_resolved.json"))
    missing_set = {t.lower() for t, info in resolved.items() if info["simplewiki_title"] is None}
    missing_originals = [t for t, info in resolved.items() if info["simplewiki_title"] is None]
    print(f"missing to recover: {len(missing_originals)}", file=sys.stderr)

    files = sorted(glob(os.path.join(RAW_DIR, '*', 'wiki_*')))
    print(f"scanning {len(files)} raw files", file=sys.stderr)

    # raw에서 (case-insensitive) 매칭. simple Wikipedia title은 unique 가정
    raw_hits = {}  # title_lower -> raw obj
    for path in files:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get('title', '').strip()
                if t.lower() in missing_set and t.lower() not in raw_hits:
                    raw_hits[t.lower()] = obj

    print(f"raw hits: {len(raw_hits)} / {len(missing_set)}", file=sys.stderr)

    report = {}
    accepted = []

    for orig in missing_originals:
        key = orig.lower()
        if key not in raw_hits:
            report[orig] = {"status": "not_in_raw"}
            continue
        obj = raw_hits[key]
        cleaned, reason = clean_lenient(obj)
        entry = {
            "raw_title": obj.get("title"),
            "raw_chars": len(obj.get("text", "")),
            "cleaned_status": reason,
        }
        if cleaned:
            entry["cleaned_chars"] = len(cleaned["text"])
            accepted.append(cleaned)
        report[orig] = entry

    # write
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for c in accepted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with REPORT.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # vital_titles_resolved.json 업데이트
    resolved_path = VITAL_DIR / "vital_titles_resolved.json"
    resolved = json.load(open(resolved_path))
    # 기존 entry에 source 필드 추가 (없으면)
    for t, info in resolved.items():
        if info.get("simplewiki_title") and "source" not in info:
            info["source"] = "clean"
    # raw_recovery 결과 반영
    for orig, rep in report.items():
        if rep.get("cleaned_status") == "ok":
            resolved[orig] = {
                "level": resolved[orig]["level"],
                "category": resolved[orig].get("category"),
                "simplewiki_title": rep["raw_title"],
                "method": "raw_recovery",
                "source": "extras",
            }
    out = {t: resolved[t] for t in sorted(resolved, key=lambda x: (resolved[x]["level"], x))}
    with resolved_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"updated {resolved_path}")

    # stats
    from collections import Counter
    c = Counter(v.get("cleaned_status", v.get("status")) for v in report.values())
    print()
    print("=== outcome ===")
    for k, n in c.most_common():
        print(f"  {k}: {n}")
    print()
    print(f"wrote {OUT_JSONL}  ({len(accepted)} docs)")
    print(f"wrote {REPORT}")
    print()
    if accepted:
        print("=== recovered ===")
        for c in accepted:
            print(f"  {c['title']!r}  ({len(c['text'])}c, {len(c['text'].split())}w)")
    print()
    rejected = [(o, v) for o, v in report.items() if v.get("cleaned_status") != "ok"]
    if rejected:
        print(f"=== still rejected ({len(rejected)}) ===")
        for o, v in rejected:
            status = v.get("cleaned_status") or v.get("status")
            raw = v.get("raw_chars", "-")
            print(f"  [{status:>14s}]  raw={raw}c  {o}")


if __name__ == "__main__":
    main()
