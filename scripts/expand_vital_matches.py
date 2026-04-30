#!/usr/bin/env python3
"""
vital_titles_resolved.json 의 missing 표제어를 추가 전략으로 simplewiki에 매칭.

매칭 후보 풀 (3종):
  - clean_set       simplewiki_clean.jsonl 안의 모든 title
  - extras_set      simplewiki_vital_extras.jsonl 안의 title (이미 raw에서 살린 것)
  - raw_dict        wikiextractor 원본의 모든 title → text 매핑

전략 (순차 적용, 한 번 잡히면 다음 단계 skip):
  A. exact (이미 처리됐지만 재확인)
  B. disambiguator strip            "Cell (biology)" → "Cell"
  C. comma-tail strip               "Lagos, Nigeria" → "Lagos"
  D. punctuation normalize          apostrophe, &↔and, hyphen↔space
  E. en wiki API redirect           batch 50, sleep 0.3
  F. raw_dict (no normalize) 직접   raw_recovery에서 놓친 것 보완

E는 환경변수 SKIP_EN=1 로 비활성화 가능.

매칭된 항목:
  - clean에 있으면 source="clean", method=원래 strategy 이름
  - raw에만 있으면 source="extras" → lenient cleaner로 본문 정제 후
    simplewiki_vital_extras.jsonl 에 추가 (dedup 후)
  - vital_titles_resolved.json 갱신
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from glob import glob
from pathlib import Path

VITAL_DIR = Path("data/external/vital_articles")
CLEAN = Path("data/simplewiki/simplewiki_clean.jsonl")
EXTRAS = Path("data/simplewiki/simplewiki_vital_extras.jsonl")
RAW_GLOB = "data/simplewiki/extracted/*/wiki_*"
USER_AGENT = "PikoGPT-vital-expand/1.0 (joey.51@kakaocorp.com)"

# ===== lenient cleaner (recover_vital_from_raw.py 와 동일) =====
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
HTML_ENTITY = {'&amp;':'&','&nbsp;':' ','&ndash;':'-','&mdash;':'-',
               '&quot;':'"','&apos;':"'",'&lt;':'<','&gt;':'>'}


def is_header_like(s):
    s = s.strip()
    if not s or not s[0].isalpha(): return False
    if len(s) > 60: return False
    if s.endswith(('.', '!', '?', ':', ',', ';', '"', "'")): return False
    if len(s.split()) > 6: return False
    return True


def cut_drop_sections(text):
    out, skipping = [], False
    for ln in text.split('\n'):
        if is_header_like(ln):
            if DROP_SECTIONS.match(ln.strip()):
                skipping = True; continue
            skipping = False
        if not skipping:
            out.append(ln)
    return '\n'.join(out)


def strip_markup(text):
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


def normalize_ws(text):
    text = RE_MULTI_SPACE.sub(' ', text)
    text = '\n'.join(ln.rstrip() for ln in text.split('\n'))
    text = RE_MULTI_NL.sub('\n\n', text)
    return text.strip()


def is_list_only(text):
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines: return True
    bullet = sum(1 for ln in lines if ln.startswith(('*', '-', '#', '•', '·')))
    return bullet / len(lines) > 0.6


def clean_lenient(title, text, min_chars=50, min_para_words=20):
    if not title or not text: return None, "empty"
    if RE_REDIRECT.match(text): return None, "redirect"
    if text.lstrip().startswith(title):
        text = text.lstrip()[len(title):].lstrip()
    text = cut_drop_sections(text)
    text = strip_markup(text)
    text = normalize_ws(text)
    paras = text.split('\n\n')
    keep = [p for p in paras if len(p.split()) >= min_para_words]
    text = '\n\n'.join(keep)
    if len(text) < min_chars: return None, f"too_short_{len(text)}c"
    if is_list_only(text): return None, "list_only"
    if 'may refer to:' in text.lower()[:200] or 'may refer to' in title.lower():
        return None, "disambig"
    return {'title': title, 'text': text}, "ok"


# ===== normalize variants =====
RE_PAREN_SUFFIX = re.compile(r'\s*\([^)]*\)\s*$')
RE_COMMA_TAIL = re.compile(r',\s*[^,]+$')


def variants(title):
    """매칭 후보 표제어 변형들 (lowercase, deduped)."""
    seen = set()
    out = []
    def add(s):
        s = s.strip().lower()
        if s and s not in seen:
            seen.add(s); out.append(s)
    add(title)
    # B. disambiguator 제거: "Cell (biology)" → "Cell"
    stripped = RE_PAREN_SUFFIX.sub('', title).strip()
    if stripped != title:
        add(stripped)
    # C. comma-tail 제거: "Lagos, Nigeria" → "Lagos"
    if ',' in title:
        head = RE_COMMA_TAIL.sub('', title).strip()
        if head and head != title:
            add(head)
            # B+C 결합
            head_stripped = RE_PAREN_SUFFIX.sub('', head).strip()
            if head_stripped != head:
                add(head_stripped)
    # D. punctuation normalize
    cur = title
    cur = cur.replace('‘', "'").replace('’', "'")
    cur = cur.replace('“', '"').replace('”', '"')
    cur = cur.replace('—', '-').replace('–', '-')
    add(cur)
    # & ↔ and
    if '&' in cur:
        add(cur.replace('&', 'and'))
    if ' and ' in cur.lower():
        add(re.sub(r'\sand\s', ' & ', cur, flags=re.IGNORECASE))
    return out


def api_query(host, titles, timeout=30):
    params = {
        "action": "query", "titles": "|".join(titles),
        "redirects": "1", "format": "json", "formatversion": "2",
    }
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def follow_resp(body, originals):
    q = body.get("query", {})
    norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
    redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
    pages = {p["title"]: p for p in q.get("pages", [])}
    out = {}
    for orig in originals:
        cur = norm.get(orig, orig)
        for _ in range(5):
            nxt = redir.get(cur)
            if nxt is None: break
            cur = nxt
        page = pages.get(cur)
        if page is None or page.get("missing"):
            out[orig] = None
        else:
            out[orig] = page["title"]
    return out


def batch(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n: yield buf; buf = []
    if buf: yield buf


def main():
    skip_en = os.environ.get("SKIP_EN") == "1"

    resolved = json.load(open(VITAL_DIR / "vital_titles_resolved.json"))
    vital_meta = json.load(open(VITAL_DIR / "vital_titles.json"))

    # Pool 1: simplewiki_clean titles → 그 자체 (원본 title)
    clean_lower2title = {}
    with open(CLEAN) as f:
        for line in f:
            t = json.loads(line).get("title", "").strip()
            if t:
                clean_lower2title[t.lower()] = t
    print(f"clean titles: {len(clean_lower2title):,}", file=sys.stderr)

    # Pool 2: 기존 simplewiki_vital_extras 의 titles
    extras_existing_titles = set()
    if EXTRAS.exists():
        with open(EXTRAS) as f:
            for line in f:
                t = json.loads(line).get("title", "").strip()
                if t:
                    extras_existing_titles.add(t)
    print(f"existing extras titles: {len(extras_existing_titles):,}", file=sys.stderr)

    # Pool 3: raw extracted (lazy)
    print(f"scanning raw...", file=sys.stderr)
    raw_lower2obj = {}
    files = sorted(glob(RAW_GLOB))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("title", "").strip()
                if t and t.lower() not in raw_lower2obj:
                    raw_lower2obj[t.lower()] = obj
    print(f"raw titles: {len(raw_lower2obj):,}", file=sys.stderr)

    # missing 추출
    missing = [t for t, info in resolved.items() if info.get("simplewiki_title") is None]
    print(f"missing entries: {len(missing):,}", file=sys.stderr)

    # 매칭 결과 누적
    new_matches = {}  # vital_title -> (sw_title_actual, method, source)
    new_extras_objs = {}  # raw obj title -> raw obj (dedup으로 중복 방지)

    def try_pools(variants_list):
        """variants 리스트를 clean → existing extras → raw 순으로 검사."""
        for v in variants_list:
            if v in clean_lower2title:
                return clean_lower2title[v], "clean"
            # existing extras도 lower set으로 검사
            for ex_t in extras_existing_titles:
                if ex_t.lower() == v:
                    return ex_t, "extras"
            if v in raw_lower2obj:
                obj = raw_lower2obj[v]
                return obj["title"], "raw"
        return None, None

    # Stage B-D (no API): variants 매칭
    no_api_remaining = []
    for orig in missing:
        vlist = variants(orig)
        # 첫 variant는 원본 lowercase — 이미 시도된 것이지만 안전망
        sw_title, source = try_pools(vlist)
        if sw_title is None:
            no_api_remaining.append(orig)
            continue
        # 어느 variant로 잡혔는지에 따라 method 명명
        if vlist[0] == sw_title.lower():
            method = "exact_recheck"
        elif source == "raw":
            method = "raw_normalized"
        else:
            method = "normalized"
        new_matches[orig] = (sw_title, method, source)

    no_api_count = len(missing) - len(no_api_remaining)
    print(f"matched without API: {no_api_count}", file=sys.stderr)
    print(f"remaining for API: {len(no_api_remaining)}", file=sys.stderr)

    # Stage E: en wiki API redirect
    if skip_en:
        print(f"SKIP_EN=1 — en API stage 건너뜀", file=sys.stderr)
    else:
        api_done = 0
        for chunk in batch(no_api_remaining, 50):
            try:
                body = api_query("en.wikipedia.org", chunk)
            except Exception as e:
                print(f"  en API error: {e}", file=sys.stderr); continue
            result = follow_resp(body, chunk)
            for orig, en_target in result.items():
                if not en_target: continue
                v = en_target.lower()
                if v in clean_lower2title:
                    new_matches[orig] = (clean_lower2title[v], "en_redirect", "clean")
                elif any(t.lower() == v for t in extras_existing_titles):
                    matching = next(t for t in extras_existing_titles if t.lower() == v)
                    new_matches[orig] = (matching, "en_redirect", "extras")
                elif v in raw_lower2obj:
                    obj = raw_lower2obj[v]
                    new_matches[orig] = (obj["title"], "en_redirect_raw", "raw")
            api_done += len(chunk)
            if api_done % 1000 == 0:
                print(f"  en API progress: {api_done}/{len(no_api_remaining)}", file=sys.stderr)
            time.sleep(0.3)

    # raw에서 매칭된 것 본문 정제
    new_extras_count = 0
    for orig, (sw_title, method, source) in new_matches.items():
        if source != "raw": continue
        obj = raw_lower2obj[sw_title.lower()]
        cleaned, reason = clean_lenient(obj["title"], obj.get("text", ""))
        if cleaned and obj["title"] not in extras_existing_titles and obj["title"] not in new_extras_objs:
            new_extras_objs[obj["title"]] = cleaned
            new_extras_count += 1
        elif not cleaned:
            # 본문이 부실하면 매칭 취소
            new_matches[orig] = None

    # 취소된 것 정리
    new_matches = {o: m for o, m in new_matches.items() if m is not None}

    # extras 파일에 append
    if new_extras_objs:
        with EXTRAS.open("a") as f:
            for t in sorted(new_extras_objs):
                f.write(json.dumps(new_extras_objs[t], ensure_ascii=False) + "\n")

    # resolved.json 업데이트
    for orig, (sw_title, method, source) in new_matches.items():
        resolved[orig] = {
            "level": resolved[orig]["level"],
            "category": resolved[orig].get("category") or vital_meta.get(orig, {}).get("category"),
            "simplewiki_title": sw_title,
            "method": method,
            "source": "extras" if source == "raw" else source,
        }
    out_path = VITAL_DIR / "vital_titles_resolved.json"
    out_sorted = {t: resolved[t] for t in sorted(resolved, key=lambda x: (resolved[x]["level"], x))}
    with out_path.open("w") as f:
        json.dump(out_sorted, f, ensure_ascii=False, indent=2)

    # 통계
    print()
    print("=== expansion summary ===")
    by_method = Counter(m for _, (_, m, _) in new_matches.items())
    by_source = Counter(s for _, (_, _, s) in new_matches.items())
    print(f"  newly matched: {len(new_matches):,}")
    print(f"  new extras docs: {new_extras_count:,}")
    print(f"  by method:")
    for k, n in by_method.most_common():
        print(f"    {k}: {n:,}")
    print(f"  by source:")
    for k, n in by_source.most_common():
        print(f"    {k}: {n:,}")

    # final 매칭률
    matched = sum(1 for v in resolved.values() if v.get("simplewiki_title"))
    total = len(resolved)
    print()
    print(f"=== final ===")
    print(f"  matched: {matched:,}/{total:,}  ({100*matched/total:.1f}%)")

    per_level = {}
    for v in resolved.values():
        lv = v["level"]
        per_level.setdefault(lv, [0, 0])
        per_level[lv][1] += 1
        if v.get("simplewiki_title"):
            per_level[lv][0] += 1
    for lv in sorted(per_level):
        ok, total_lv = per_level[lv]
        print(f"  L{lv}: {ok:,}/{total_lv:,}  ({100*ok/total_lv:.1f}%)")


if __name__ == "__main__":
    main()
