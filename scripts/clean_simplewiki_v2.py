#!/usr/bin/env python3
"""WikiExtractor JSON 출력을 LLM 학습용 clean JSONL로 정제 (v2).

v1 (clean_simplewiki.py) 대비 변경 — 정제 휴리스틱 대거 완화·개선:

  1. title 처리
     - v1: title을 별도 필드로 두고 본문 첫 줄에 title 있으면 잘라냄
     - v2: 본문 첫머리에 "# {title}\n\n" 추가, title-from-text 자르기 제거

  2. redirect
     - v1: drop
     - v2: redirect target의 raw 본문을 가져와 사용 (chain 최대 5단계 follow)

  3. 하단 메타 컷
     - v1: References, External links, See also, Notes, Further reading,
            Bibliography, Sources, Citations 8종 모두 컷
     - v2: External links, References, Bibliography, Sources, Citations
            5종만 컷. See also, Notes, Further reading은 보존

  4. 마크업 제거
     - v1: 빈 문자열 치환 → "word1{{tag}}word2" → "word1word2" 버그
     - v2: 공백 치환 후 multi-space 정리

  5. 단락 50단어 컷 — 제거 (False cut 주범 — Cleopatra 등 짧은 vital 컷)

  6. is_list_only — 제거 (sub-header 다수 페이지 false cut: Hong Kong, University)

  7. 본문 최소 길이 — 200자 → 50자

  8. dedup
     - v1: title-lower + text MD5
     - v2: title-lower만 (text MD5 dedup 제거 — 한 글자 차이 oversensitive)

입력: data/simplewiki/extracted/*/wiki_*  (WikiExtractor --json 출력)
출력: data/simplewiki/simplewiki_clean_v2.jsonl
"""
import json
import os
import re
import sys
from glob import glob

INPUT_DIR = "/Users/joey51/works/pikogpt/data/simplewiki/extracted"
OUTPUT    = "/Users/joey51/works/pikogpt/data/simplewiki/simplewiki_clean_v2.jsonl"

DROP_SECTIONS = re.compile(
    r'^\s*(references|external links?|bibliography|sources|citations)\s*$',
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
RE_REDIRECT_LINE  = re.compile(r'^\s*#?\s*redirect\s*:?\s*\[\[([^\]]+)\]\]', re.IGNORECASE)

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
    """Header가 DROP_SECTIONS(5종)에 매치되면 다음 헤더 전까지 통째 drop."""
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
    """마크업을 모두 공백으로 치환 (v1의 빈 문자열 치환 → 단어 붙음 버그 수정)."""
    text = RE_IMAGE_FILE.sub(' ', text)

    def link_repl(m: re.Match) -> str:
        inner = m.group(1)
        return inner.split('|')[-1] if '|' in inner else inner

    for _ in range(2):
        text = RE_DOUBLE_BRACKET.sub(link_repl, text)
        text = RE_DOUBLE_BRACE.sub(' ', text)
    text = RE_HTML_TAG.sub(' ', text)
    for ent, ch in HTML_ENTITY.items():
        text = text.replace(ent, ch)
    text = RE_NUM_ENTITY.sub(lambda m: chr(int(m.group(1))), text)
    text = RE_CITATION.sub('', text)
    text = RE_PAREN_SENT.sub(' ', text)
    return text


def normalize_ws(text: str) -> str:
    text = RE_MULTI_SPACE.sub(' ', text)
    text = '\n'.join(ln.rstrip() for ln in text.split('\n'))
    text = RE_MULTI_NL.sub('\n\n', text)
    return text.strip()


def detect_redirect(text: str):
    """본문이 redirect면 target title 반환, 아니면 None."""
    head = text.lstrip().split('\n', 1)[0]
    m = RE_REDIRECT_LINE.match(head)
    if m:
        target = m.group(1).split('|')[0].split('#')[0].strip()
        return target if target else None
    return None


def render_text(title: str, body: str) -> str:
    """본문 첫머리에 title을 강조해 추가."""
    return f"# {title}\n\n{body}".strip()


def clean_body(text: str) -> str:
    """제목/redirect 처리 외 본문 정제 공통 단계."""
    text = cut_drop_sections(text)
    text = strip_markup(text)
    text = normalize_ws(text)
    return text


def is_disambig(title: str, text: str) -> bool:
    return ('may refer to:' in text.lower()[:200]
            or 'may refer to' in title.lower())


def main() -> None:
    files = sorted(glob(os.path.join(INPUT_DIR, '*', 'wiki_*')))
    print(f"Processing {len(files)} extracted files", file=sys.stderr)

    # 1차 pass: 모든 raw articles를 dict로 로드 (redirect resolution용)
    print("loading raw articles ...", file=sys.stderr)
    raw_by_title = {}        # title (case-sensitive) → text
    raw_by_lower = {}        # lower(title) → original title
    n_raw = 0
    for path in files:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_raw += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get('title', '').strip()
                if not t:
                    continue
                raw_by_title[t] = obj.get('text', '')
                raw_by_lower.setdefault(t.lower(), t)
    print(f"  raw articles: {n_raw:,} (unique titles: {len(raw_by_title):,})",
          file=sys.stderr)

    def follow_redirect(title: str, max_depth=5):
        """title의 본문이 redirect면 chain을 따라가 target 본문 반환.
        반환값: (final_title, body) 또는 None.
        """
        cur = title
        for _ in range(max_depth):
            body = raw_by_title.get(cur)
            if body is None:
                # case-insensitive fallback
                actual = raw_by_lower.get(cur.lower())
                if actual is None:
                    return None
                cur = actual
                body = raw_by_title.get(cur)
                if body is None:
                    return None
            tgt = detect_redirect(body)
            if tgt is None:
                return cur, body
            cur = tgt
        return None  # too deep

    # 2차 pass: 정제 + redirect 해소
    seen_titles = set()
    n_out = n_dup = n_filt = n_redir_resolved = n_redir_failed = 0

    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for title, text in raw_by_title.items():
            redirect_target = detect_redirect(text)
            if redirect_target is not None:
                resolved = follow_redirect(redirect_target)
                if resolved is None:
                    n_redir_failed += 1
                    n_filt += 1
                    continue
                _, body = resolved
                # body 자체가 redirect였을 가능성 cover됨 (follow_redirect 안에서)
                n_redir_resolved += 1
            else:
                body = text

            body = clean_body(body)

            if len(body) < 50:
                n_filt += 1
                continue
            if is_disambig(title, body):
                n_filt += 1
                continue

            title_lo = title.lower()
            if title_lo in seen_titles:
                n_dup += 1
                continue
            seen_titles.add(title_lo)

            rendered = render_text(title, body)
            out.write(json.dumps({'title': title, 'text': rendered},
                                 ensure_ascii=False) + '\n')
            n_out += 1

    print(f"  redirects resolved:    {n_redir_resolved:,}", file=sys.stderr)
    print(f"  redirects failed:      {n_redir_failed:,}", file=sys.stderr)
    print(f"  filtered (rules):      {n_filt:,}", file=sys.stderr)
    print(f"  duplicates dropped:    {n_dup:,}", file=sys.stderr)
    print(f"  output articles:       {n_out:,}", file=sys.stderr)
    if os.path.exists(OUTPUT):
        print(f"  output file:           {OUTPUT} "
              f"({os.path.getsize(OUTPUT):,} bytes)", file=sys.stderr)


if __name__ == '__main__':
    main()
