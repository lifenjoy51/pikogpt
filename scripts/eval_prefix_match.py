#!/usr/bin/env python3
"""prefix match 평가 — sampler 로그에서 26 prompt 결과 추출 후
4-char base와의 prefix 일치 길이 측정. 사용:

  python3 scripts/eval_prefix_match.py logs/sampler-XXX.log
"""
import re, sys, os

# wordstart-4prefix 데이터셋의 글자별 4-char base
BASES = {
    'a': 'appr', 'b': 'brea', 'c': 'comp', 'd': 'dist', 'e': 'elec',
    'f': 'fire', 'g': 'gene', 'h': 'hand', 'i': 'inte', 'j': 'just',
    'k': 'know', 'l': 'land', 'm': 'mode', 'n': 'news', 'o': 'over',
    'p': 'pres', 'q': 'quar', 'r': 'reco', 's': 'stra', 't': 'tran',
    'u': 'unde', 'v': 'viol', 'w': 'wate', 'y': 'yell', 'z': 'zool',
}

def match_len(base: str, sample: str) -> int:
    n = 0
    for k in range(min(len(base), len(sample)) + 1):
        if sample.startswith(base[:k]):
            n = k
    return n

def parse_samples(log_text: str) -> dict:
    pat = re.compile(
        r"=== prompt: '(.+?)' ===\n.*?\[1\] (.*?)(?=\n# 텍스트 생성 완료|\n=== prompt|$)",
        re.DOTALL,
    )
    samples = {}
    for prompt, sample in pat.findall(log_text):
        # prompt: "<|bos|>a" 형태 — 끝 letter 추출
        m = re.search(r"<\|bos\|>([a-z])$", prompt)
        if not m:
            # 옛 포맷 폴백: "\na" or just "a"
            letter = prompt.replace("\n", "").strip()
            if len(letter) != 1 or not letter.isalpha():
                continue
        else:
            letter = m.group(1)
        # 첫 단어 추출 — sample 앞부분의 \n/<|bos|> 등 제거하고 "단어 끝(eos/newline) 전까지"
        s = sample
        # special token 제거 (sampler가 출력에서 빼주지만 안전하게 정리)
        for special in ['<|bos|>', '<|eos|>', '<|unk|>', '<|turn|>', '<|sep|>']:
            s = s.replace(special, '')
        s = s.lstrip('\n').lstrip()
        first = s.split('\n', 1)[0] if '\n' in s else s
        samples[letter] = first
    return samples

def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <sampler-log-path>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        text = f.read()
    samples = parse_samples(text)
    print(f"# log: {os.path.basename(path)}")
    print(f"{'L':>2} {'base':>5} {'sample':>22} {'m':>3}")
    print('-' * 38)
    hits = [0] * 6
    total = 0
    for L in 'abcdefghijklmnopqrstuvwxyz':
        if L not in BASES:
            continue
        sample = samples.get(L, '?')
        m = match_len(BASES[L], sample) if sample != '?' else 0
        print(f"{L:>2} {BASES[L]:>5} {sample[:22]:>22} {m:>3}")
        for k in range(1, 6):
            if m >= k:
                hits[k] += 1
        total += 1
    print('-' * 38)
    print(f"≥1글자: {hits[1]}/{total}  "
          f"≥2글자: {hits[2]}/{total}  "
          f"≥3글자: {hits[3]}/{total}  "
          f"≥4글자: {hits[4]}/{total}")

if __name__ == "__main__":
    main()
