"""Text cleaning, normalization and word-level edit distance."""
import re

import pandas as pd

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(v):
    """Cell value -> clean string ('' for blanks; 1.0 -> '1')."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return CONTROL_CHARS.sub("", str(v)).strip()


def to_number(s):
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return s


def col_letter(n):
    """1 -> A, 27 -> AA"""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def colkey(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def yn(b):
    return "Y" if b else "N"


def norm_label(v):
    return re.sub(r"\s+", " ", clean_text(v).lower())


def word_edit_ops(ref, hyp):
    """Substitutions, deletions, insertions between two word lists (Levenshtein)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D, i = D + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return S, D, I


GAP_RE = re.compile(r"_{2,}")
TEACHER_RES = [re.compile(r'teacher\s*:\s*["\u201c][^"\u201d]*["\u201d]', re.I),  # Teacher:"uhum, ring."
               re.compile(r'\(\s*["\u201c][^)]*\)'),                              # ("uhm"), ("i see i see")
               re.compile(r"\bteacher\s*:", re.I)]                                # a bare "Teacher:" label


class Normalizer:
    """Turns transcripts into word lists for WER and has_error, following the part1 settings."""

    def __init__(self, p1):
        self.extra = p1["extra_normalization"]
        self.drop_fillers = p1["drop_fillers"]
        self.fillers = set(p1["fillers"])
        self.spelling = dict(p1["spelling_map"])
        self.merge = p1["merge_compounds"]

    def clean_reference(self, text):
        """Remove teacher speech and gap markers (__) -> (text, had_gap, had_teacher). No-op unless extra."""
        if not self.extra:
            return text, False, False
        had_gap, had_teacher = bool(GAP_RE.search(text)), False
        for rx in TEACHER_RES:
            text, n = rx.subn(" ", text)
            had_teacher |= n > 0
        return GAP_RE.sub(" ", text), had_gap, had_teacher

    def words(self, text):
        """Lowercase, strip punctuation (+ informal spellings and fillers when extra). Numbers stay as written."""
        t = text.lower().replace("\u2019", "'").replace("\u2018", "'")
        if not self.extra:
            t = re.sub(r"[^\w\s']", " ", t)
            return [w.strip("'") for w in t.split() if w.strip("'")]
        t = re.sub(r"[^\w\s']", " ", t.replace("_", " "))
        words = [self.spelling.get(w.strip("'"), w.strip("'")) for w in t.split() if w.strip("'")]
        return [w for w in words if not (self.drop_fillers and w in self.fillers)]

    def merge_compounds(self, a, b):
        """Join neighbouring words in `a` when the joined word appears in `b` (swim suits -> swimsuits)."""
        if not (self.extra and self.merge):
            return a
        joined, out, i = set(b), [], 0
        while i < len(a):
            if i + 1 < len(a) and a[i] + a[i + 1] in joined:
                out.append(a[i] + a[i + 1])
                i += 2
            else:
                out.append(a[i])
                i += 1
        return out
