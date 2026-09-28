"""Rule-based transliteration of Indic scripts to Latin + a phonetic key.

The nine major Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
Tamil, Telugu, Kannada, Malayalam) share the same ISCII-derived layout: the same
offset inside each 128-codepoint block is the same phonetic letter. So one table
(indexed by offset) transliterates all of them. This is pure code, no external data.

``indic_to_latin("राम मां लॉजिस्टिक्स प्राइवेट लिमिटेड")`` -> "ram man lojistiks praivet limited"
"""
from __future__ import annotations

import re

BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

# independent vowels (offset -> latin)
VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
          0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
          0x60: "ri", 0x61: "li"}
# consonants (offset -> latin without inherent vowel)
CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
        0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
        0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
        0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
        0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
        0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
        0x70: "r", 0x71: "v"}  # Bengali/Assamese extra ra/wa live at 0x70/0x71 of their block
# dependent vowel signs
SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
         0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4F: "au",
         0x55: "", 0x56: "ai", 0x57: "au", 0x62: "li", 0x63: "li"}
VIRAMA = 0x4D
NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}
SKIP = {0x3C, 0x3D, 0x51, 0x52, 0x53, 0x54, 0x64, 0x65}   # nukta, avagraha, accents, danda
ZW = {"\u200c", "\u200d", "\u200b", "\ufeff"}
# Malayalam specifics: chillu letters (final consonants) and the "tt"/"nt" conjuncts
CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k", 0x0D54: "m",
          0x0D55: "y", 0x0D56: "l"}
PRE = [("\u0D31\u0D4D\u0D31", "\u0D1F\u0D4D\u0D1F"),   # Malayalam rra+rra -> tta+tta (tt)
       ("\u0D28\u0D4D\u0D31", "\u0D28\u0D4D\u0D1F"),   # Malayalam na+rra -> na+tta (nt)
       ("\u0BB1\u0BCD\u0BB1", "\u0B9F\u0BCD\u0BB0")]   # Tamil rra+rra -> tta+ra (tr)

_INDIC_RE = re.compile(r"[ऀ-ൿ]")


def _offset(ch: str):
    cp = ord(ch)
    for b in BLOCKS:
        if b <= cp < b + 0x80:
            return cp - b
    return None


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


def indic_to_latin(s: str) -> str:
    """Transliterate every Indic-script run in ``s`` to Latin; other characters pass through."""
    if not s or not _INDIC_RE.search(s):
        return s
    for a, b in PRE:
        s = s.replace(a, b)
    out = []
    pending = False            # a consonant waiting for its (inherent) vowel
    for k, ch in enumerate(s):
        if ch in ZW:
            continue
        if ord(ch) in CHILLU:
            if pending:
                out.append("a")
            out.append(CHILLU[ord(ch)])
            pending = False
            continue
        off = _offset(ch)
        if off is None:
            if pending:
                pending = False   # schwa deletion at word end
            out.append(ch)
            continue
        if off in CONS:
            if pending:
                out.append("a")   # inherent vowel between consonants
            out.append(CONS[off])
            pending = True
        elif off in SIGNS:
            out.append(SIGNS[off])
            pending = False
        elif off == VIRAMA:
            pending = False
        elif off in VOWELS:
            if pending:
                out.append("a")
            out.append(VOWELS[off])
            pending = False
        elif off in NASAL:
            if pending:
                out.append("a")
                pending = False
            nxt = _offset(s[k + 1]) if k + 1 < len(s) else None
            # anusvara before a labial consonant sounds like "m" (bombay, champion)
            out.append("m" if off == 0x02 and nxt in (0x2A, 0x2B, 0x2C, 0x2D, 0x2E) else NASAL[off])
        elif 0x66 <= off <= 0x6F:  # native digits
            if pending:
                pending = False
            out.append(str(off - 0x66))
        elif off in SKIP:
            continue
        else:
            if pending:
                pending = False
    return "".join(out)


_PH = [(re.compile(p), r) for p, r in [
    (r"ph", "f"), (r"(?<=[bcdgjklmnprstvz])h", ""), (r"sh", "s"), (r"ck", "k"), (r"q", "k"), (r"c", "k"),
    (r"x", "ks"), (r"z", "j"), (r"w", "v"), (r"y", "i"), (r"ee", "i"), (r"oo", "u"),
    (r"(?<=.)[aeiou]+", ""), (r"(.)\1+", r"\1")]]


def phonetic(token: str) -> str:
    """Crude cross-script phonetic key: consonant skeleton keeping the first letter.
    'logistics' and transliterated 'lojistiks' -> 'ljstks' / 'ljstks'."""
    t = token
    for p, r in _PH:
        t = p.sub(r, t)
    return t


def phonetic_text(s: str) -> str:
    return " ".join(phonetic(t) for t in s.split())
