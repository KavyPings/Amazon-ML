"""Record normalisation (v2) — language-agnostic, built from what the training data shows.

Noise observed in the real data and handled here:
* Indic-script names/addresses (Devanagari, Bengali, Telugu, Tamil, ...)  -> transliteration
* OCR-style digit swaps inside words (M0rton, Dona1d)                     -> 0->o, 1->l ...
* markers: ">>", "<<", "***", "...", "(ID: 81317)", "#89694"             -> removed
* aliases: "X formerly (known as) Y", "X aka Y", "X dba Y"                 -> alias variants
* domain forms: "physicaltherapycare.com", "LouniversalCom"               -> suffix stripped, no-space form
* dotted/split legal forms "L.L.C.", "S.A.S", "E.U.R.L."                  -> joined single letters
* inserted fillers ("Center", "Services", "Partners", "Shri", "Dr")      -> separate filler field
* repeated words ("Co Co", "LLC LLC")                                      -> de-duplicated
* abbreviations (Rd/Road, Pvt/Private, R./Rue, BD/Boulevard, KY/Kentucky) -> seed + mined maps

All maps are either generic language knowledge written in code or mined from the
provided training pairs. No external data is used.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .translit import indic_to_latin, phonetic

# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------
LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "cos", "llc", "llp", "lp", "ltd", "limited",
    "pvt", "private", "plc", "pllc", "pc", "pa", "lllp", "gmbh", "ag", "bv", "nv", "pte", "pty", "sa", "sas",
    "sasu", "sarl", "eurl", "sci", "snc", "sca", "scop", "selarl", "scp", "ets", "etablissements", "cie",
    "compagnie", "societe", "ste", "opc", "ltda", "srl", "spa", "oy", "ab", "as", "kg", "plc", "dpc", "psc",
}
# canonical legal class, used to compare legal forms across spellings
LEGAL_CLASS = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp",
    "pllc": "pllc", "pc": "pc", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sci": "sci", "sa": "sa", "snc": "snc",
}
FILLER = {"center", "centre", "services", "service", "partners", "the", "shri", "sri", "shree", "smt", "dr", "mr",
          "mrs", "ms", "m", "s", "and", "of", "www", "com", "kumari", "sh", "late", "group"}
ALIAS_RE = re.compile(r"\b(?:formerly known as|formerly|f/k/a|fka|also known as|a/k/a|aka|d/b/a|dba|"
                      r"doing business as|trading as|t/a)\b:?", re.I)
NOISE_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\s*\d+|>>+|<<+|\*{2,}|\.{3,}|[@|~^]", re.I)
DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9\-]+?)(?:\.co\.in|\.com|\.in|\.net|\.org|\.co|com)$", re.I)
OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})

STREET = {  # generic street-type / address abbreviations (EN)
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "cir": "circle",
    "pkwy": "parkway", "hwy": "highway", "sq": "square", "ter": "terrace", "trl": "trail", "cres": "crescent",
    "pt": "point", "mt": "mount", "ft": "fort", "hts": "heights", "apt": "apartment", "ste": "suite",
    "bldg": "building", "fl": "floor", "flr": "floor", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "nr": "near", "opp": "opposite",
    "no": "number", "h": "house", "hno": "house", "twp": "township", "cty": "city", "mkt": "market",
    "stn": "station", "ngr": "nagar", "clny": "colony", "sec": "sector", "extn": "extension", "ext": "extension",
}
STREET_FR = {  # French voie abbreviations (applied for country == France)
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "bvd": "boulevard",
    "all": "allee", "pl": "place", "imp": "impasse", "ch": "chemin", "che": "chemin", "rte": "route",
    "fbg": "faubourg", "qu": "quai", "sq": "square", "st": "saint", "ste": "sainte", "crs": "cours",
    "res": "residence", "lot": "lotissement", "za": "zone", "zi": "zone", "zac": "zone", "cite": "cite",
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "or": "orissa", "od": "orissa",
    "odisha": "orissa", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "ts": "telangana",
    "tg": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand",
    "uttaranchal": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh", "jk": "jammu and kashmir",
    "py": "puducherry", "pondicherry": "puducherry",
}
LANDMARK = {"near", "opposite", "behind", "beside", "besides", "next", "adjacent", "pres", "face", "facing",
            "landmark", "across", "above", "below", "opp", "nr"}
ADDR_FILLER = {"city", "township", "dp", "county", "district", "dist", "po", "ps", "via", "tehsil", "taluk",
               "null", "na", "none", "india", "usa", "us", "france", "street", "road", "number", "house",
               "floor", "near", "the", "de", "la", "le", "du", "des", "of", "and"}

_NON_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)
_SPACES = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_HAS_ALPHA = re.compile(r"[a-z]")


def basic_clean(s: str) -> str:
    """Lowercase, strip accents, '&' -> 'and', punctuation -> space, collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ").replace("+", " and ").replace("'", "").replace("’", "")
    s = _NON_WORD.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def _ocr_fix(tok: str) -> str:
    """Undo digit-for-letter swaps inside words ("m0rton" -> "morton"), leave numbers alone."""
    if tok.isdigit() or not _HAS_ALPHA.search(tok):
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    n_dig = len(tok) - n_alpha
    if n_dig and n_alpha >= 2 and n_alpha >= n_dig and not tok[0].isdigit():
        return tok.translate(OCR)
    return tok


def _join_singles(toks):
    """'l l c' -> 'llc', 's a s' -> 'sas' (dotted abbreviations split by cleaning)."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def _dedupe(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


def name_tokens(raw: str, tmap: dict) -> list[str]:
    """Full cleaning of one name (or alias part) into tokens."""
    s = NOISE_RE.sub(" ", raw)
    s = indic_to_latin(s)
    stripped = s.strip()
    m = DOMAIN_RE.match(stripped) if " " not in stripped else None
    if m:
        s = m.group(1)
    toks = [_ocr_fix(t) for t in basic_clean(s).split()]
    toks = _join_singles(toks)
    toks = [tmap.get(t, t) for t in toks]
    return _dedupe(" ".join(toks).split())


def split_alias(raw: str) -> list[str]:
    parts = [p.strip(" ,;:-()") for p in ALIAS_RE.split(raw)]
    return [p for p in parts if p]


def addr_tokens(raw: str, country: str, amap: dict) -> list[str]:
    s = indic_to_latin(raw or "")
    toks = [_ocr_fix(t) for t in basic_clean(s).split()]
    fr = country == "france"
    out = []
    for t in toks:
        t = amap.get(t, t)
        if fr and t in STREET_FR:
            t = STREET_FR[t]
        elif t in STREET:
            t = STREET[t]
        out.append(t)
    return _dedupe(out)


def state_of(addr_raw: str, country: str, smap: dict) -> str:
    """Canonical state/region from comma fields (learned field->state map + code tables)."""
    fields = [basic_clean(indic_to_latin(f)) for f in (addr_raw or "").split(",")]
    tab = US_STATES if country == "us" else IN_STATES if country == "india" else {}
    for f in reversed(fields):
        if not f:
            continue
        if f in smap:
            return smap[f]
        if f in tab:
            return tab[f]
        if f in tab.values():
            return f
    return ""


# ---------------------------------------------------------------------------
# learned maps (from TRUE training pairs only)
# ---------------------------------------------------------------------------
def _is_abbrev(short: str, long: str) -> bool:
    if len(long) - len(short) < 2 or not short or len(short) > 5 or short[0] != long[0] or short.isdigit():
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def mine_token_map(pairs_tokens, min_count: int = 20, min_share: float = 0.5, positional: bool = True) -> dict:
    """{variant: canonical} from (reference_tokens, variant_tokens) of TRUE pairs.

    Evidence:
    * abbreviation: a differing variant token that is an in-order abbreviation of a reference token
    * positional alignment (optional): same token count, token i differs -> variant_i ~ ref_i
      (learns transliterations such as 'praivet'->'private', 'limitet'->'limited')
    Guards: legal/filler words are left to the dedicated tables; a variant that is itself a
    common reference word (e.g. 'mode', 'jai') is never remapped; chains are resolved.
    """
    cnt: Counter = Counter()
    ref_freq: Counter = Counter()
    for ref, var in pairs_tokens:
        ref_freq.update(set(ref))
        sr, sv = set(ref), set(var)
        dr, dv = sr - sv, sv - sr
        if dr and dv and len(dr) <= 5 and len(dv) <= 5:
            for x in dv:
                for y in dr:
                    if _is_abbrev(x, y):
                        cnt[(x, y)] += 1
        if positional and len(ref) == len(var) and 0 < len(dr) <= 3:
            for a, b in zip(ref, var):
                if a != b and b not in sr and not a.isdigit() and not b.isdigit() and len(b) >= 2:
                    cnt[(b, a)] += 1
    by_var = defaultdict(list)
    for (v, r), c in cnt.items():
        if v in LEGAL or v in FILLER:     # variants may map INTO legal words (praivarr -> private)
            continue
        by_var[v].append((c, r))
    out = {}
    for v, lst in by_var.items():
        lst.sort(reverse=True)
        total = sum(c for c, _ in lst)
        c, r = lst[0]
        if c >= min_count and c / total >= min_share and v != r and ref_freq[v] <= 0.2 * c:
            out[v] = r
    for v in list(out):              # resolve chains a->b->c, drop cycles
        seen, r = {v}, out[v]
        while r in out and r not in seen:
            seen.add(r)
            r = out[r]
        if r in seen:
            del out[v]
        else:
            out[v] = r
    return out


def mine_state_map(s1_states, pool_addrs, min_count: int = 20, purity: float = 0.95) -> dict:
    """Learn {address field -> state} from pool records whose owner's state is known."""
    cnt, tot = Counter(), Counter()
    for st, addr in zip(s1_states, pool_addrs):
        if not st:
            continue
        for f in {basic_clean(indic_to_latin(x)) for x in (addr or "").split(",")}:
            if f and not any(ch.isdigit() for ch in f):
                cnt[(f, st)] += 1
                tot[f] += 1
    out = {}
    for (f, st), n in cnt.items():
        if n >= min_count and n / tot[f] >= purity:
            if f not in out or n > cnt[(f, out[f])]:
                out[f] = st
    return out


# ---------------------------------------------------------------------------
# record normalisation
# ---------------------------------------------------------------------------
def normalize_record(name: str, addr: str, country: str, maps: dict) -> tuple:
    """One record -> tuple aligned with COLS."""
    tmap, amap, smap = maps.get("tokens", {}), maps.get("addr", {}), maps.get("state", {})
    parts = split_alias(name) or [name]
    toks_all = name_tokens(name if len(parts) == 1 else " ".join(parts), tmap)
    variants = [name_tokens(p, tmap) for p in parts] if len(parts) > 1 else [toks_all]
    alias = len(variants) > 1
    # "Brand formerly/aka/dba Real Name": the real business name comes AFTER the marker
    main = variants[-1] if alias else toks_all

    def core_of(tt):
        c = [t for t in tt if t not in LEGAL and t not in FILLER]
        return c or [t for t in tt if t not in FILLER] or tt

    core = core_of(main)
    legal = sorted({LEGAL_CLASS.get(t, t) for t in main if t in LEGAL})
    a_toks = addr_tokens(addr, country, amap)
    nums = [x.lstrip("0") or "0" for x in _DIGITS.findall(" ".join(a_toks))]   # "004245" -> "4245"
    pcs = [x for x in nums if 5 <= len(x) <= 6]
    alpha = [t for t in a_toks if not t.isdigit() and t not in ADDR_FILLER and len(t) > 1]
    flags = (int(any(t in FILLER for t in main)) | (int(any(t in LANDMARK for t in a_toks)) << 1)
             | (int(bool(_INDIC_ANY.search(name))) << 2))
    return (
        " ".join(toks_all),                                   # name_clean
        " ".join(core),                                       # core
        " ".join(core_of(variants[0])) if alias else "",      # alias_core (the brand before the marker)
        " ".join(phonetic(t) for t in core),                  # core_phon
        " ".join(legal),                                      # legal
        " ".join(a_toks),                                     # addr_clean
        " ".join(alpha),                                      # addr_alpha
        " ".join(nums),                                       # nums
        pcs[-1] if pcs else "",                               # postcode
        nums[0] if nums else "",                              # house
        state_of(addr, country, smap),                        # state
        flags,                                                # bit flags: filler | landmark | indic
    )


_INDIC_ANY = re.compile(r"[\u0900-\u0D7F]")
COLS = ["name_clean", "core", "alias_core", "core_phon", "legal", "addr_clean", "addr_alpha", "nums",
        "postcode", "house", "state", "flags"]
STR_DTYPE = "string[pyarrow]"


def _norm_chunk(args):
    names, addrs, countries, maps = args
    rows = [normalize_record(n, a, c, maps) for n, a, c in zip(names, addrs, countries)]
    cols = list(zip(*rows)) if rows else [[] for _ in COLS]
    return [list(c) for c in cols]


def _to_frame(col_lists) -> dict:
    out = {}
    for name, vals in zip(COLS, col_lists):
        if name == "flags":
            out[name] = np.asarray(vals, dtype=np.int8)
        else:
            out[name] = pd.array(vals, dtype=STR_DTYPE)
    return out


def normalize_frame(df: pd.DataFrame, maps: dict, n_jobs: int = 1, chunk: int = 50_000) -> pd.DataFrame:
    """Normalised, memory-lean frame (Arrow-backed strings) for a source table.

    Columns: entity_id, COLS..., country_key, has_filler, landmark, was_indic.
    Runs in parallel over chunks when n_jobs > 1.
    """
    countries = [basic_clean(c) for c in df["country"].tolist()]
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    n = len(df)
    chunks = [(names[s:s + chunk], addrs[s:s + chunk], countries[s:s + chunk], maps) for s in range(0, n, chunk)]
    frames = []
    if n_jobs > 1 and n > 2 * chunk:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for part in ex.map(_norm_chunk, chunks):
                frames.append(pd.DataFrame(_to_frame(part)))
    else:
        for c in chunks:
            frames.append(pd.DataFrame(_to_frame(_norm_chunk(c))))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(_to_frame([[] for _ in COLS]))
    out.index = df.index
    out.insert(0, "entity_id", pd.array(df["entity_id"].tolist(), dtype=STR_DTYPE))
    out["country_key"] = pd.array(countries, dtype=STR_DTYPE)
    fl = out.pop("flags").to_numpy()
    out["has_filler"] = (fl & 1).astype(np.int8)
    out["landmark"] = ((fl >> 1) & 1).astype(np.int8)
    out["was_indic"] = ((fl >> 2) & 1).astype(np.int8)
    return out


def learn_maps(s1: pd.DataFrame, pool: pd.DataFrame, truth: dict, cfg, rng) -> dict:
    """Mine token / address / state maps from TRUE training pairs (training ids only)."""
    maps = {"tokens": {}, "addr": {}, "state": {}}
    s1_idx = s1.set_index("entity_id")
    p_idx = pool.set_index("entity_id")
    pairs = [(a, b) for a, v in truth.items() for b in v]
    names_b = p_idx["business_name"]
    indic = [(a, b) for a, b in pairs if _INDIC_ANY.search(names_b.at[b])]   # every transliteration example
    rest = [(a, b) for a, b in pairs if not _INDIC_ANY.search(names_b.at[b])]
    rng.shuffle(rest)
    pairs = indic[:2_000_000] + rest[:600_000]
    a_ids = [a for a, _ in pairs]
    b_ids = [b for _, b in pairs]
    an, bn = s1_idx.loc[a_ids, "business_name"].tolist(), p_idx.loc[b_ids, "business_name"].tolist()
    aa, ba = s1_idx.loc[a_ids, "business_address"].tolist(), p_idx.loc[b_ids, "business_address"].tolist()
    ac = [basic_clean(c) for c in s1_idx.loc[a_ids, "country"].tolist()]
    maps["tokens"] = mine_token_map(((name_tokens(x, {}), name_tokens(y, {})) for x, y in zip(an, bn)),
                                    cfg.abbrev_min_count, cfg.abbrev_min_share)
    # address: only abbreviation evidence (positional alignment on addresses is too noisy)
    amap = mine_token_map(((addr_tokens(x, c, {}), addr_tokens(y, c, {})) for x, y, c in zip(aa, ba, ac)),
                          cfg.abbrev_min_count, cfg.abbrev_min_share, positional=False)
    states = set(US_STATES) | set(US_STATES.values()) | set(IN_STATES) | set(IN_STATES.values())
    maps["addr"] = {k: v for k, v in amap.items() if k not in states and v not in states and len(k) <= 5}
    s1_state = [state_of(x, c, {}) for x, c in zip(aa, ac)]
    maps["state"] = mine_state_map(s1_state, ba)
    return maps
