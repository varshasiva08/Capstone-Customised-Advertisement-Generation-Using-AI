"""
Deterministic brief parser (no LLM).

Turns a free-text brand/ad brief into a demographic profile:

    ethnicity, gender, age (decade), body_type, [region]

Design goals (and why this replaced the phi3:mini parser):

* Reproducible and testable. Same text -> same output, no model server, no
  sampling. `tests/test_parser.py` is a regression suite over a labelled set.
* Context-aware colour words. "Black woman" is an ethnicity; "black blazer",
  "Black Friday" and "woman in black" are not. Same for "white".
* Honest about what it cannot do. Anything outside the supported vocabulary
  (Indigenous, Pacific Islander, mixed heritage, minors, groups, "athletic" body
  type ...) is returned in `unsupported`/`warnings` instead of being silently
  dropped or guessed.
* Ambiguity is surfaced, not resolved: "Asian" (South vs East), "brown", or two
  different ethnicities in one brief leave the field unset and explain why.
* Every value carries its evidence (matched text) and a confidence
  ("high" = explicit, "low" = inferred from a vague word such as "young").

Public API
----------
parse_brief_detailed(text, cfg=None) -> ParseResult
parse_brief(text, current, cfg, required_fields=None, **_ignored) -> dict
    Drop-in for the old signature; new explicit information overrides `current`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

HIGH, LOW = "high", "low"


@dataclass
class Match:
    value: str
    confidence: str          # "high" | "low"
    evidence: str            # text that triggered it


@dataclass
class ParseResult:
    fields: dict = field(default_factory=dict)        # name -> Match
    unsupported: list = field(default_factory=list)   # human-readable messages
    warnings: list = field(default_factory=list)
    ambiguous: dict = field(default_factory=dict)     # name -> [candidate values]

    def profile(self, include_low: bool = True) -> dict:
        return {k: m.value for k, m in self.fields.items()
                if include_low or m.confidence == HIGH}

    def to_dict(self) -> dict:
        return {
            "fields": {k: vars(m) for k, m in self.fields.items()},
            "unsupported": list(self.unsupported),
            "warnings": list(self.warnings),
            "ambiguous": dict(self.ambiguous),
        }


# ---------------------------------------------------------------------------
# Text normalisation / tokenisation
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("–", "-").replace("—", "-").replace("’", "'")
    return t.lower()


def _tokens(norm: str) -> list:
    """Word tokens (keeps internal hyphens/apostrophes), punctuation dropped."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", norm)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# canonical ethnicity -> terms (lower-case, multiword allowed)
ETHNICITY_TERMS = {
    "South Asian": [
        "south asian", "indian", "desi", "pakistani", "bangladeshi",
        "sri lankan", "nepali", "nepalese", "bhutanese",
    ],
    "East Asian": [
        "east asian", "chinese", "japanese", "korean", "taiwanese",
        "mongolian", "hong konger", "han chinese",
    ],
    "Southeast Asian": [
        "southeast asian", "south east asian", "south-east asian", "thai",
        "vietnamese", "filipino", "filipina", "indonesian", "malaysian",
        "malay", "singaporean", "burmese", "cambodian", "khmer", "laotian",
    ],
    "African American": [
        "african american", "african-american", "african", "nigerian",
        "ghanaian", "kenyan", "ethiopian", "somali", "jamaican",
        "afro-caribbean", "afro caribbean", "caribbean",
    ],
    "Latin American": [
        "latina", "latino", "latinx", "hispanic", "latin american",
        "mexican", "brazilian", "colombian", "argentinian", "argentine",
        "chilean", "peruvian", "venezuelan", "cuban", "puerto rican",
        "dominican",
    ],
    "Middle Eastern": [
        "middle eastern", "middle-eastern", "arab", "arabic", "arabian",
        "emirati", "saudi", "iranian", "persian", "lebanese", "iraqi",
        "syrian", "jordanian", "egyptian",
    ],
    "White": [
        "caucasian", "european", "british", "french", "german", "italian",
        "swedish", "russian", "scandinavian", "nordic", "irish", "polish",
        "dutch",
    ],
}

# colour words that are only an ethnicity in person context
COLOUR_ETHNICITY = {"black": "African American", "white": "White"}

# regional terms (imply South Asian)
REGION_TERMS = {
    "South Indian": ["south indian", "tamil", "telugu", "kannadiga", "malayali",
                     "keralite"],
    "North Indian": ["north indian", "rajasthani", "kashmiri", "haryanvi",
                     "delhiite"],
    "Bengali": ["bengali", "bangla"],
    "Punjabi": ["punjabi"],
    "Gujarati": ["gujarati"],
    "Maharashtrian": ["maharashtrian", "marathi"],
    "Northeast Indian": ["northeast indian", "north-east indian", "assamese",
                         "manipuri", "mizo", "naga"],
}

# words that, right after an ethnicity term, mean it is not about the person
NON_PERSON_AFTER_ETHNICITY = {
    "food", "foods", "cuisine", "restaurant", "restaurants", "dish", "dishes",
    "takeaway", "takeout", "cooking", "kitchen", "spices", "spice", "tea",
    "market", "newspaper", "language", "festival", "wedding", "print",
    "prints", "fabric", "fabrics", "pattern", "patterns", "inspired",
    "design", "designs", "textile", "textiles", "ocean", "summer"
}
# ... and words that mean the person-noun is a *product*, not a model
CLOTHING_OR_OBJECT = {
    "dress", "dresses", "blazer", "blazers", "suit", "suits", "saree", "sari",
    "kurta", "kurti", "shirt", "shirts", "top", "tops", "pants", "trousers",
    "jacket", "coat", "skirt", "gown", "outfit", "heels", "shoes", "sneakers",
    "boots", "bag", "handbag", "watch", "sunglasses", "glasses", "tie",
    "leather", "background", "backdrop", "friday", "cat", "lehenga", "scarf",
    "jeans", "sweater", "hoodie", "turban", "hijab", "veil", "blouse",
    "colour", "color", "tone", "tones", "theme", "palette", "and", "or",
    "wearing", "with", "in", "on", "for"
}
PERSON_NOUNS = {
    "woman", "women", "man", "men", "lady", "ladies", "girl", "girls", "boy",
    "boys", "female", "male", "model", "models", "person", "people", "guy",
    "gentleman", "american", "skin", "skinned", "aunty", "auntie", "uncle",
    "mother", "mom", "mum", "father", "dad", "customer", "customers",
    "professional", "influencer", "actress", "actor", "athlete", "creator",
}
# filler words that may sit between a colour word and the person noun
COLOUR_FILLERS = {
    "plus", "size", "plus-size", "plussize", "slim", "thin", "lean", "curvy",
    "young", "older", "old", "middle-aged", "mature", "tall", "short",
    "beautiful", "confident", "stylish", "elegant", "petite", "fuller", "medium",
    "average", "regular", "build", "20s", "30s", "40s", "50s", "60s",
}

UNSUPPORTED_ETHNICITY = [
    (r"\b(native american|american indian|red indian|indigenous|aboriginal|"
     r"first nations?|inuit|alaska native)\b",
     "Indigenous identities are not in the supported list."),
    (r"\b(pacific islander|polynesian|maori|hawaiian|samoan|tongan|fijian)\b",
     "Pacific Islander identities are not in the supported list."),
    (r"\b(mixed[- ]race|mixed[- ]heritage|biracial|bi-racial|multiracial|"
     r"multi-racial|half[- ](?:indian|chinese|black|white|asian|japanese|korean))\b",
     "Mixed-heritage briefs are not supported; pick one category."),
]

# garment/accessory words after slim/thin/lean => it describes the product
_CONNECTORS = {"and", "or", "wearing", "with", "in", "on", "for"}
PRODUCT_WORDS = (CLOTHING_OR_OBJECT - _CONNECTORS) | {
    "fit", "fitting", "cut", "chain", "chains", "strap", "straps", "band",
    "frame", "frames", "gold", "silver", "line", "lines", "crust", "layer",
    "layers", "sole", "heel", "bracelet", "necklace", "ring", "belt", "case",
    "screen", "film", "wallet", "phone", "laptop", "pendant", "profile",
}

WORD_NUM = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}
DECADE_WORD = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
               "sixties": 60, "seventies": 70, "teens": 10}

FEMALE_WORDS = {
    "woman", "women", "womens", "women's", "female", "females", "lady",
    "ladies", "girl", "girls", "she", "her", "hers", "aunty", "auntie",
    "mother", "mom", "mum", "mama", "wife", "daughter", "sister", "actress",
    "businesswoman", "bride", "grandmother", "grandma", "granny",
}
MALE_WORDS = {
    "man", "men", "mens", "men's", "male", "males", "gentleman", "gentlemen",
    "guy", "guys", "boy", "boys", "he", "his", "him", "uncle", "father", "dad",
    "husband", "son", "brother", "actor", "businessman", "groom",
    "grandfather", "grandpa",
}
LOW_CONF_GENDER = {"boy", "boys", "girl", "girls"}   # often used of adults

MINOR_PATTERN = re.compile(
    r"\b(child|children|kid|kids|toddler|baby|babies|infant|teen|teens|"
    r"teenager|teenagers|teenage|adolescent|minor|schoolgirl|schoolboy|"
    r"preteen|pre-teen|tween|little (?:girl|boy)|school[- ]?(?:kid|child|age))\b")
GROUP_PATTERN = re.compile(
    r"\b(couple|family|families|group|duo|friends|siblings|twins|crowd|"
    r"(?:two|three|four|five|2|3|4|5) (?:women|men|people|models|girls|boys))\b")
NONBINARY_PATTERN = re.compile(
    r"\b(non-?binary|gender[- ]?neutral|genderfluid|androgynous|unisex|agender)\b")

SLIM_TERMS = ["slim", "slender", "thin", "lean", "skinny", "svelte"]
PLUS_TERMS_RE = (r"plus[- ]?sized?|curvy|full[- ]figured|fuller|larger[- ]bodied|"
                 r"heavyset|voluptuous|big[- ]bodied|chubby|overweight|fat")
MEDIUM_TERMS_RE = (r"average[- ](?:build|size|body|figure|weight)|medium[- ](?:build|size|body|figure)|"
                   r"mid[- ]?size|regular[- ](?:build|size|body|figure)|normal[- ](?:build|body)|"
                   r"average|medium")
UNSUPPORTED_BODY = re.compile(
    r"\b(athletic|toned|muscular|bodybuilder|sporty build|pregnant|"
    r"hourglass|pear[- ]shaped|petite)\b")


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

def _find_term(norm: str, term: str):
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", norm)


def _colour_is_person(tokens: list, i: int) -> bool:
    """Is tokens[i] ('black'/'white') describing a person rather than a colour?

    Scan up to four following words: a person noun ("woman", "model", ...)
    => ethnicity; a clothing/object/connector word ("blazer", "friday", "and",
    "in" ...) => colour. Filler words (body/age adjectives) are skipped.
    """
    for j in range(i + 1, min(len(tokens), i + 5)):
        w = tokens[j]
        if w in PERSON_NOUNS:
            return True
        if w in CLOTHING_OR_OBJECT:
            return False
    return False


def parse_ethnicity(text: str, res: ParseResult) -> None:
    norm = _norm(text)
    tokens = _tokens(norm)

    for pat, msg in UNSUPPORTED_ETHNICITY:
        m = re.search(pat, norm)
        if m:
            res.unsupported.append(f"'{m.group(0)}': {msg}")
    # remove unsupported spans so e.g. "american indian" doesn't read as Indian
    scrub = norm
    for pat, _ in UNSUPPORTED_ETHNICITY:
        scrub = re.sub(pat, " ", scrub)

    found = {}  # canonical -> evidence
    region = None

    # explicit terms (longest first), skipping cuisine/object uses
    for canon, terms in ETHNICITY_TERMS.items():
        for term in sorted(terms, key=len, reverse=True):
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", scrub):
                after = _tokens(scrub[m.end(): m.end() + 40])[:1]
                if after and after[0] in NON_PERSON_AFTER_ETHNICITY:
                    continue
                # "black dress" style handled separately; here only real terms
                # nationality prefix like "british indian": the second wins later
                found.setdefault(canon, m.group(0))
    # "<x>-american" compounds: chinese american, indian american, ...
    # (the first term was already captured above, "american" alone is ignored)

    # colour words in person context
    for i, tok in enumerate(tokens):
        if tok in COLOUR_ETHNICITY and _colour_is_person(tokens, i):
            found.setdefault(COLOUR_ETHNICITY[tok], tok)

    # "british"/"european" modifier directly before another ethnicity term
    # (e.g. "british indian") describes heritage: drop the White reading.
    if "White" in found and len(found) > 1:
        if re.search(r"\b(british|canadian|australian)[- ]"
                     r"(indian|pakistani|chinese|korean|japanese|nigerian|"
                     r"bangladeshi|jamaican|arab)\b", scrub):
            del found["White"]

    # regions (imply South Asian)
    for reg, terms in REGION_TERMS.items():
        for term in sorted(terms, key=len, reverse=True):
            m = _find_term(scrub, term)
            if m:
                region = region or (reg, m.group(0))
                found.setdefault("South Asian", m.group(0))
                break

    # ambiguous words
    if re.search(r"(?<![a-z])asian(?![a-z])", scrub) and \
            "South Asian" not in found and "East Asian" not in found \
            and "Southeast Asian" not in found:
        res.ambiguous["ethnicity"] = ["South Asian", "East Asian", "Southeast Asian"]
        res.warnings.append("'Asian' is ambiguous - choose South, East or Southeast Asian.")
    if re.search(r"\bbrown(?:[- ]skinned)? (?:woman|women|man|men|girl|lady|model|person)\b", scrub):
        res.warnings.append("'brown' does not identify an ethnicity - choose one explicitly.")
        res.ambiguous.setdefault("ethnicity", [])

    if len(found) > 1:
        res.ambiguous["ethnicity"] = sorted(found)
        res.warnings.append(
            "Several ethnicities mentioned (" + ", ".join(sorted(found)) +
            ") - mixed or multi-person briefs aren't supported; pick one.")
        return
    if len(found) == 1:
        canon, ev = next(iter(found.items()))
        res.fields["ethnicity"] = Match(canon, HIGH, ev)
        if canon == "African American" and ev not in ("african american", "african-american", "black"):
            res.warnings.append(
                f"'{ev}' mapped to the 'African American' (Black) appearance category.")
        if region and canon == "South Asian":
            res.fields["region"] = Match(region[0], HIGH, region[1])


def parse_gender(text: str, res: ParseResult) -> None:
    norm = _norm(text)
    tokens = _tokens(norm)
    nb = NONBINARY_PATTERN.search(norm)
    if nb:
        res.unsupported.append(
            f"'{nb.group(0)}': only female and male presentation are supported.")
    grp = GROUP_PATTERN.search(norm)
    if grp:
        res.unsupported.append(
            f"'{grp.group(0)}': one person per ad is supported; describe a single model.")

    female = [t for t in tokens if t in FEMALE_WORDS]
    male = [t for t in tokens if t in MALE_WORDS]
    if female and male:
        res.ambiguous["gender"] = ["female", "male"]
        res.warnings.append("Both female and male terms appear - choose one gender.")
        return
    hits, val = (female, "female") if female else (male, "male")
    if not hits:
        return
    strong = [h for h in hits if h not in LOW_CONF_GENDER]
    res.fields["gender"] = Match(val, HIGH if strong else LOW, hits[0])


def _decade_from_number(n: int, ev: str, res: ParseResult, conf: str = HIGH) -> Optional[Match]:
    if n < 18:
        res.unsupported.append(f"'{ev}': minors are not supported.")
        return None
    if n < 20:
        res.unsupported.append(f"'{ev}': ages 18-19 are not supported; choose 20s.")
        return None
    if n >= 70:
        res.unsupported.append(f"'{ev}': ages 70+ are not supported (max 60s).")
        return None
    return Match(f"{(n // 10) * 10}s", conf, ev)


def parse_age(text: str, res: ParseResult) -> None:
    norm = _norm(text)
    mi = MINOR_PATTERN.search(norm)
    if mi:
        res.unsupported.append(f"'{mi.group(0)}': minors are not supported - "
                               "describe an adult model.")

    cands = []  # list of Match

    # decade numerals: 40s, 40's, "mid-40s", "late 20s"
    for m in re.finditer(r"(?<![0-9a-z])(?:(early|mid|late)[- ]?)?(\d0)'?s(?![a-z0-9])", norm):
        n = int(m.group(2))
        if n in (20, 30, 40, 50, 60):
            cands.append(Match(f"{n}s", HIGH, m.group(0)))
        elif n == 10 and re.search(r"\b(in|her|his|their)\b", norm[max(0, m.start() - 12): m.start()]):
            res.unsupported.append(f"'{m.group(0)}': minors are not supported.")
            return
        elif n >= 70 and re.search(r"\b(in|her|his|their|late|early|mid)\b", norm[max(0, m.start() - 12): m.start()]):
            res.unsupported.append(f"'{m.group(0)}': ages 70+ are not supported (max 60s).")
    # decade words: forties, mid-thirties
    for m in re.finditer(r"(?<![a-z])(?:(early|mid|late)[- ]?)?(twenties|thirties|forties|fifties|sixties|seventies|teens)(?![a-z])", norm):
        n = DECADE_WORD[m.group(2)]
        if n == 10:
            res.unsupported.append(f"'{m.group(0)}': minors are not supported.")
            return
        if n >= 70:
            res.unsupported.append(f"'{m.group(0)}': ages 70+ are not supported (max 60s).")
            continue
        cands.append(Match(f"{n}s", HIGH, m.group(0)))
    # twenty-something / 30-something
    for m in re.finditer(r"(?<![a-z0-9])(twenty|thirty|forty|fifty|sixty|\d0)[- ]?something(?![a-z])", norm):
        w = m.group(1)
        n = WORD_NUM.get(w) or int(w)
        cands.append(Match(f"{n}s", HIGH, m.group(0)))
    # exact ages: "35 year old", "aged 35", "35yo"
    for m in re.finditer(r"(?<![0-9a-z$])(\d{2})[- ]?(?:years?[- ]old|yrs?[- ]old|y/?o)(?![a-z])|(?<![a-z])(?:aged?|age:) ?(\d{2})(?![0-9a-z])", norm):
        n = int(m.group(1) or m.group(2))
        mt = _decade_from_number(n, m.group(0).strip(), res)
        if mt:
            cands.append(mt)
    # ranges: "30-40 years", "25 to 35"
    rng = re.search(r"(?<![0-9a-z])(\d{2})\s*(?:-|to)\s*(\d{2})\s*(?:years|yrs|yo|y/o)?", norm)
    if rng and 18 <= int(rng.group(1)) < int(rng.group(2)) <= 80 and \
            re.search(r"year|yrs|\bage|aged|\bin (?:her|his)", norm):
        lo, hi = int(rng.group(1)) // 10 * 10, int(rng.group(2)) // 10 * 10
        if lo != hi:
            res.ambiguous["age"] = [f"{lo}s", f"{hi}s"]
            res.warnings.append(f"Age range '{rng.group(0).strip()}' spans decades - pick one.")
            return

    # vague words (low confidence)
    if not cands:
        vague = [
            (r"\b(elderly|senior citizen|senior|grandmother|grandma|granny|grandfather|grandpa)\b", "60s"),
            (r"\bmiddle[- ]aged\b", "40s"),
            (r"\b(older|mature)\b(?! (?:look|style))", "50s"),
            (r"\b(young|youthful)\b(?! at heart)", "20s"),
        ]
        for pat, val in vague:
            m = re.search(pat, norm)
            if m:
                cands.append(Match(val, LOW, m.group(0)))
                break

    vals = sorted({c.value for c in cands})
    if len(vals) > 1:
        res.ambiguous["age"] = vals
        res.warnings.append("Several age groups mentioned (" + ", ".join(vals) + ") - pick one.")
        return
    if cands:
        # prefer a high-confidence candidate as evidence
        best = sorted(cands, key=lambda c: c.confidence != HIGH)[0]
        res.fields["age"] = best


def parse_body_type(text: str, res: ParseResult) -> None:
    norm = _norm(text)
    tokens = _tokens(norm)

    un = UNSUPPORTED_BODY.search(norm)
    if un:
        res.unsupported.append(
            f"body type '{un.group(0)}' isn't available (slim, medium or plus-size).")

    cands = []

    # clothing sizes
    for m in re.finditer(r"(?<![a-z0-9])size[- ]?(\d{1,2})(?![0-9a-z])", norm):
        n = int(m.group(1))
        if n <= 6:
            cands.append(Match("slim", HIGH, m.group(0)))
        elif n <= 12:
            cands.append(Match("medium", HIGH, m.group(0)))
        elif n <= 28:
            cands.append(Match("plus-size", HIGH, m.group(0)))
    for m in re.finditer(r"(?<![a-z0-9])(?:size[- ]?)?(xxl|xl|[123]x)(?![a-z0-9])", norm):
        cands.append(Match("plus-size", HIGH, m.group(0)))

    for m in re.finditer(rf"(?<![a-z0-9])({PLUS_TERMS_RE})(?![a-z0-9])", norm):
        low = m.group(1) in ("fat", "chubby", "overweight")
        cands.append(Match("plus-size", LOW if low else HIGH, m.group(0)))
    for m in re.finditer(rf"(?<![a-z0-9])({MEDIUM_TERMS_RE})(?![a-z0-9])", norm):
        bare = m.group(1) in ("average", "medium")
        if bare:
            # "medium black handbag" is a product size; "average woman" is a body
            nxt = _tokens(norm[m.end(): m.end() + 20])[:1]
            if not (nxt and (nxt[0] in PERSON_NOUNS or
                             nxt[0] in {"body", "build", "figure", "weight"})):
                continue
        cands.append(Match("medium", LOW if bare else HIGH, m.group(0)))

    for term in SLIM_TERMS:
        for m in re.finditer(rf"(?<![a-z0-9]){term}(?![a-z0-9])", norm):
            nxt = _tokens(norm[m.end(): m.end() + 24])[:1]
            if nxt and nxt[0] in PRODUCT_WORDS:
                continue                       # "slim fit", "thin gold chain"
            cands.append(Match("slim", HIGH, m.group(0)))

    vals = sorted({c.value for c in cands})
    if len(vals) > 1:
        res.ambiguous["body_type"] = vals
        res.warnings.append("Conflicting body types (" + ", ".join(vals) + ") - pick one.")
        return
    if cands:
        best = sorted(cands, key=lambda c: c.confidence != HIGH)[0]
        res.fields["body_type"] = best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_brief_detailed(text: str, cfg: dict = None) -> ParseResult:
    res = ParseResult()
    if not text or not text.strip():
        return res
    parse_ethnicity(text, res)
    parse_gender(text, res)
    parse_age(text, res)
    parse_body_type(text, res)
    if cfg:  # drop values outside the configured option lists
        opts = cfg.get("options", {})
        for k in list(res.fields):
            if k in opts and res.fields[k].value not in opts[k]:
                res.unsupported.append(
                    f"{k} '{res.fields[k].value}' is not enabled in this configuration.")
                del res.fields[k]
    return res


def parse_brief(text: str, current: dict, cfg: dict = None,
                required_fields: list = None, **_ignored) -> dict:
    """
    Backwards-compatible wrapper. Explicit information in `text` overrides
    what is already in `current` (a user correcting themselves wins); fields
    the text doesn't mention are left as they were.

    Extra keyword arguments (model=, host=) from the old Ollama-based
    signature are accepted and ignored.
    """
    res = parse_brief_detailed(text, cfg)
    merged = dict(current or {})
    merged.update(res.profile())
    if required_fields is not None:
        # keep optional keys (region) but never return fields outside schema
        allowed = set(required_fields) | {"region"}
        merged = {k: v for k, v in merged.items() if k in allowed}
    return merged
