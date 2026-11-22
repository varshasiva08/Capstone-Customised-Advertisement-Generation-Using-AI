"""
Labelled briefs for the deterministic parser.

IMPORTANT - what this set is and is not
---------------------------------------
Written by the same people who wrote the parser rules, so it is a REGRESSION
suite (it stops old bugs coming back and documents intended behaviour). It is
not an independent accuracy estimate. For a paper, have someone else write
~150 briefs blind (without reading text_parser.py) and score the parser on those.

Format: (brief, expected_profile, extras)
  expected_profile : exact dict of fields the parser should return (anything
                     not listed must be ABSENT - false positives count).
  extras           : optional dict with
                       "warn": substring expected in warnings/unsupported
                       "amb":  field expected in `ambiguous`
"""

from itertools import product

# ---------------------------------------------------------------------------
# A. Templated grid cases (easy; checks the vocabulary is wired correctly)
# ---------------------------------------------------------------------------
_ETH_WORD = {
    "South Asian": ["Indian", "South Asian", "Pakistani"],
    "East Asian": ["Chinese", "Japanese", "Korean"],
    "Southeast Asian": ["Thai", "Vietnamese", "Filipina"],
    "African American": ["African American", "Nigerian", "Black"],
    "Latin American": ["Latina", "Mexican", "Hispanic"],
    "Middle Eastern": ["Arab", "Iranian", "Lebanese"],
    "White": ["white", "Caucasian", "French"],
}
_AGE_PHRASE = {"20s": ["in her 20s", "late twenties"],
               "30s": ["in her 30s", "35 years old"],
               "40s": ["in her forties", "mid-40s"],
               "50s": ["in her 50s", "fifty-something"],
               "60s": ["in her 60s", "aged 64"]}
_BODY_PHRASE = {"slim": ["slim", "slender"], "medium": ["average build", "medium build"],
                "plus-size": ["plus-size", "curvy"]}

TEMPLATED = []
_i = 0
for eth, age, body in product(_ETH_WORD, _AGE_PHRASE, _BODY_PHRASE):
    eword = _ETH_WORD[eth][_i % 3]
    aphrase = _AGE_PHRASE[age][_i % 2]
    bphrase = _BODY_PHRASE[body][_i % 2]
    _i += 1
    if _i % 3:            # keep ~1 in 3 of the 105 combinations
        continue
    # "Black"/"white" need a person noun to count as ethnicity
    brief = f"{bphrase} {eword} woman {aphrase}, jewellery campaign"
    gender = "female"
    if _i % 2 == 0:
        brief = f"{bphrase} {eword} man {aphrase}, watch campaign"
        gender = "male"
    TEMPLATED.append((brief,
                      {"ethnicity": eth, "gender": gender, "age": age, "body_type": body},
                      {}))

# ---------------------------------------------------------------------------
# B. Hand-written cases, including every failure found in the old parser
# ---------------------------------------------------------------------------
P = lambda **k: k   # noqa: E731  (tiny helper for readable dicts)

HAND = [
    # --- colour words: person vs garment ---------------------------------
    ("young Indian woman in a black dress for a jewellery ad",
     P(ethnicity="South Asian", gender="female", age="20s"), {}),
    ("a woman in her 30s wearing a black blazer",
     P(gender="female", age="30s"), {}),
    ("20-something Black Friday campaign model", P(age="20s"), {}),
    ("Black Friday sale, woman in her 40s",
     P(gender="female", age="40s"), {}),
    ("Black woman in her forties, curvy",
     P(ethnicity="African American", gender="female", age="40s", body_type="plus-size"), {}),
    ("a Black model, plus size, 30s",
     P(ethnicity="African American", age="30s", body_type="plus-size"), {}),
    ("white blazer and black trousers on a woman in her 30s",
     P(gender="female", age="30s"), {}),
    ("white woman in a white saree, 50s",
     P(ethnicity="White", gender="female", age="50s"), {}),
    ("woman in black, early 20s", P(gender="female", age="20s"), {}),
    ("black and white striped suit for women",
     P(gender="female"), {}),
    ("Black woman wearing a black leather jacket",
     P(ethnicity="African American", gender="female"), {}),
    ("white background, black font, ad for a plus size woman",
     P(gender="female", body_type="plus-size"), {}),
    ("white man in his 50s", P(ethnicity="White", gender="male", age="50s"), {}),
    ("a white handbag held by a woman in her 20s",
     P(gender="female", age="20s"), {}),

    # --- unsupported / ambiguous ethnicity --------------------------------
    ("an older Latina woman",
     P(ethnicity="Latin American", gender="female", age="50s"), {}),
    ("Hispanic man in his 30s", P(ethnicity="Latin American", gender="male", age="30s"), {}),
    ("Asian woman in her 30s", P(gender="female", age="30s"), {"amb": "ethnicity"}),
    ("Native American woman in her 40s", P(gender="female", age="40s"),
     {"warn": "Indigenous"}),
    ("American Indian elder", P(), {"warn": "Indigenous"}),
    ("mixed race woman, 30s", P(gender="female", age="30s"), {"warn": "Mixed-heritage"}),
    ("Polynesian man in his 20s", P(gender="male", age="20s"), {"warn": "Pacific"}),
    ("Indian or Chinese woman in her 30s", P(gender="female", age="30s"), {"amb": "ethnicity"}),
    ("brown woman in her 40s", P(gender="female", age="40s"), {"warn": "brown"}),
    ("Chinese-American woman, mid 30s", P(ethnicity="East Asian", gender="female", age="30s"), {}),
    ("British Indian woman in her 30s",
     P(ethnicity="South Asian", gender="female", age="30s"), {}),
    ("Indian restaurant ad with a woman in her 50s",
     P(gender="female", age="50s"), {}),
    ("African print dress, woman in her 20s", P(gender="female", age="20s"), {}),
    ("Nigerian woman in her thirties",
     P(ethnicity="African American", gender="female", age="30s"), {"warn": "mapped"}),
    ("dusky Indian woman in her 40s",
     P(ethnicity="South Asian", gender="female", age="40s"), {}),

    # --- regions ----------------------------------------------------------
    ("Tamil woman in her mid-thirties",
     P(ethnicity="South Asian", region="South Indian", gender="female", age="30s"), {}),
    ("south indian lady in her 50s, saree",
     P(ethnicity="South Asian", region="South Indian", gender="female", age="50s"), {}),
    ("Punjabi man in his forties",
     P(ethnicity="South Asian", region="Punjabi", gender="male", age="40s"), {}),
    ("Bengali woman, 30s, plus size",
     P(ethnicity="South Asian", region="Bengali", gender="female", age="30s", body_type="plus-size"), {}),
    ("campaign shot in Kerala with a woman in her 30s",
     P(gender="female", age="30s"), {}),
    ("Malayali woman in her 20s",
     P(ethnicity="South Asian", region="South Indian", gender="female", age="20s"), {}),
    ("Pakistani woman, forties", P(ethnicity="South Asian", gender="female", age="40s"), {}),

    # --- age phrasing -----------------------------------------------------
    ("woman in her late 20s", P(gender="female", age="20s"), {}),
    ("thirty-something Korean model, size 18",
     P(ethnicity="East Asian", age="30s", body_type="plus-size"), {}),
    ("a 45 year old man", P(gender="male", age="40s"), {}),
    ("35-year-old woman", P(gender="female", age="30s"), {}),
    ("aged 52, female", P(gender="female", age="50s"), {}),
    ("she is 28yo", P(gender="female", age="20s"), {}),
    ("woman, 30 to 40 years old", P(gender="female"), {"amb": "age"}),
    ("woman in her 20s and 30s", P(gender="female"), {"amb": "age"}),
    ("woman in her 70s", P(gender="female"), {"warn": "70+"}),
    ("teenage girl, 16 years old", P(gender="female"), {"warn": "minors"}),
    ("kids clothing ad with a woman in her 30s",
     P(gender="female", age="30s"), {"warn": "minors"}),
    ("18 year old woman", P(gender="female"), {"warn": "18-19"}),
    ("1990s retro fashion, woman in her 40s", P(gender="female", age="40s"), {}),
    ("80s style campaign for men", P(gender="male"), {}),
    ("middle-aged man", P(gender="male", age="40s"), {}),
    ("elderly woman", P(gender="female", age="60s"), {}),
    ("young woman", P(gender="female", age="20s"), {}),
    ("older woman", P(gender="female", age="50s"), {}),
    ("woman in her sixties", P(gender="female", age="60s"), {}),
    ("40's woman", P(gender="female", age="40s"), {}),
    ("woman, early thirties", P(gender="female", age="30s"), {}),
    ("woman size 12 in her 40s", P(gender="female", age="40s", body_type="medium"), {}),
    ("Sale: up to 50% off, woman in her 30s", P(gender="female", age="30s"), {}),

    # --- body type --------------------------------------------------------
    ("plus size Indian woman in her 40s, navy suit",
     P(ethnicity="South Asian", gender="female", age="40s", body_type="plus-size"), {}),
    ("slim fit blazer ad, woman in her 30s", P(gender="female", age="30s"), {}),
    ("woman with a thin gold chain necklace", P(gender="female"), {}),
    ("slim woman in her 20s", P(gender="female", age="20s", body_type="slim"), {}),
    ("lean man, 30s", P(gender="male", age="30s", body_type="slim"), {}),
    ("fat aunty, 50s, desi",
     P(ethnicity="South Asian", gender="female", age="50s", body_type="plus-size"), {}),
    ("regular customers, a woman in her 30s",
     P(gender="female", age="30s"), {}),
    ("woman with a regular build in her 30s",
     P(gender="female", age="30s", body_type="medium"), {}),
    ("average woman, 40s", P(gender="female", age="40s", body_type="medium"), {}),
    ("medium black leather handbag held by a woman",
     P(gender="female"), {}),
    ("size 6 model, 20s", P(age="20s", body_type="slim"), {}),
    ("size 10 woman", P(gender="female", body_type="medium"), {}),
    ("size 22 woman in her 30s", P(gender="female", age="30s", body_type="plus-size"), {}),
    ("XXL model, man in his 40s", P(gender="male", age="40s", body_type="plus-size"), {}),
    ("athletic woman in her 20s", P(gender="female", age="20s"), {"warn": "athletic"}),
    ("petite woman, 30s", P(gender="female", age="30s"), {"warn": "petite"}),
    ("slim and plus-size women", P(gender="female"), {"amb": "body_type"}),
    ("curvy woman", P(gender="female", body_type="plus-size"), {}),
    ("plus-size clothing brand, Indian woman in her 30s",
     P(ethnicity="South Asian", gender="female", age="30s", body_type="plus-size"), {}),

    # --- gender -----------------------------------------------------------
    ("men's grooming brand, man in his 30s", P(gender="male", age="30s"), {}),
    ("for men and women", P(), {"amb": "gender"}),
    ("non-binary model in their 20s", P(age="20s"), {"warn": "female and male"}),
    ("a couple in their 30s", P(age="30s"), {"warn": "one person"}),
    ("gentleman in his 60s, Arab", P(ethnicity="Middle Eastern", gender="male", age="60s"), {}),
    ("lady in her 50s", P(gender="female", age="50s"), {}),
    ("guy in his 20s", P(gender="male", age="20s"), {}),
    ("girl in her twenties", P(gender="female", age="20s"), {}),
    ("uncle in his fifties, Indian", P(ethnicity="South Asian", gender="male", age="50s"), {}),

    # --- vague / no information ------------------------------------------
    ("sportswear brand targeting diverse women", P(gender="female"), {}),
    ("luxury watch brand", P(), {}),
    ("", P(), {}),
    ("   ", P(), {}),
    ("diverse models", P(), {}),
    ("Chinese food delivery ad", P(), {}),
    ("summer collection launch", P(), {}),
    ("ethnic wear for South Asian women in their 30s",
     P(ethnicity="South Asian", gender="female", age="30s"), {}),

    # --- stress / formatting ---------------------------------------------
    ("INDIAN WOMAN IN HER 40S", P(ethnicity="South Asian", gender="female", age="40s"), {}),
    ("Indian—woman—40s", P(ethnicity="South Asian", gender="female", age="40s"), {}),
    ("indian woman (40s), slim.", P(ethnicity="South Asian", gender="female", age="40s", body_type="slim"), {}),
    ("Korean woman, 30s; Korean man, 30s", P(ethnicity="East Asian", age="30s"), {"amb": "gender"}),
    ("woman 30s Indian Chinese", P(gender="female", age="30s"), {"amb": "ethnicity"}),
]

ALL_CASES = [("templated",) + c for c in TEMPLATED] + [("hand",) + c for c in HAND]
