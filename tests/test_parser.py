import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from eval.parser_accuracy import evaluate
from profiler.text_parser import parse_brief, parse_brief_detailed

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))


def test_labelled_cases_all_pass():
    stats, failures = evaluate(cfg=CFG)
    assert not failures, "\n".join(f"{f['brief']!r}: expected {f['expected']} got {f['got']} {f['notes']}" for f in failures)


def test_no_llm_dependency():
    import profiler.text_parser as tp
    src = open(tp.__file__).read()
    assert "import ollama" not in src and "import requests" not in src
    assert not hasattr(tp, "check_ollama")


def test_parse_brief_merge_and_override():
    cur = {"ethnicity": "East Asian", "age": "30s"}
    out = parse_brief("actually a Black woman", cur, CFG, ["ethnicity", "gender", "body_type", "age"])
    assert out["ethnicity"] == "African American" and out["age"] == "30s" and out["gender"] == "female"


def test_legacy_kwargs_ignored():
    out = parse_brief("slim Indian woman 30s", {}, CFG, ["ethnicity"], model="phi3:mini", host="x")
    assert out == {"ethnicity": "South Asian"}


def test_deterministic():
    a = parse_brief_detailed("Tamil woman in her mid-thirties, size 18").to_dict()
    b = parse_brief_detailed("Tamil woman in her mid-thirties, size 18").to_dict()
    assert a == b
