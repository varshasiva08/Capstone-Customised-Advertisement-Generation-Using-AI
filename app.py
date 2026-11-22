import io
import os
import random
import time

import streamlit as st
import yaml
from dotenv import load_dotenv
from PIL import Image as PILImage

from modules.correction_loop import run_loop
from modules.prompt_adapter import make_prompt_builder
from profiler.product_category import infer_category_from_text
from profiler.profile_gen_with_rag import generate_profiles
from profiler.prompt_builder import clean_scene
from profiler.text_parser import parse_brief_detailed

load_dotenv()


@st.cache_resource
def load_config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
OPT = CFG["options"]
REGIONS = list(CFG.get("regions", {}).get("South Asian", {}))
os.makedirs("outputs", exist_ok=True)

st.set_page_config(page_title="AdFidelity", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── App background ── */
.stApp { background: #0B0D17; }
section[data-testid="stSidebar"] { background: #10121E !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem; }

/* ── Animations ── */
@keyframes fadeUp   { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:translateY(0) } }
@keyframes fadeIn   { from { opacity:0 } to { opacity:1 } }
@keyframes glow     { 0%,100%{box-shadow:0 0 0 0 rgba(224,64,251,0)} 50%{box-shadow:0 0 0 8px rgba(224,64,251,0.10)} }
@keyframes shimmer  { 0%{background-position:-200% center} 100%{background-position:200% center} }

/* ── Sidebar wordmark ── */
.af-wordmark {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E040FB 0%, #B388FF 60%, #F5F0E8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 2px 0;
    display: inline-block;
}
.af-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3A3D5C;
    margin-bottom: 1.4rem;
}

/* ── Section label ── */
.af-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3A3D5C;
    margin: 1rem 0 0.35rem 0;
}

/* ── Profile chip ── */
.af-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(224,64,251,0.08);
    border: 1px solid rgba(224,64,251,0.2);
    border-radius: 100px;
    padding: 0.28rem 0.8rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    color: #C9A0FF;
    margin-bottom: 0.8rem;
    animation: fadeUp 0.3s ease both;
}
.af-chip-dot { width:5px; height:5px; border-radius:50%; background:#E040FB; flex-shrink:0; }

/* ── Step tracker ── */
.af-steps { display:flex; align-items:center; gap:0; margin-bottom:1.2rem; animation:fadeIn 0.4s 0.1s ease both; }
.af-step-item { display:flex; align-items:center; gap:0.3rem; }
.af-step-dot {
    width:24px; height:24px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.62rem; font-weight:700; font-family:'Inter',sans-serif;
    transition: all 0.3s ease;
}
.af-step-dot.done    { background:rgba(224,64,251,0.12); border:1.5px solid rgba(224,64,251,0.35); color:#E040FB; }
.af-step-dot.active  { background:#E040FB; border:1.5px solid #E040FB; color:#fff; animation:glow 2s ease infinite; }
.af-step-dot.pending { background:rgba(255,255,255,0.03); border:1.5px solid rgba(255,255,255,0.08); color:#2E3050; }
.af-step-label { font-family:'Inter',sans-serif; font-size:0.64rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; }
.af-step-label.done    { color:rgba(224,64,251,0.55); }
.af-step-label.active  { color:#F5F0E8; }
.af-step-label.pending { color:#252740; }
.af-step-line { width:24px; height:1.5px; margin:0 3px; }
.af-step-line.done    { background:rgba(224,64,251,0.25); }
.af-step-line.pending { background:rgba(255,255,255,0.05); }

/* ── Main header ── */
.af-main-header {
    padding: 0.6rem 0 1rem 0;
    animation: fadeUp 0.4s ease both;
}
.af-main-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E040FB 0%, #B388FF 60%, #F5F0E8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    display: inline-block;
}
.af-main-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3A3D5C;
    margin: 3px 0 0 2px;
}

/* ── Chat messages ── */
.stChatMessage {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.055) !important;
    border-radius: 14px !important;
    animation: fadeUp 0.25s ease both;
}
.stChatMessage[data-testid="chat-message-user"] {
    background: rgba(224,64,251,0.06) !important;
    border-color: rgba(224,64,251,0.14) !important;
}

/* ── Chat input ── */
.stChatInput > div {
    border-radius: 14px !important;
    border: 1px solid rgba(224,64,251,0.2) !important;
    background: rgba(255,255,255,0.035) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stChatInput > div:focus-within {
    border-color: rgba(224,64,251,0.55) !important;
    box-shadow: 0 0 0 3px rgba(224,64,251,0.07) !important;
}

/* ── Sidebar inputs ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
    color: #E8E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] input:focus {
    border-color: rgba(224,64,251,0.4) !important;
    box-shadow: 0 0 0 2px rgba(224,64,251,0.07) !important;
}

/* ── Selectbox ── */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
    color: #E8E8F0 !important;
}

/* ── Toggle ── */
.stToggle label { font-family:'Inter',sans-serif; font-size:0.85rem; color:#7A7DA8; }

/* ── Radio ── */
.stRadio label { font-family:'Inter',sans-serif; font-size:0.82rem; color:#7A7DA8; }

/* ── File uploader ── */
.stFileUploader {
    border: 1.5px dashed rgba(224,64,251,0.18) !important;
    border-radius: 10px !important;
    background: rgba(224,64,251,0.025) !important;
}

/* ── Primary button ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E040FB 0%, #B388FF 100%);
    background-size: 200% auto;
    border: none;
    border-radius: 11px;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    padding: 0.6rem 1.2rem;
    color: #fff;
    width: 100%;
    animation: shimmer 3s linear infinite;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(224,64,251,0.3);
}

/* ── Secondary button ── */
div.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 9px;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #5A5D80;
    transition: all 0.15s ease;
}
div.stButton > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.07);
    color: #E8E8F0;
}

/* ── Divider ── */
hr { border:none !important; border-top:1px solid rgba(255,255,255,0.05) !important; margin:1rem 0 !important; }

/* ── Status ── */
.stStatus {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* ── Info ── */
.stAlert {
    background: rgba(224,64,251,0.06) !important;
    border: 1px solid rgba(224,64,251,0.14) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
    color: #C9A0FF !important;
}

/* ── Images/video ── */
img   { border-radius: 10px; }
video { border-radius: 10px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(224,64,251,0.18); border-radius: 3px; }

/* ── Number input ── */
section[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
    color: #E8E8F0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
AXES = [("ethnicity", "Ethnicity"), ("gender", "Gender"), ("age", "Age"), ("body_type", "Body type")]
POLICY_LABELS = {
    "cpdc": "Adaptive correction (recommended)",
    "resample": "Simple retry (same prompt, new seed)",
    "vanilla": "Off - single generation",
}


@st.cache_resource(show_spinner="Loading face-attribute verifier (first run downloads weights)...")
def load_verifier():
    from scoring.cache import CachedScorer
    from scoring.registry import get_scorer
    return CachedScorer(get_scorer(CFG["scoring"]["verifier"], CFG))


def pills(report):
    """HTML pass/fail pills, one per axis."""
    out = []
    for key, label in AXES:
        ax = report.axes.get(key) if report else None
        if ax is None or not ax.scored:
            out.append(f'<span class="af-pill na" title="Not automatically verified">{label} · not auto-checked</span>')
        elif ax.passed:
            out.append(f'<span class="af-pill ok">{label} ✓ {ax.predicted}</span>')
        else:
            out.append(f'<span class="af-pill bad">{label} ✗ {ax.predicted or "?"} (want {ax.target})</span>')
    if report is not None and not report.face_found:
        out = ['<span class="af-pill bad">No face detected</span>']
    return " ".join(out)


def profile_summary(p):
    bits = [p.get("ethnicity"), p.get("gender"), p.get("age"), p.get("body_type"), p.get("region")]
    return " · ".join(b for b in bits if b) or "-"


ss = st.session_state
ss.setdefault("brief", "")
ss.setdefault("parsed", None)
ss.setdefault("suggestions", [])
ss.setdefault("result", None)
ss.setdefault("ctx", {})

st.markdown("""
<style>
.af-pill{display:inline-block;padding:3px 10px;margin:2px 3px 2px 0;border-radius:999px;font-size:.74rem;
  font-weight:600;border:1px solid rgba(255,255,255,.12)}
.af-pill.ok{background:rgba(46,204,113,.14);color:#6EE7A8;border-color:rgba(46,204,113,.4)}
.af-pill.bad{background:rgba(255,82,82,.12);color:#FF8A8A;border-color:rgba(255,82,82,.4)}
.af-pill.na{background:rgba(255,255,255,.05);color:#9A9CB5}
.af-note{font-size:.8rem;color:#9A9CB5;margin:.2rem 0 .6rem}
header[data-testid="stHeader"]{background:transparent !important}
.af-section-title{color:#F1F1F8 !important;font-size:1.15rem !important;margin:1.4rem 0 .5rem !important}
.af-label,.af-sub,.af-main-sub,[data-testid="stWidgetLabel"] p{color:#B4B6CE !important}
.stTextArea textarea,.stTextInput input{background:rgba(255,255,255,.05) !important;color:#ECECF6 !important;
  border:1px solid rgba(255,255,255,.14) !important}
[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.05) !important}
[data-testid="stFileUploaderDropzone"] *{color:#C9CBE0 !important}
.stButton>button{color:#F1F1F8}
[data-baseweb="textarea"],[data-baseweb="base-input"],[data-baseweb="input"]{background:rgba(255,255,255,.06) !important}
[data-baseweb="textarea"] textarea,[data-baseweb="base-input"] input{background:transparent !important;
  color:#ECECF6 !important;-webkit-text-fill-color:#ECECF6 !important}
[data-testid="stTextArea"] div,[data-testid="stTextArea"] textarea{background:#171A2B !important;
  color:#ECECF6 !important;-webkit-text-fill-color:#ECECF6 !important;border-color:rgba(255,255,255,.14) !important}
[data-testid="stTextInput"] div,[data-testid="stTextInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"] div{
  background:#171A2B !important;color:#ECECF6 !important;-webkit-text-fill-color:#ECECF6 !important}
[data-testid="stSelectbox"] svg{fill:#B4B6CE !important}
textarea::placeholder,input::placeholder{color:#8C8EA8 !important;-webkit-text-fill-color:#8C8EA8 !important}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR - creative + verification settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2 class="af-wordmark">AdFidelity</h2>', unsafe_allow_html=True)
    st.markdown('<p class="af-sub">Demographic-faithful ad studio</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p class="af-label">Brand name</p>', unsafe_allow_html=True)
    brand = st.text_input("Brand", placeholder="Tanishq, Zara, Fabindia...", label_visibility="collapsed") or "Brand"

    st.markdown('<p class="af-label">Clothing</p>', unsafe_allow_html=True)
    clothing = st.text_input("Clothing", placeholder="white blazer, red kurta, navy saree...",
                             label_visibility="collapsed") or "professional outfit"

    st.markdown('<p class="af-label">Background</p>', unsafe_allow_html=True)
    bg_mode = st.radio("BgMode", ["Colour", "Scene", "Custom"], horizontal=True, label_visibility="collapsed")
    scene = None
    background = CFG["defaults"]["background"]
    if bg_mode == "Scene":
        presets = CFG.get("background_presets", {})
        preset = st.selectbox("Scene", list(presets), label_visibility="collapsed")
        scene = presets.get(preset, "")
    elif bg_mode == "Custom":
        scene = st.text_area("Custom scene", placeholder="realistic luxury hotel lobby with cinematic lighting",
                             label_visibility="collapsed", max_chars=200)
    else:
        background = st.selectbox("Background", OPT["background"], label_visibility="collapsed")
    scene = clean_scene(scene) or None
    background_focus = st.selectbox("Background focus", ["Auto", "Blurred", "Sharp"])

    st.markdown('<p class="af-label">Product (optional)</p>', unsafe_allow_html=True)
    product_file = st.file_uploader("Product", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    product_text = st.text_input("Product name", placeholder="e.g. black leather handbag (optional)",
                                 label_visibility="collapsed")

    st.markdown('<p class="af-label">Tagline</p>', unsafe_allow_html=True)
    custom_tagline = st.text_input("Tagline", placeholder="Leave blank to auto-pick...", label_visibility="collapsed")

    st.divider()
    st.markdown('<p class="af-label">Verify & auto-correct</p>', unsafe_allow_html=True)
    policy = st.selectbox("Mode", list(POLICY_LABELS), format_func=POLICY_LABELS.get, label_visibility="collapsed",
                          help="After each image a face-attribute model checks ethnicity, gender and age against "
                               "your brief; failing axes are corrected in the next prompt.")
    max_calls = 1 if policy == "vanilla" else st.slider("Max attempts", 2, 6, CFG["loop"]["max_calls"])
    provider_choice = st.selectbox("Image source", ["Live (FLUX)", "Demo (sample images)"],
                                   help="Demo mode uses bundled sample photos - for trying the interface offline.")
    provider = "mock" if provider_choice.startswith("Demo") else None

    st.divider()
    animate_enabled = st.toggle("Generate video", value=False, help="Animates the final image.")
    if animate_enabled:
        motion_style = st.radio("Motion", ["sway", "walk"],
                                format_func=lambda x: "↔ Sway & turn" if x == "sway" else "→ Walk & pose")
        ad_duration = st.radio("Duration", [5, 10, 15, 20], format_func=lambda x: f"{x}s", horizontal=True)
    else:
        motion_style, ad_duration = "sway", 15

    st.divider()
    if st.button("↺ New brief"):
        for k in ("brief", "parsed", "suggestions", "result", "ctx"):
            ss.pop(k, None)
        for k in list(ss.keys()):
            if k.startswith("sel_"):
                ss.pop(k)
        st.rerun()

# ─────────────────────────────────────────────
# MAIN - 1) brief
# ─────────────────────────────────────────────
st.markdown("""
<div class="af-main-header">
  <h1 class="af-main-title">Ad Generation Studio</h1>
  <p class="af-main-sub">Say who the ad is for. AdFidelity checks the image and fixes it until it matches.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<h3 class="af-section-title">1 · Describe the person</h3>', unsafe_allow_html=True)
brief = st.text_area("Brief", value=ss.brief, height=90, label_visibility="collapsed",
                     placeholder="e.g. plus-size South Indian woman in her 40s, navy saree  ·  "
                                 "or  ·  sportswear brand targeting diverse women")
c1, c2, _ = st.columns([1, 1.3, 3])
if c1.button("Analyse brief", type="primary"):
    ss.brief = brief
    ss.parsed = parse_brief_detailed(brief, CFG)
    ss.suggestions = []
    for k in [k for k in ss.keys() if k.startswith("sel_")]:
        ss.pop(k)
    ss.result = None
    if not ss.parsed.fields:
        ss.suggestions = generate_profiles(brief, n=6, cfg=CFG)
if c2.button("Suggest 6 diverse profiles"):
    ss.brief = brief
    ss.suggestions = generate_profiles(brief, n=6, cfg=CFG, fixed=(ss.parsed.profile() if ss.parsed else None))

parsed = ss.parsed
if parsed is not None:
    chips = []
    for name, m in parsed.fields.items():
        tag = "" if m.confidence == "high" else " (unsure)"
        chips.append(f'<span class="af-pill {"ok" if m.confidence == "high" else "na"}" '
                     f'title="matched: {m.evidence}">{name}: {m.value}{tag}</span>')
    st.markdown(" ".join(chips) or '<span class="af-note">No demographic details found in the brief.</span>',
                unsafe_allow_html=True)
    for msg in parsed.unsupported:
        st.warning(msg)
    for msg in parsed.warnings:
        st.info(msg)
    for name, cands in parsed.ambiguous.items():
        st.info(f"“{name}” is ambiguous ({', '.join(cands)}) - please choose below.")

if ss.suggestions:
    st.markdown('<p class="af-note">Pick a profile to use:</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, p in enumerate(ss.suggestions):
        if cols[i % 3].button(f"{i+1}. {profile_summary(p)}", key=f"sug_{i}"):
            for f in ("ethnicity", "gender", "age", "body_type"):
                ss[f"sel_{f}"] = p.get(f)
            ss["sel_region"] = p.get("region") or "-"
            ss.suggestions = []
            st.rerun()

# ─────────────────────────────────────────────
# MAIN - 2) confirm profile
# ─────────────────────────────────────────────
st.markdown('<h3 class="af-section-title">2 · Confirm the profile</h3>', unsafe_allow_html=True)
base = parsed.profile() if parsed else {}


def pick(field, options, label):
    key = f"sel_{field}"
    if key not in ss:
        v = base.get(field) or CFG["defaults"].get(field)
        ss[key] = v if v in options else None
    if ss[key] not in options:
        ss[key] = None
    return st.selectbox(label, options, key=key, index=None, placeholder="choose...")


r = st.columns(5)
with r[0]:
    ethnicity = pick("ethnicity", OPT["ethnicity"], "Ethnicity")
with r[1]:
    gender = pick("gender", OPT["gender"], "Gender")
with r[2]:
    age = pick("age", OPT["age"], "Age")
with r[3]:
    body_type = pick("body_type", OPT["body_type"], "Body type")
with r[4]:
    if ethnicity == "South Asian":
        ss.setdefault("sel_region", base.get("region") if base.get("region") in REGIONS else "-")
        region = st.selectbox("Region (optional)", ["-"] + REGIONS, key="sel_region")
        region = None if region == "-" else region
    else:
        region = None
        st.selectbox("Region (optional)", ["-"], disabled=True, key="sel_region_off")

profile = {"ethnicity": ethnicity, "gender": gender, "age": age, "body_type": body_type}
if region:
    profile["region"] = region
missing = [k for k in ("ethnicity", "gender", "age", "body_type") if not profile.get(k)]
if missing:
    st.markdown(f'<p class="af-note">Still needed: {", ".join(missing)}</p>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="af-chip"><span class="af-chip-dot"></span>{profile_summary(profile)}</div>',
                unsafe_allow_html=True)
st.markdown('<p class="af-note">Ethnicity, gender and age are verified automatically on the result. '
            'Body type is used in the prompt but is not auto-verified.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN - 3) generate
# ─────────────────────────────────────────────
st.markdown('<h3 class="af-section-title">3 · Generate</h3>', unsafe_allow_html=True)
go = st.button("✦ Generate ad image", type="primary", disabled=bool(missing))

if go and not missing:
    ss.result = None
    product_description, product_category, product_img = None, "other", None
    if product_file is not None:
        from modules.product_describe import describe_product
        product_img = PILImage.open(product_file).convert("RGB")
        product_description = (product_text.strip() if product_text.strip()
                               else describe_product(product_img))
    elif product_text:
        product_description = product_text.strip()
    if product_description:
        product_category = infer_category_from_text(product_description) or "other"

    build = make_prompt_builder(clothing, background, CFG, scene=scene,
                                product_description=product_description,
                                product_category=product_category, background_focus=background_focus)
    from generator.adapter import make_generate_fn
    gen = make_generate_fn(CFG, provider, profile)
    run_cfg = {**CFG, "loop": {**CFG["loop"], "max_calls": max_calls}}
    gallery = st.container()
    tried = []

    try:
        verifier = load_verifier()
        with st.status("Generating and checking...", expanded=True) as status:
            def on_attempt(att):
                tried.append(att)
                status.write(f"Attempt {att.index + 1}: "
                             f"{'all checks passed ✓' if att.report.all_passed else 'needs correction'}")

            result = run_loop(profile, policy=policy, build_prompt_fn=build, generate_fn=gen,
                              verifier=verifier, cfg=run_cfg, max_calls=max_calls,
                              base_seed=random.randint(0, 99999), on_attempt=on_attempt)
            status.update(label=f"Done - {result.calls} generation(s), "
                                f"{'verified ✓' if result.best.report.all_passed else 'best available'}",
                          state="complete")
        ss.result = result
        ss.ctx = {"profile": profile, "clothing": clothing, "scene": scene, "brand": brand,
                  "demo": bool(getattr(gen, "is_mock", False))}
        result.best.image.save(f"outputs/final_{int(time.time())}.png")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        st.error(f"Generation failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())

result = ss.result
if result is not None:
    best = result.best
    if ss.ctx.get("demo"):
        st.info("Demo mode: images are bundled samples, not generated for this brief.")
    left, right = st.columns([1.1, 1])
    with left:
        st.image(best.image, caption=f"Selected: attempt {best.index + 1} of {result.calls}",
                 width='stretch')
        buf = io.BytesIO()
        best.image.save(buf, format="PNG")
        st.download_button("⬇ Download image", buf.getvalue(), "adfidelity.png", "image/png")
    with right:
        st.markdown(pills(best.report), unsafe_allow_html=True)
        if best.report.all_passed:
            st.success("Matches the brief on every automatically verified axis.")
        elif not best.report.face_found:
            st.warning("No face was detected, so nothing could be verified.")
        else:
            st.warning("Closest attempt found - not every axis matched. Try more attempts or adjust the brief.")
        rd = best.report.reading
        if rd.face_found:
            st.caption(f"Verifier estimate: age ≈ {rd.age:.0f}, "
                       f"{'woman' if rd.gender.get('Woman', 0) >= rd.gender.get('Man', 0) else 'man'}. "
                       "Automatic age and ethnicity estimates are approximate.")
            top = sorted(rd.race.items(), key=lambda kv: -kv[1])[:3]
            st.caption("Verifier ethnicity probabilities: " + ", ".join(f"{k} {v:.0f}%" for k, v in top)
                       + f" · face found at {rd.face_confidence:.2f} confidence")
    if result.calls > 1:
        st.markdown('<p class="af-note">All attempts</p>', unsafe_allow_html=True)
        cols = st.columns(min(result.calls, 4))
        for a in result.attempts:
            with cols[a.index % len(cols)]:
                st.image(a.image, width='stretch',
                         caption=f"#{a.index + 1}{' ★' if a.index == result.best_index else ''}")
                st.markdown(pills(a.report), unsafe_allow_html=True)
    with st.expander("Prompt trace"):
        for a in result.attempts:
            st.markdown(f"**Attempt {a.index + 1}** - corrections: `{a.applied or 'none'}`")
            st.code(a.prompt, language=None)

    # ─── Video / storyboard ───
    if animate_enabled:
        st.divider()
        if st.button("🎬 Animate this image"):
            try:
                from modules.ad_compositor import apply_ad_overlay, generate_tagline
                from modules.motion import animate_image
                with st.status("Animating...", expanded=True) as vs:
                    video_path = animate_image(best.image, config=CFG, profile=ss.ctx["profile"],
                                               clothing=ss.ctx["clothing"], motion_style=motion_style,
                                               scene=ss.ctx["scene"])
                    tagline = (custom_tagline.strip() if custom_tagline.strip()
                               else generate_tagline(ss.brief, ss.ctx["profile"], ss.ctx["brand"], cfg=CFG))
                    ad_path = video_path.replace(".mp4", "_ad.mp4")
                    apply_ad_overlay(video_path=video_path, brand_name=ss.ctx["brand"], tagline=tagline,
                                     out_path=ad_path, product_img=(PILImage.open(product_file) if product_file is not None else None),
                                     target_duration_sec=ad_duration)
                    vs.update(label="Video ready ✓", state="complete")
                st.video(ad_path)
            except Exception as e:                           # noqa: BLE001
                st.error(f"Animation failed: {type(e).__name__}: {e}")

        st.markdown('<h3 class="af-section-title">🎬 Storyboard ad (20s)</h3>', unsafe_allow_html=True)
        sb_ctx = st.text_input("Context / setting", placeholder="e.g. busy coffee shop, sunset beach, office...")
        if st.button("Generate 20s ad"):
            with st.status("Building storyboard ad...", expanded=True) as sts:
                try:
                    from modules.narrative import run_storyboard
                    ad_path = run_storyboard(ss.ctx["profile"], ss.ctx["clothing"], ss.ctx["brand"],
                                             context=sb_ctx.strip() or None, cfg=CFG,
                                             product_file=product_file, brief=ss.brief,
                                             target_duration=20, log=sts.write)
                    sts.update(label="Storyboard ad ready ✓", state="complete")
                    if ad_path and os.path.exists(ad_path):
                        st.video(ad_path)
                except Exception as e:                       # noqa: BLE001
                    sts.update(label="Storyboard failed")
                    st.error(f"{type(e).__name__}: {e}")
