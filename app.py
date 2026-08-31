import os
import re
import time
import streamlit as st
import yaml
from dotenv import load_dotenv

from profiler.profile_gen import generate_profiles
from profiler.prompt_builder import build_prompt
from profiler.text_parser import check_ollama, parse_brief

# ── RAG import (graceful fallback if not installed yet) ──────────────────────
try:
    from rag.brief_rag import BriefRAG
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

load_dotenv()

@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_rag():
    """Load RAG store once — cached so it persists across reruns."""
    if _RAG_AVAILABLE:
        return BriefRAG()
    return None

CFG      = load_config()
REQUIRED = CFG["required_fields"]

OLLAMA_CFG   = CFG.get("ollama", {})
OLLAMA_HOST  = OLLAMA_CFG.get("host", "http://localhost:11434")
OLLAMA_MODEL = OLLAMA_CFG.get("model", "phi3:mini")

rag = load_rag()

os.makedirs("outputs", exist_ok=True)

st.set_page_config(
    page_title="AdFidelity",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    check_ollama(OLLAMA_HOST)
except RuntimeError as e:
    st.error(f"**Ollama is not running.**\n\n{e}\n\nPlease start Ollama before using AdFidelity.")
    st.stop()

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

/* ── RAG badge ── */
.af-rag-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(179,136,255,0.08);
    border: 1px solid rgba(179,136,255,0.2);
    border-radius: 100px;
    padding: 0.22rem 0.7rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    color: #B388FF;
    margin-bottom: 0.6rem;
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
PRODUCT_QUESTION = "Would you like to add a **product** to this ad? Reply **yes** or **no**."
YES_WORDS = {"yes","yeah","yep","yup","y","sure","ok","okay"}
NO_WORDS  = {"no","nope","nah","n","skip","none"}

PRODUCT_KEYWORDS = {
    "handbag":    ["handbag","bag","purse","tote","clutch","satchel"],
    "sunglasses": ["sunglasses","glasses","eyewear","shades"],
    "jewelry":    ["jewellery","jewelry","necklace","earrings","bracelet",
                   "ring","pendant","chain","anklet","bangles","bangle"],
    "clothing":   ["saree","sari","kurta","kurti","lehenga","salwar",
                   "dress","outfit","suit","gown","top","shirt","blouse"],
    "other":      ["watch","scarf","belt","shoes","heels","sandals",
                   "cap","hat","accessory","accessories"],
}

def detect_product_in_brief(text):
    lower = text.lower()
    for cat, kws in PRODUCT_KEYWORDS.items():
        for kw in kws:
            if kw in lower:
                return cat, kw
    return None, None

def missing_fields(profile):
    return [f for f in REQUIRED if not profile.get(f)]

def parse_yes_no(text):
    words = re.findall(r"[a-z]+", text.lower())
    if any(w in YES_WORDS for w in words): return True
    if any(w in NO_WORDS  for w in words): return False
    return None

def stage_index(stage):
    return {"collecting":0,"pick_profile":1,"ask_product":2,"ready":3}.get(stage,0)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
ss = st.session_state
ss.setdefault("messages", [{"role":"assistant","content":(
    "Describe your brand or advertisement — e.g. "
    "*'sportswear brand targeting diverse women'* or "
    "*'plus size Indian woman in her 40s, navy suit'*."
)}])
ss.setdefault("profile",  {})
ss.setdefault("profiles", [])
ss.setdefault("mode",     None)
ss.setdefault("stage",    "collecting")
ss.setdefault("want_product", None)
ss.setdefault("product_category_detected", None)
ss.setdefault("original_brief", "")
ss.setdefault("seed", 42)
ss.setdefault("brand_name", "")
ss.setdefault("rag_context", "")          # ← NEW: stores RAG context for current brief
ss.setdefault("last_fidelity_score", None) # ← NEW: stores score to save back to RAG

# ─────────────────────────────────────────────
# SIDEBAR — controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2 class="af-wordmark">AdFidelity</h2>', unsafe_allow_html=True)
    st.markdown('<p class="af-sub">Ad Generation Studio</p>', unsafe_allow_html=True)

    # RAG status badge
    if _RAG_AVAILABLE and rag is not None:
        count = rag.count()
        st.markdown(
            f'<div class="af-rag-badge">🧠 RAG · {count} past campaign{"s" if count != 1 else ""}</div>',
            unsafe_allow_html=True
        )

    # Step tracker
    steps   = [("Brief","collecting"),("Profile","pick_profile"),("Product","ask_product"),("Generate","ready")]
    current = stage_index(ss.stage)
    pills   = '<div class="af-steps">'
    for i,(label,_) in enumerate(steps):
        cls  = "done" if i < current else ("active" if i == current else "pending")
        icon = "✓"    if i < current else str(i+1)
        pills += f'<div class="af-step-item"><div class="af-step-dot {cls}">{icon}</div><span class="af-step-label {cls}">{label}</span></div>'
        if i < len(steps)-1:
            pills += f'<div class="af-step-line {"done" if i < current else "pending"}"></div>'
    pills += '</div>'
    st.markdown(pills, unsafe_allow_html=True)

    # Profile chip
    if ss.stage in ("ask_product","ready","pick_profile") and ss.profile and len(ss.messages) > 2:
        p = ss.profile
        st.markdown(
            f'<div class="af-chip"><span class="af-chip-dot"></span>'
            f'{p.get("ethnicity","—")} · {p.get("body_type","—")} · {p.get("age","—")}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    if ss.stage == "ready":
        # Brand
        st.markdown('<p class="af-label">Brand name</p>', unsafe_allow_html=True)
        brand_input = st.text_input("Brand", placeholder="Tanishq, Zara, Fabindia...", label_visibility="collapsed")
        ss.brand_name = brand_input if brand_input else "Brand"

        # Ad Style
        st.markdown('<p class="af-label">Clothing</p>', unsafe_allow_html=True)
        clothing = st.text_input("Clothing", placeholder="white blazer, red kurta, navy saree...", label_visibility="collapsed")
        if not clothing:
            clothing = "professional outfit"

        st.markdown('<p class="af-label">Background</p>', unsafe_allow_html=True)
        background = st.selectbox("Background", CFG["options"]["background"], label_visibility="collapsed")

        st.markdown('<p class="af-label">Seed</p>', unsafe_allow_html=True)
        seed = st.number_input("Seed", 0, 999999, ss.seed, label_visibility="collapsed",
                               help="Fixed seed for reproducible outputs.")
        ss.seed = seed

        # Product
        product_file     = None
        product_category = ss.get("product_category_detected") or "handbag"
        if ss.want_product:
            st.markdown('<p class="af-label">Product image</p>', unsafe_allow_html=True)
            product_file = st.file_uploader("Product", type=["png","jpg"], label_visibility="collapsed")

        # Tagline
        st.markdown('<p class="af-label">Tagline</p>', unsafe_allow_html=True)
        custom_tagline = st.text_input("Tagline", placeholder="Leave blank to auto-generate...", label_visibility="collapsed")

        # Animation
        st.divider()
        st.markdown('<p class="af-label">Animation</p>', unsafe_allow_html=True)
        animate_enabled = st.toggle("Generate video", value=False,
                                    help="Animates via Wan2.1-I2V — uses HF credits.")
        if animate_enabled:
            motion_style = st.radio("Motion", ["sway","walk"],
                                    format_func=lambda x: "↔ Sway & turn" if x=="sway" else "→ Walk & pose")
            ad_duration  = st.radio("Duration", [5,10,15,20],
                                    format_func=lambda x: f"{x}s", horizontal=True)
        else:
            motion_style = "sway"
            ad_duration  = 15

        st.divider()

        # Generate button
        if not missing_fields(ss.profile):
            generate_clicked = st.button("✦ Generate Ad", type="primary")
        else:
            st.info("Complete the profile in chat to unlock.")
            generate_clicked = False

    else:
        # Placeholders so variables exist
        clothing = "professional outfit"
        background = CFG["options"]["background"][0] if CFG["options"]["background"] else "white"
        seed = ss.seed
        product_file = None
        product_category = "handbag"
        custom_tagline = ""
        animate_enabled = False
        motion_style = "sway"
        ad_duration = 15
        generate_clicked = False

    st.divider()
    st.button("↺ New brief", on_click=lambda: ss.update({
        "profile":{}, "profiles":[], "mode":None,
        "stage":"collecting", "want_product":None,
        "original_brief":"", "product_category_detected":None,
        "brand_name":"",
        "rag_context":"",
        "last_fidelity_score": None,
        "messages": ss.messages[:1]
    }))

# ─────────────────────────────────────────────
# MAIN AREA — header + chat
# ─────────────────────────────────────────────
st.markdown("""
<div class="af-main-header">
  <h1 class="af-main-title">Ad Generation Studio</h1>
  <p class="af-main-sub">Demographic-faithful · AI-powered · End-to-end</p>
</div>
""", unsafe_allow_html=True)

# Chat history
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ─────────────────────────────────────────────
# CHAT INPUT + LOGIC
# ─────────────────────────────────────────────
if user_input := st.chat_input("Describe your brand or ad..."):
    ss.messages.append({"role":"user","content":user_input})
    ss.original_brief = ss.get("original_brief","") + " " + user_input

    if ss.stage == "pick_profile":
        nums = re.findall(r"\b([1-6])\b", user_input)
        if nums:
            chosen = int(nums[0]) - 1
            ss.profile = ss.profiles[chosen]
            ss.mode    = "single"
            ss.stage   = "ask_product"
            p = ss.profile
            ss.messages.append({"role":"assistant","content":
                f"Going with **Profile {chosen+1}**: {p['ethnicity']}, {p['body_type']}, {p['age']}.\n\n" + PRODUCT_QUESTION})
        else:
            ss.profile = parse_brief(user_input, {}, CFG, REQUIRED, model=OLLAMA_MODEL, host=OLLAMA_HOST)
            missing = missing_fields(ss.profile)
            if not missing:
                ss.mode  = "single"
                ss.stage = "ask_product"
                p = ss.profile
                ss.messages.append({"role":"assistant","content":
                    f"Got it — **{p['ethnicity']}, {p['body_type']}, {p['age']}**.\n\n" + PRODUCT_QUESTION})
            else:
                ss.messages.append({"role":"assistant","content":
                    "Reply with a number **1–6** to pick a profile, or describe the demographic more specifically."})

    elif ss.stage == "ask_product":
        answer = parse_yes_no(user_input)
        if answer is True:
            ss.want_product = True
            ss.stage = "ready"
            ss.messages.append({"role":"assistant","content":
                "Got it — upload your product image in the panel on the left, then hit **Generate**."})
        elif answer is False:
            ss.want_product = False
            ss.stage = "ready"
            ss.messages.append({"role":"assistant","content":
                "No product — fill in the details on the left and hit **Generate**."})
        else:
            ss.messages.append({"role":"assistant","content":
                "Just need a **yes** or **no** — " + PRODUCT_QUESTION})

    else:
        existing = ss.profile if len(ss.messages) > 2 else {}
        ss.profile = parse_brief(user_input, existing, CFG, REQUIRED, model=OLLAMA_MODEL, host=OLLAMA_HOST)
        missing = missing_fields(ss.profile)

        if not missing:
            ss.mode = "single"
            detected_cat, detected_kw = detect_product_in_brief(ss.get("original_brief",""))
            if detected_cat:
                ss.want_product = True
                ss.product_category_detected = detected_cat
                ss.stage = "ready"
                ss.messages.append({"role":"assistant","content":
                    f"Got it — **{ss.profile['body_type']} {ss.profile['ethnicity']} woman in her {ss.profile['age']}** with **{detected_kw}**.\n\n"
                    "Upload your product image on the left and hit **Generate**."})
            else:
                ss.stage = "ask_product"
                ss.messages.append({"role":"assistant","content":
                    f"Got it — **{ss.profile['body_type']} {ss.profile['ethnicity']} woman in her {ss.profile['age']}**.\n\n" + PRODUCT_QUESTION})

        elif len(missing) == 3:
            # ── RAG: retrieve context before generating profiles ──────────────
            ss.mode = "multi"
            brief_text = ss.get("original_brief", user_input)

            rag_context = ""
            rag_note    = ""
            if _RAG_AVAILABLE and rag is not None and rag.count() >= 2:
                rag_context = rag.retrieve(brief_text)
                ss.rag_context = rag_context
                rag_note = "\n\n> 🧠 *Using similar past campaigns as context.*"

            with st.spinner("Generating profiles..."):
                try:
                    ss.profiles = generate_profiles(
                        user_input,
                        model=OLLAMA_MODEL,
                        host=OLLAMA_HOST,
                        rag_context=rag_context      # ← RAG context passed here
                    )
                    reply = "Generated **6 demographic profiles** for your brand:\n\n"
                    for i,p in enumerate(ss.profiles,1):
                        reply += f"**{i}.** {p['ethnicity']}, {p['body_type']}, {p['age']}\n"
                    reply += "\nWhich profile? Reply **1–6**, or describe a different demographic."
                    reply += rag_note
                    ss.stage = "pick_profile"
                    ss.messages.append({"role":"assistant","content":reply})
                except Exception as e:
                    ss.messages.append({"role":"assistant","content":f"Profile generation failed: {e}"})
        else:
            ss.mode = "single"
            questions = []
            for field in missing:
                opts  = ", ".join(CFG["options"][field])
                label = {"ethnicity":"Ethnicity","body_type":"Body type","age":"Age group"}[field]
                questions.append(f"- **{label}**: {opts}")
            got   = {k:v for k,v in ss.profile.items() if k in REQUIRED and v}
            intro = ("Got so far: " + ", ".join(f"**{v}**" for v in got.values()) + ".\n\n" if got else "")
            ss.messages.append({"role":"assistant","content":
                intro + "Just need a few more details:\n\n" + "\n".join(questions)})

    st.rerun()

# ─────────────────────────────────────────────
# OUTPUT — render after generate clicked
# ─────────────────────────────────────────────
if generate_clicked and ss.stage == "ready" and not missing_fields(ss.profile):
    st.divider()
    out = st.container()

    from generator.flux_pipeline import generate_image
    from modules.product_describe import describe_product

    try:
        with out.status("Generating image...", expanded=True) as status:
            product_description = None
            if ss.want_product and product_file is not None:
                status.write("👁️ Identifying product...")
                from PIL import Image as PILImage
                product_img = PILImage.open(product_file)
                product_description = describe_product(product_img)
                status.write(f"🛍️ *{product_description}*")

            prompt = build_prompt(
                ss.profile, clothing, background, CFG,
                product_description=product_description,
                product_category=product_category
            )
            status.write("⚙️ Calling FLUX.1-schnell...")
            img, device, res, steps_used = generate_image(prompt, int(seed), CFG)
            status.update(label="Image ready ✓", state="complete")

        out_path = f"outputs/output_seed{seed}.png"
        img.save(out_path)

        # Show image
        c_img, c_vid = out.columns(2) if animate_enabled else (out, None)
        c_img.image(img, caption=f"{device.upper()} · {res[0]}×{res[1]} · seed {seed}", use_container_width=True)

        # ── RAG: store this run after successful generation ───────────────────
        # We store with a default score of 7.0 if no CLIP scorer is available.
        # If your teammate runs clip_fidelity_scorer.py separately and gets a
        # score, they can update the store manually or integrate the scorer here.
        if _RAG_AVAILABLE and rag is not None:
            fidelity_score = ss.get("last_fidelity_score") or 7.0
            rag.store(
                brief=ss.get("original_brief", ""),
                profile=ss.profile,
                fidelity_score=fidelity_score
            )

        if animate_enabled:
            from modules.motion import animate_image
            from modules.ad_compositor import apply_ad_overlay, generate_tagline
            try:
                with out.status("Animating via Wan2.1-I2V...", expanded=True) as vs:
                    vs.write(f"🎬 Style: {'Sway & turn' if motion_style=='sway' else 'Walk & pose'}")
                    vs.write("🚀 Calling Wan2.1-I2V — ~60–120 s...")
                    video_path = animate_image(
                        img, config=CFG,
                        profile=ss.profile,
                        clothing=clothing,
                        motion_style=motion_style
                    )
                    vs.update(label="Animation ready ✓", state="complete")

                with out.status("Building ad overlay...", expanded=True) as cs:
                    brand_name = ss.get("brand_name","Brand")
                    if custom_tagline and custom_tagline.strip():
                        tagline = custom_tagline.strip()
                        cs.write(f"💬 Tagline: *{tagline}*")
                    else:
                        cs.write("✍️ Generating tagline...")
                        tagline = generate_tagline(ss.get("original_brief",""), ss.profile, brand_name)
                        cs.write(f"💬 *{tagline}*")

                    cs.write("🎨 Compositing brand card...")
                    prod_img_pil = None
                    if product_file is not None:
                        from PIL import Image as PILImage
                        prod_img_pil = PILImage.open(product_file)

                    ad_path = video_path.replace(".mp4","_ad.mp4")
                    apply_ad_overlay(
                        video_path=video_path,
                        brand_name=brand_name,
                        tagline=tagline,
                        out_path=ad_path,
                        product_img=prod_img_pil,
                        target_duration_sec=ad_duration,
                    )
                    cs.update(label="Ad ready ✓", state="complete")

                out.video(ad_path)
                out.caption(f"📽️ {ad_path}")

            except Exception as e:
                out.error(f"Animation failed: {type(e).__name__}: {e}")

    except Exception as e:
        import traceback
        out.error(f"Generation failed: {type(e).__name__}: {e}")
        out.code(traceback.format_exc())