import os
import re
import time
import streamlit as st
import yaml
from dotenv import load_dotenv

from profiler.profile_gen import generate_profiles
from profiler.prompt_builder import build_prompt
from profiler.text_parser import check_ollama, parse_brief

load_dotenv()

# ---------- Config ----------
@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

CFG      = load_config()
REQUIRED = CFG["required_fields"]

OLLAMA_CFG   = CFG.get("ollama", {})
OLLAMA_HOST  = OLLAMA_CFG.get("host", "http://localhost:11434")
OLLAMA_MODEL = OLLAMA_CFG.get("model", "phi3:mini")

os.makedirs("outputs", exist_ok=True)

# ---------- Page config (must be first Streamlit call) ----------
st.set_page_config(
    page_title="AdFidelity",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Ollama check ----------
try:
    check_ollama(OLLAMA_HOST)
except RuntimeError as e:
    st.error(
        f"**Ollama is not running.**\n\n{e}\n\n"
        "AdFidelity requires Ollama for brief parsing (phi3:mini). "
        "Please start it before using the app."
    )
    st.stop()

# ---------- Global styles ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem;
    max-width: 780px;
}

/* ── Background ── */
.stApp {
    background: #0D0F1A;
}

/* ── Animations ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(224, 64, 251, 0); }
    50%       { box-shadow: 0 0 0 6px rgba(224, 64, 251, 0.12); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes dot-bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40%            { transform: translateY(-6px); }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* ── Header ── */
.af-header {
    padding: 1.6rem 0 1.2rem 0;
    margin-bottom: 0.4rem;
    animation: fadeSlideUp 0.5s ease both;
}
.af-wordmark {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #E040FB 0%, #B388FF 60%, #F5F0E8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    margin: 0;
}
.af-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4A4D6B;
    margin: 2px 0 0 2px;
}

/* ── Step tracker ── */
.af-steps {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 1.6rem;
    animation: fadeIn 0.4s 0.2s ease both;
}
.af-step-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.af-step-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
    position: relative;
    z-index: 1;
}
.af-step-dot.done {
    background: rgba(224, 64, 251, 0.15);
    border: 1.5px solid rgba(224, 64, 251, 0.4);
    color: #E040FB;
}
.af-step-dot.active {
    background: #E040FB;
    border: 1.5px solid #E040FB;
    color: #fff;
    animation: pulse-glow 2s ease infinite;
}
.af-step-dot.pending {
    background: rgba(255,255,255,0.04);
    border: 1.5px solid rgba(255,255,255,0.1);
    color: #3A3D55;
}
.af-step-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.af-step-label.done   { color: rgba(224,64,251,0.6); }
.af-step-label.active { color: #F5F0E8; }
.af-step-label.pending{ color: #2E3050; }
.af-step-line {
    width: 32px;
    height: 1.5px;
    margin: 0 4px;
}
.af-step-line.done   { background: rgba(224,64,251,0.3); }
.af-step-line.pending{ background: rgba(255,255,255,0.06); }

/* ── Profile chip ── */
.af-profile-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(224, 64, 251, 0.07);
    border: 1px solid rgba(224, 64, 251, 0.18);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    margin-bottom: 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: #C9A0FF;
    animation: fadeSlideUp 0.35s ease both;
}
.af-chip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #E040FB;
    flex-shrink: 0;
}

/* ── Chat messages ── */
.stChatMessage {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    animation: fadeSlideUp 0.3s ease both;
}
.stChatMessage[data-testid="chat-message-user"] {
    background: rgba(224,64,251,0.07) !important;
    border-color: rgba(224,64,251,0.15) !important;
}

/* ── Chat input ── */
.stChatInput {
    animation: pulse-glow 3s ease infinite;
}
.stChatInput > div {
    border-radius: 14px !important;
    border: 1px solid rgba(224,64,251,0.25) !important;
    background: rgba(255,255,255,0.04) !important;
    transition: border-color 0.2s ease;
}
.stChatInput > div:focus-within {
    border-color: rgba(224,64,251,0.6) !important;
    box-shadow: 0 0 0 3px rgba(224,64,251,0.08) !important;
}

/* ── Primary button ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E040FB 0%, #B388FF 100%);
    background-size: 200% auto;
    border: none;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.6rem 1.4rem;
    letter-spacing: 0.02em;
    color: #fff;
    transition: all 0.2s ease;
    animation: shimmer 3s linear infinite;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(224,64,251,0.35);
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(0);
}

/* ── Secondary buttons ── */
div.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #6B6F99;
    transition: all 0.15s ease;
}
div.stButton > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.07);
    color: #F5F0E8;
}

/* ── Section label ── */
.af-section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3A3D5C;
    margin-bottom: 0.5rem;
    margin-top: 0.2rem;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #F5F0E8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(224,64,251,0.4) !important;
    box-shadow: 0 0 0 2px rgba(224,64,251,0.08) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #F5F0E8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Toggle ── */
.stToggle > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: #8A8DB5;
}

/* ── Radio ── */
.stRadio > div {
    gap: 0.5rem;
    flex-direction: row !important;
}
.stRadio label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #8A8DB5;
}

/* ── File uploader ── */
.stFileUploader {
    border: 1.5px dashed rgba(224,64,251,0.2) !important;
    border-radius: 12px !important;
    background: rgba(224,64,251,0.03) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1.4rem 0 !important;
}

/* ── Status / spinner ── */
.stStatus {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* ── Info box ── */
.stAlert {
    background: rgba(224,64,251,0.06) !important;
    border: 1px solid rgba(224,64,251,0.15) !important;
    border-radius: 12px !important;
    color: #C9A0FF !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}

/* ── Images & video ── */
img { border-radius: 12px; }
video { border-radius: 12px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(224,64,251,0.2);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
PRODUCT_QUESTION = (
    "Would you like to add a **product** to this ad? "
    "Reply **yes** or **no**."
)

YES_WORDS = {"yes", "yeah", "yep", "yup", "y", "sure", "ok", "okay"}
NO_WORDS  = {"no", "nope", "nah", "n", "skip", "none"}

# Product keywords → category mapping
PRODUCT_KEYWORDS = {
    "handbag":    ["handbag", "bag", "purse", "tote", "clutch", "satchel"],
    "sunglasses": ["sunglasses", "glasses", "eyewear", "shades"],
    "jewelry":    ["jewellery", "jewelry", "necklace", "earrings", "bracelet",
                   "ring", "pendant", "chain", "anklet", "bangles", "bangle"],
    "clothing":   ["saree", "sari", "kurta", "kurti", "lehenga", "salwar",
                   "dress", "outfit", "suit", "gown", "top", "shirt", "blouse"],
    "other":      ["watch", "scarf", "belt", "shoes", "heels", "sandals",
                   "cap", "hat", "accessory", "accessories"],
}


def detect_product_in_brief(text: str):
    """
    Returns (category, keyword_found) if a product is mentioned in the brief,
    or (None, None) if no product is detected.
    """
    lower = text.lower()
    for category, keywords in PRODUCT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return category, kw
    return None, None


def missing_fields(profile):
    return [f for f in REQUIRED if not profile.get(f)]


def parse_yes_no(text):
    words = re.findall(r"[a-z]+", text.lower())
    if any(w in YES_WORDS for w in words):
        return True
    if any(w in NO_WORDS for w in words):
        return False
    return None


def stage_index(stage):
    return {"collecting": 0, "pick_profile": 1, "ask_product": 2, "ready": 3}.get(stage, 0)


# ---------- Session state ----------
ss = st.session_state
ss.setdefault("messages", [{
    "role": "assistant",
    "content": (
        "Describe your brand or advertisement — e.g. "
        "*'sportswear brand targeting diverse women'* or "
        "*'plus size Indian woman in her 40s, navy suit'*."
    )
}])
ss.setdefault("profile",      {})
ss.setdefault("profiles",     [])
ss.setdefault("mode",         None)
ss.setdefault("stage",        "collecting")
ss.setdefault("want_product", None)
ss.setdefault("product_category_detected", None)
ss.setdefault("original_brief", "")
ss.setdefault("seed",         42)

# ---------- Header ----------
st.markdown("""
<div class="af-header">
  <h1 class="af-wordmark">AdFidelity</h1>
  <p class="af-tagline">Demographic-faithful ad generation</p>
</div>
""", unsafe_allow_html=True)

# ---------- Step indicator ----------
steps = [
    ("Brief", "collecting"),
    ("Profile", "pick_profile"),
    ("Product", "ask_product"),
    ("Generate", "ready"),
]
current = stage_index(ss.stage)
pills = '<div class="af-steps">'
for i, (label, _) in enumerate(steps):
    if i < current:
        cls = "done"
        icon = "✓"
    elif i == current:
        cls = "active"
        icon = str(i + 1)
    else:
        cls = "pending"
        icon = str(i + 1)
    pills += f'''
    <div class="af-step-item">
        <div class="af-step-dot {cls}">{icon}</div>
        <span class="af-step-label {cls}">{label}</span>
    </div>'''
    if i < len(steps) - 1:
        line_cls = "done" if i < current else "pending"
        pills += f'<div class="af-step-line {line_cls}"></div>'
pills += '</div>'
st.markdown(pills, unsafe_allow_html=True)

# ---------- Active profile summary ----------
if ss.stage in ("ask_product", "ready", "pick_profile") and ss.mode == "single" and ss.profile and len(ss.messages) > 2:
    p = ss.profile
    st.markdown(
        f'<div class="af-profile-chip">'
        f'<span class="af-chip-dot"></span>'
        f'{p.get("ethnicity", "—")} &nbsp;·&nbsp; '
        f'{p.get("body_type", "—")} &nbsp;·&nbsp; '
        f'{p.get("age", "—")}'
        f'</div>',
        unsafe_allow_html=True
    )
elif ss.stage in ("ask_product", "ready") and ss.mode == "multi" and ss.profiles:
    st.markdown(
        f'<div class="af-profile-chip">'
        f'<span class="af-chip-dot"></span>'
        f'{len(ss.profiles)} profiles generated'
        f'</div>',
        unsafe_allow_html=True
    )

# ---------- Chat ----------
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_input := st.chat_input("Describe your brand or ad..."):
    ss.messages.append({"role": "user", "content": user_input})
    # Keep the cumulative brief text for product detection
    ss.original_brief = ss.get("original_brief", "") + " " + user_input

    if ss.stage == "pick_profile":
        # Try to parse a number 1–6
        nums = re.findall(r"\b([1-6])\b", user_input)
        if nums:
            chosen = int(nums[0]) - 1
            ss.profile = ss.profiles[chosen]
            ss.mode    = "single"
            ss.stage   = "ask_product"
            p = ss.profile
            ss.messages.append({"role": "assistant", "content":
                f"Great — going with **Profile {chosen + 1}**: "
                f"{p['ethnicity']}, {p['body_type']}, {p['age']}.\n\n"
                + PRODUCT_QUESTION})
        else:
            # They described something new — re-parse as a fresh brief
            ss.profile = parse_brief(
                user_input, {}, CFG, REQUIRED,
                model=OLLAMA_MODEL, host=OLLAMA_HOST
            )
            missing = missing_fields(ss.profile)
            if not missing:
                ss.mode  = "single"
                ss.stage = "ask_product"
                p = ss.profile
                ss.messages.append({"role": "assistant", "content":
                    f"Got it — using **{p['ethnicity']}, {p['body_type']}, {p['age']}**.\n\n"
                    + PRODUCT_QUESTION})
            else:
                ss.messages.append({"role": "assistant", "content":
                    f"Reply with a number **1–6** to pick a profile, "
                    f"or describe the demographic you want more specifically."})

    elif ss.stage == "ask_product":
        answer = parse_yes_no(user_input)
        if answer is True:
            ss.want_product = True
            ss.stage = "ready"
            ss.messages.append({"role": "assistant", "content":
                "Got it — upload your product image below, "
                "then choose clothing, background and hit **Generate**."})
        elif answer is False:
            ss.want_product = False
            ss.stage = "ready"
            ss.messages.append({"role": "assistant", "content":
                "No product — choose clothing and background below, then hit **Generate**."})
        else:
            ss.messages.append({"role": "assistant", "content":
                "Just need a **yes** or **no** — " + PRODUCT_QUESTION})

    else:
        # If this is the first message in a fresh collecting session,
        # always start from an empty profile — never inherit stale data
        existing = ss.profile if len(ss.messages) > 2 else {}
        ss.profile = parse_brief(
            user_input, existing, CFG, REQUIRED,
            model=OLLAMA_MODEL, host=OLLAMA_HOST
        )
        missing = missing_fields(ss.profile)

        if not missing:
            ss.mode  = "single"
            # Check if brief already mentions a product
            detected_cat, detected_kw = detect_product_in_brief(
                ss.get("original_brief", "") + " " + user_input
            )
            if detected_cat:
                ss.want_product = True
                ss.product_category_detected = detected_cat
                ss.stage = "ready"
                ss.messages.append({"role": "assistant", "content":
                    f"Got it — generating an ad featuring a **{ss.profile['body_type']} "
                    f"{ss.profile['ethnicity']} woman in her {ss.profile['age']}** "
                    f"with **{detected_kw}**.\n\n"
                    f"Upload your product image below and hit **Generate**."})
            else:
                ss.stage = "ask_product"
                ss.messages.append({"role": "assistant", "content":
                    f"Got it — generating an ad featuring a **{ss.profile['body_type']} "
                    f"{ss.profile['ethnicity']} woman in her {ss.profile['age']}**.\n\n"
                    + PRODUCT_QUESTION})

        elif len(missing) == 3:
            ss.mode = "multi"
            with st.spinner("Generating demographic profiles..."):
                try:
                    ss.profiles = generate_profiles(
                        user_input, model=OLLAMA_MODEL, host=OLLAMA_HOST)
                    reply = "I've built **6 demographic profiles** for your brand:\n\n"
                    for i, p in enumerate(ss.profiles, 1):
                        reply += (f"**{i}.** {p['ethnicity']}, "
                                  f"{p['body_type']}, {p['age']}\n")
                    reply += "\nWhich profile would you like to use? Reply with a number (1–6), or describe a different demographic."
                    ss.stage = "pick_profile"
                    ss.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    ss.messages.append({"role": "assistant",
                        "content": f"Profile generation failed: {e}"})
        else:
            ss.mode  = "single"
            # Ask ALL missing fields in one message
            questions = []
            for field in missing:
                opts = ", ".join(CFG["options"][field])
                label = {"ethnicity": "ethnicity", "body_type": "body type", "age": "age group"}[field]
                questions.append(f"- **{label.capitalize()}**: {opts}")

            got   = {k: v for k, v in ss.profile.items() if k in REQUIRED and v}
            intro = ("Got so far: " + ", ".join(f"**{v}**" for v in got.values()) + ".\n\n" if got else "")
            reply = intro + "Just need a few more details:\n\n" + "\n".join(questions)
            ss.messages.append({"role": "assistant", "content": reply})

    st.rerun()

# ---------- Generation panel ----------
if ss.stage == "ready":
    st.divider()

    # Brand name
    st.markdown('<p class="af-section-label">Brand</p>', unsafe_allow_html=True)
    brand_name = st.text_input(
        "Brand name",
        placeholder="e.g. Tanishq, Fabindia, Zara...",
        label_visibility="collapsed"
    )
    ss.brand_name = brand_name if brand_name else "Brand"

    # Clothing + background
    st.markdown('<p class="af-section-label">Ad Style</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    clothing   = c1.text_input(
        "Clothing",
        placeholder="e.g. white blazer suit, red kurta, navy saree...",
        help="Describe any outfit — it will always be generated modestly."
    )
    background = c2.selectbox("Background", CFG["options"]["background"], label_visibility="visible")

    if not clothing:
        clothing = "professional outfit"

    # Seed (compact, not hidden behind settings)
    seed = st.number_input(
        "Seed", 0, 999999, ss.seed,
        help="Fixed seed for reproducible outputs.",
        label_visibility="visible"
    )
    ss.seed = seed

    # Product upload
    product_file     = None
    product_category = ss.get("product_category_detected") or "handbag"
    if ss.want_product:
        st.markdown('<p class="af-section-label">Product</p>', unsafe_allow_html=True)
        product_file = st.file_uploader(
            "Upload product image",
            type=["png", "jpg"],
            help="AI will describe the product and place it in the ad.",
            label_visibility="collapsed"
        )
        if product_file:
            cat_options = ["handbag", "sunglasses", "jewelry", "clothing", "other"]
            default_idx = cat_options.index(product_category) if product_category in cat_options else 0
            product_category = st.selectbox(
                "How is it worn / held?",
                cat_options,
                index=default_idx,
            )

    # Animation toggle + options
    st.markdown('<p class="af-section-label">Animation</p>', unsafe_allow_html=True)
    animate_enabled = st.toggle(
        "Generate video",
        value=False,
        help="Animates the final image into an ad video via Wan2.1-I2V."
    )
    if animate_enabled:
        c_style, c_dur = st.columns(2)
        motion_style = c_style.radio(
            "Motion style",
            options=["sway", "walk"],
            format_func=lambda x: "↔ Sway & turn" if x == "sway" else "→ Walk & pose",
            help="Sway: gentle turn to show product. Walk: walks forward then poses."
        )
        ad_duration = c_dur.radio(
            "Duration",
            options=[5, 10, 15, 20],
            format_func=lambda x: f"{x}s",
        )
    else:
        motion_style = "sway"
        ad_duration  = 15

    # Optional tagline
    st.markdown('<p class="af-section-label">Tagline</p>', unsafe_allow_html=True)
    custom_tagline = st.text_input(
        "Tagline",
        placeholder="Leave blank to auto-generate from your brief...",
        label_visibility="collapsed"
    )



    # ---------- Output renderer ----------
    def render_output(profile, key_suffix, container):
        from generator.flux_pipeline import generate_image
        from modules.product_describe import describe_product

        try:
            with container.status("Generating ad...", expanded=True) as status:
                product_description = None
                if ss.want_product and product_file is not None:
                    status.write("👁️ Identifying product...")
                    from PIL import Image as PILImage
                    product_img = PILImage.open(product_file)
                    product_description = describe_product(product_img)
                    status.write(f"🛍️ *{product_description}*")

                prompt = build_prompt(
                    profile, clothing, background, CFG,
                    product_description=product_description,
                    product_category=product_category
                )

                status.write("⚙️ Calling FLUX.1-schnell...")
                img, device, res, steps_used = generate_image(prompt, int(seed), CFG)
                status.update(label="Image ready", state="complete")

            out_path = f"outputs/output_{key_suffix}_seed{seed}.png"
            img.save(out_path)
            container.image(img, caption=f"{device.upper()} · {res[0]}×{res[1]} · seed {seed}")

            if animate_enabled:
                from modules.motion import animate_image
                from modules.ad_compositor import apply_ad_overlay, generate_tagline
                try:
                    with container.status("Animating via Wan2.1-I2V...", expanded=True) as vs:
                        vs.write(f"🎬 Style: {'Sway & turn' if motion_style == 'sway' else 'Walk & pose'}...")
                        vs.write("🚀 Calling Wan2.1-I2V — this takes ~60–120 s...")
                        video_path = animate_image(
                            img, config=CFG,
                            profile=profile,
                            clothing=clothing,
                            motion_style=motion_style
                        )
                        vs.update(label="Animation ready", state="complete")

                    with container.status("Building ad overlay...", expanded=True) as cs:
                        brief_text = ss.get("original_brief", "")
                        brand_name = ss.get("brand_name", "Brand")

                        # Use custom tagline if provided, otherwise generate
                        if custom_tagline and custom_tagline.strip():
                            tagline = custom_tagline.strip()
                            cs.write(f"💬 Using your tagline: *{tagline}*")
                        else:
                            cs.write("✍️ Generating tagline...")
                            tagline = generate_tagline(brief_text, profile, brand_name)
                            cs.write(f"💬 *{tagline}*")

                        cs.write("🎨 Compositing brand card...")
                        prod_img_pil = None
                        if product_file is not None:
                            from PIL import Image as PILImage
                            prod_img_pil = PILImage.open(product_file)

                        ad_path = video_path.replace(".mp4", "_ad.mp4")
                        apply_ad_overlay(
                            video_path=video_path,
                            brand_name=brand_name,
                            tagline=tagline,
                            out_path=ad_path,
                            product_img=prod_img_pil,
                            target_duration_sec=ad_duration,
                        )
                        cs.update(label="Ad ready", state="complete")

                    container.video(ad_path)
                    container.caption(f"📽️ `{ad_path}`")
                except Exception as e:
                    container.error(f"Animation failed: {type(e).__name__}: {e}")

        except Exception as e:
            import traceback
            container.error(f"Generation failed: {type(e).__name__}: {e}")
            container.code(traceback.format_exc())

    # ---------- Generate buttons ----------
    st.divider()

    if not missing_fields(ss.profile):
        if st.button("🎨 Generate Ad", type="primary", use_container_width=True):
            render_output(ss.profile, "single", st.container())
    else:
        st.info("Complete the profile details in the chat above to unlock generation.")

# ---------- Footer ----------
st.divider()
col1, col2 = st.columns([4, 1])
col2.button(
    "↺ New brief",
    on_click=lambda: ss.update({
        "profile": {}, "profiles": [], "mode": None,
        "stage": "collecting", "want_product": None,
        "original_brief": "", "product_category_detected": None,
        "messages": ss.messages[:1]
    })
)