import os
import re
import streamlit as st
import yaml
from dotenv import load_dotenv
 
from profiler.profile_gen import generate_profiles
from profiler.prompt_builder import build_prompt
from profiler.text_parser import check_ollama, parse_brief
from generator.modules.motion import animate_image
load_dotenv()
 
# ---------- Config ----------
@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)
 
CFG = load_config()
REQUIRED = CFG["required_fields"]
 
OLLAMA_CFG   = CFG.get("ollama", {})
OLLAMA_HOST  = OLLAMA_CFG.get("host", "http://localhost:11434")
OLLAMA_MODEL = OLLAMA_CFG.get("model", "phi3:mini")
 
os.makedirs("outputs", exist_ok=True)
 
# ---------- Ollama required check ----------
try:
    check_ollama(OLLAMA_HOST)
except RuntimeError as e:
    st.error(
        f"**Ollama is not running.**\n\n{e}\n\n"
        "AdFidelity requires Ollama for brief parsing (phi3:mini). "
        "Please start it before using the app."
    )
    st.stop()
 
st.set_page_config(page_title="AdFidelity", page_icon="🎯", layout="centered")
 
# ---------- Styling ----------
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 0; max-width: 720px; }
 
.af-header { display: flex; align-items: center; gap: 0.6rem;
    padding: 0.9rem 0 0.6rem 0;
    border-bottom: 1px solid rgba(120,120,120,0.15); margin-bottom: 0.5rem; }
.af-header h1 { margin: 0; font-size: 1.25rem; font-weight: 700; color: #6C63FF; }
.af-header p  { margin: 0; font-size: 0.8rem; color: #888; }
 
.af-placeholder { border: 2px dashed rgba(120,120,120,0.30); border-radius: 12px;
    padding: 50px 20px; text-align: center; color: #999; font-size: 0.9rem; margin-top: 0.5rem; }
 
div.stButton > button[kind="primary"] {
    background-color: #6C63FF; border: none; border-radius: 10px; font-weight: 600; }
div.stButton > button[kind="primary"]:hover { background-color: #5a52e0; }
 
.coming-soon { display: inline-block; font-size: 0.7rem;
    background: rgba(108,99,255,0.12); color: #6C63FF;
    border-radius: 6px; padding: 1px 7px; margin-left: 4px; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)
 
# ---------- Follow-up questions ----------
FOLLOW_UPS = {
    "ethnicity": "Which **ethnicity** should the model be? Options: {opts}",
    "body_type": "What **body type**? Options: {opts}",
    "age":       "Which **age group**? Options: {opts}",
}
 
PRODUCT_QUESTION = (
    "Would you like to add a **product** to this ad? If yes, I'll also "
    "ask about the **motion/animation** style — both are placeholders "
    "(Modules 4 & 5, coming soon). Reply **yes** or **no**."
)
 
YES_WORDS = {"yes", "yeah", "yep", "yup", "y", "sure", "ok", "okay"}
NO_WORDS  = {"no", "nope", "nah", "n", "skip", "none"}
 
 
def missing_fields(profile):
    return [f for f in REQUIRED if not profile.get(f)]
 
 
def parse_yes_no(text):
    words = re.findall(r"[a-z]+", text.lower())
    if any(w in YES_WORDS for w in words):
        return True
    if any(w in NO_WORDS for w in words):
        return False
    return None
 
 
# ---------- Session state ----------
ss = st.session_state
ss.setdefault("messages", [{"role": "assistant",
    "content": "Describe your brand or advertisement — e.g. "
               "*'sportswear brand targeting diverse women'* or "
               "*'plus size indian woman in her 40s, navy suit'*."}])
ss.setdefault("profile",      {})
ss.setdefault("profiles",     [])
ss.setdefault("mode",         None)
ss.setdefault("stage",        "collecting")
ss.setdefault("want_product", None)
 
# ---------- Header ----------
st.markdown("""
<div class="af-header">
  <div>
    <h1>🎯 AdFidelity</h1>
    <p>Demographic-faithful ad model generator</p>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ---------- Settings ----------
with st.expander("⚙️ Settings"):
    enable_generation = st.toggle(
        "Enable image generation",
        value=False,
        help="Calls the FLUX.1-schnell Hugging Face Inference API. Requires HF_TOKEN in .env.")
 
    if enable_generation:
        seed = st.number_input("Seed", 0, 999999, 42,
            help="Fixed seed for reproducible outputs. Change to get different variations.")
    else:
        seed = 42
 
# ---------- Chat messages ----------
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 
# ---------- Chat input ----------
if user_input := st.chat_input("Describe your brand or ad..."):
    ss.messages.append({"role": "user", "content": user_input})
 
    if ss.stage == "ask_product":
        answer = parse_yes_no(user_input)
        if answer is True:
            ss.want_product = True
            ss.stage = "ready"
            ss.messages.append({"role": "assistant", "content":
                "Got it — set the product & motion placeholders below, "
                "pick clothing and background, then hit **Generate**."})
        elif answer is False:
            ss.want_product = False
            ss.stage = "ready"
            ss.messages.append({"role": "assistant", "content":
                "No problem — pick clothing and background below, then hit **Generate**."})
        else:
            ss.messages.append({"role": "assistant", "content":
                "Just need a **yes** or **no** — " + PRODUCT_QUESTION})
 
    else:
        ss.profile = parse_brief(
            user_input, ss.profile, CFG, REQUIRED,
            model=OLLAMA_MODEL, host=OLLAMA_HOST
        )
        missing = missing_fields(ss.profile)
 
        if not missing:
            ss.mode = "single"
            ss.stage = "ask_product"
            ss.messages.append({"role": "assistant", "content":
                f"Got it — generating an ad featuring a **{ss.profile['body_type']} "
                f"{ss.profile['ethnicity']} woman in her {ss.profile['age']}**.\n\n"
                + PRODUCT_QUESTION})
 
        elif len(missing) == 3:
            ss.mode = "multi"
            with st.spinner("Generating 6 demographic profiles..."):
                try:
                    ss.profiles = generate_profiles(
                        user_input, model=OLLAMA_MODEL, host=OLLAMA_HOST)
                    reply = "Generated **6 demographic profiles** for your brand:\n\n"
                    for i, p in enumerate(ss.profiles, 1):
                        reply += (f"**Profile {i}** — {p['ethnicity']}, "
                                  f"{p['body_type']}, {p['age']}\n")
                    reply += "\n" + PRODUCT_QUESTION
                    ss.stage = "ask_product"
                    ss.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    ss.messages.append({"role": "assistant",
                        "content": f"Profile generation failed: {e}"})
        else:
            ss.mode = "single"
            field = missing[0]
            question = FOLLOW_UPS[field].format(opts=", ".join(CFG["options"][field]))
            got = {k: v for k, v in ss.profile.items() if k in REQUIRED and v}
            reply = ("Got so far: " + ", ".join(f"**{v}**" for v in got.values())
                     + ".\n\n" if got else "") + question
            ss.messages.append({"role": "assistant", "content": reply})
 
    st.rerun()
 
# ---------- Generation panel ----------
if ss.stage == "ready":
    st.divider()
 
    if ss.want_product:
        st.markdown(
            "##### 🛍️ Product & 🎬 Motion <span class='coming-soon'>placeholder</span>",
            unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.file_uploader("Upload product image", type=["png", "jpg"], disabled=True,
                          help="Module 4 — product compositing, coming soon")
        p1.selectbox("Logo position", ["Top Left", "Top Right", "Bottom Right"], disabled=True)
        animation_style = p2.selectbox("Animation style", ["Subtle sway", "Camera pan", "Zoom in"],
              help="Module 5 — Viggle motion template")
        st.divider()
 
    c1, c2 = st.columns(2)
    clothing   = c1.selectbox("Clothing",   CFG["options"]["clothing"])
    background = c2.selectbox("Background", CFG["options"]["background"])
 
    def render_output(profile, key_suffix, container):
        if not enable_generation:
            container.markdown(
                '<div class="af-placeholder">🖼️ Image will appear here<br>'
                '<small>Enable image generation in Settings to generate real images</small></div>',
                unsafe_allow_html=True)
            return
 
        from generator.flux_pipeline import generate_image
        try:
            prompt = build_prompt(profile, clothing, background, CFG)
            with container.status("Generating...", expanded=True) as status:
                status.write("⚙️ Calling FLUX.1-schnell...")
                img, device, res, steps = generate_image(prompt, int(seed), CFG)
                status.update(label="Done!", state="complete")
 
            container.image(img, caption=f"{device.upper()} · {res[0]}×{res[1]} · seed {seed}")
            out_path = f"outputs/output_{key_suffix}_seed{seed}.png"
            img.save(out_path)
            container.caption(f"Saved to `{out_path}`")
            # ---- Module 5: animation ----
            if ss.want_product:
                with container.status("Animating...", expanded=True) as anim_status:
                    anim_status.write("🎬 Applying motion...")
                    try:
                        video_path = animate_image(
                            image=img,
                            style=animation_style,
                            api_key=os.getenv("VIGGLE_API_KEY"),
                        )
                        anim_status.update(label="Done!", state="complete")
                        container.video(video_path)
                        container.caption(f"Saved to `{video_path}`")
                    except Exception as e:
                        container.error(f"Animation failed: {e}")
            # ---- end Module 5 ----
 
        except Exception as e:
            import traceback
            container.error(f"Generation failed: {type(e).__name__}: {e}")
            container.code(traceback.format_exc())
 
    if ss.mode == "single" and not missing_fields(ss.profile):
        if st.button("🎨 Generate Advertisement", type="primary", use_container_width=True):
            render_output(ss.profile, "single", st.container())
 
    elif ss.mode == "multi" and ss.profiles:
        if st.button("🎨 Generate All 6 Profiles", type="primary", use_container_width=True):
            for i, profile in enumerate(ss.profiles, 1):
                with st.expander(
                    f"Profile {i} — {profile['ethnicity']}, "
                    f"{profile['body_type']}, {profile['age']}",
                    expanded=(i == 1)
                ):
                    render_output(profile, f"profile{i}", st.container())
    
 
st.divider()
if st.button("🔄 Start new brief", use_container_width=False):
    ss.profile      = {}
    ss.profiles     = []
    ss.mode         = None
    ss.stage        = "collecting"
    ss.want_product = None
    ss.messages     = ss.messages[:1]
    st.rerun()



video_path = animate_image(
    image=img,                          # the PIL.Image from generate_image()
    driving_video_path=video_path,      # from the new file_uploader
    api_key=os.getenv("VIGGLE_API_KEY"),
)
