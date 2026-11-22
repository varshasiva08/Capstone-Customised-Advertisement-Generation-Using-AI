"""
Blind rating tool for the human ground truth (run by each rater separately).

    streamlit run eval/rating_app.py -- results/pilot2/rating

The rater types a name, then labels each image. Progress is saved after every
image to  <folder>/ratings_<name>.csv  and resumes automatically.
Raters never see the target profile or any automatic score.
"""
import csv
import os
import sys

import streamlit as st

FOLDER = next((a for a in sys.argv[1:] if not a.startswith("-")), "rating")
COLS = ["image_id", "perceived_gender", "perceived_ethnicity", "perceived_age_decade",
        "perceived_body_type", "face_visible", "notes"]
GENDER = ["female", "male", "unclear"]
ETH = ["South Asian", "East Asian", "Southeast Asian", "African American", "Latin American",
       "Middle Eastern", "White", "unclear / mixed"]
AGE = ["20s", "30s", "40s", "50s", "60s", "under 20", "70+"]
BODY = ["slim", "medium", "plus-size", "cannot see full body"]

st.set_page_config(page_title="Rate images", layout="wide")
st.title("Image rating")
st.caption("Look at the person in each image and answer from what you see. "
           "Do not guess what was requested. If you can't tell, choose 'unclear'.")

name = st.sidebar.text_input("Your name (no spaces)").strip().lower()
if not name:
    st.info("Enter your name in the sidebar to begin.")
    st.stop()

sheet = os.path.join(FOLDER, "rating_sheet_blind.csv")
ids = [r["image_id"] for r in csv.DictReader(open(sheet, newline=""))]
out = os.path.join(FOLDER, f"ratings_{name}.csv")
done = {}
if os.path.exists(out):
    for r in csv.DictReader(open(out, newline="")):
        done[r["image_id"]] = r

todo = [i for i in ids if i not in done]
st.sidebar.progress(len(done) / len(ids), text=f"{len(done)} / {len(ids)} rated")
if not todo:
    st.success("All images rated. Thank you! Send the ratings file to the project owner:")
    st.code(out)
    st.stop()

cur = todo[0]
left, right = st.columns([1.3, 1])
left.image(os.path.join(FOLDER, "images", cur + ".png"), width="stretch")
with right:
    with st.form(f"f_{cur}", clear_on_submit=True):
        visible = st.checkbox("A face is clearly visible", value=True)
        g = st.radio("Gender (as it appears)", GENDER, horizontal=True)
        e = st.selectbox("Ethnicity (as it appears)", ETH)
        a = st.radio("Apparent age decade", AGE, horizontal=True)
        b = st.radio("Body type", BODY, horizontal=True)
        note = st.text_input("Notes (optional)")
        if st.form_submit_button("Save and next", type="primary"):
            new = not os.path.exists(out)
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, COLS)
                if new:
                    w.writeheader()
                w.writerow({"image_id": cur, "perceived_gender": g, "perceived_ethnicity": e,
                            "perceived_age_decade": a, "perceived_body_type": b,
                            "face_visible": int(visible), "notes": note})
            st.rerun()
