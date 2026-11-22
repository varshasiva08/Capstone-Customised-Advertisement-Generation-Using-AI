"""Build a small blind set (21 images) for a THIRD rater to settle the gender disagreements.
Run from the project folder (adfidelity):   python make_gender_check_set.py
Creates  rater3\\images\\g001.png ... and  rater3\\rating_sheet_blind.csv  (send this folder, plus rating_app.py).
Creates  rater3_key_PRIVATE.csv  (maps g001.. back to batch + image id). Do NOT send that file.
"""
import csv, os, random, shutil

PAIRS = (
    [("main", i) for i in ["r0000","r0010","r0016","r0020","r0035","r0051","r0052","r0056","r0076",
                           "r0087","r0091","r0093","r0118","r0142","r0144"]]
    + [("extra1", i) for i in ["r0000","r0030","r0054","r0060","r0071"]]
    + [("extra2", "r0009")]
)
random.Random(11).shuffle(PAIRS)
os.makedirs(os.path.join("rater3", "images"), exist_ok=True)
with open("rater3/rating_sheet_blind.csv", "w", newline="") as fs, open("rater3_key_PRIVATE.csv", "w", newline="") as fk:
    ws, wk = csv.writer(fs), csv.writer(fk)
    ws.writerow(["image_id", "image_path"]); wk.writerow(["new_id", "batch", "orig_id"])
    for n, (b, i) in enumerate(PAIRS, 1):
        src = os.path.join("results", b, "rating", "images", i + ".png")
        if not os.path.exists(src):
            print("MISSING", src); continue
        nid = f"g{n:03d}"
        shutil.copy(src, os.path.join("rater3", "images", nid + ".png"))
        ws.writerow([nid, f"images/{nid}.png"]); wk.writerow([nid, b, i])
print(len(PAIRS), "images prepared in rater3\\ ; key in rater3_key_PRIVATE.csv")
