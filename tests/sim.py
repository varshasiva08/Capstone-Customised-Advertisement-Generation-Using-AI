"""
Simulated generator + verifier for TESTING THE CONTROL LOGIC ONLY.

This is not evidence about FLUX or any real model. It exists so the loop /
controller can be unit-tested without GPUs, API keys or model downloads. The
latent model: each (profile, seed) draws base attribute strengths; correction
tokens found in the prompt shift them in the intended direction.
"""
import hashlib
import random

from PIL import Image

from scoring.types import FaceReading


def _rng(*parts):
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


class SimWorld:
    def __init__(self, cfg, profile_hint=None, age_bias=-8.0, eth_base=(0.2, 0.8),
                 correction_works=True, noise=6.0, face_rate=1.0):
        self.cfg, self.age_bias, self.eth_base = cfg, age_bias, eth_base
        self.correction_works, self.noise, self.face_rate = correction_works, noise, face_rate
        self.calls = 0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _levels_in_prompt(prompt, ladder, fmt):
        best = 0
        for k in range(1, 5):
            tok = ladder.get(f"level_{k}")
            if tok and tok.format(**fmt) in prompt:
                best = max(best, k)
        return best

    def generate(self, prompt, seed):
        self.calls += 1
        img = Image.new("RGB", (8, 8))
        img.latent = {"prompt": prompt, "seed": seed}
        return img

    def make_verifier(self, profile):
        world = self

        class V:
            name = "sim"

            def read(self, image):
                lat = image.latent
                r = _rng(profile["ethnicity"], profile["age"], lat["seed"])
                if r.random() > world.face_rate:
                    return FaceReading(False, scorer="sim")
                cc = world.cfg["corrections"]
                fmt = {"ethnicity": profile["ethnicity"],
                       "gender_noun": "man" if profile["gender"] == "male" else "woman",
                       "gender_adj": "x"}
                p = lat["prompt"]
                eth_lvl = world._levels_in_prompt(p, cc["ethnicity"], fmt)
                yng = world._levels_in_prompt(p, cc["age"]["younger"], fmt)
                old = world._levels_in_prompt(p, cc["age"]["older"], fmt)
                lo = int(profile["age"][:2])
                age = lo + 5 + world.age_bias + r.gauss(0, world.noise)
                if world.correction_works:
                    age += 4.0 * old - 4.0 * yng
                p_eth = r.uniform(*world.eth_base)
                if world.correction_works:
                    p_eth = min(0.99, p_eth + 0.18 * eth_lvl)
                classes = world.cfg["scoring"]["race_map"][profile["ethnicity"]]
                race = {c: 0.0 for c in ["indian", "asian", "black", "white", "middle eastern", "latino hispanic"]}
                race[classes[0]] = 100 * p_eth
                other = [c for c in race if c != classes[0]][0]
                race[other] = 100 * (1 - p_eth)
                g = 99.0
                gender = {"Woman": g, "Man": 100 - g} if profile["gender"] == "female" else {"Man": g, "Woman": 100 - g}
                return FaceReading(True, 0.99, (0, 0, 10, 10), age, gender, race, "sim", 1)
        return V()
