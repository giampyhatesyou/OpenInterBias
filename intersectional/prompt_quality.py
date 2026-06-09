"""Prompt quality: detect when the generation prompt already states an attribute.

OpenBias is supposed to only measure attributes the prompt does NOT specify (the `present_in_prompt`
flag from the bias-proposal LLM). On real data that flag is unreliable: many captions explicitly
name the gender ("a man standing in a kitchen") while gender is still flagged
`present_in_prompt=False`. When the prompt fixes one attribute, the measured value on it is not a
free choice of the model, and pairing it with another attribute can create a spurious joint
correlation that comes from the prompt, not the model.

This module detects, for a given caption (= the generation prompt; SDXL uses the raw COCO caption,
`pos_prompt=''`), whether it lexically states a demographic attribute. Used to (a) REPORT per-pair
leakage rates and (b) optionally EXCLUDE leaky observations from the joint analysis.
"""
import json
import re

# word lists per canonical attribute (heuristic; lower-cased, word-boundary matched)
ATTR_WORDS = {
    "gender": r"\b(man|men|woman|women|boy|girl|male|female|lady|ladies|guy|guys|gentleman|"
              r"gentlemen|he|she|his|her|mother|father|mom|dad|son|daughter|husband|wife|king|"
              r"queen|waiter|waitress|businessman|policeman|fireman|actress|actor)\b",
    "race": r"\b(white|black|asian|caucasian|african|african-american|hispanic|latino|latina|"
            r"indian|arab|european|chinese|japanese)\b",
    "age": r"\b(baby|babies|toddler|child|children|kid|kids|teen|teenager|young|old|older|elderly|"
           r"senior|adult|grandfather|grandmother|grandma|grandpa|boy|girl|infant|elder)\b",
}
_RE = {a: re.compile(p, re.I) for a, p in ATTR_WORDS.items()}


def mentions(attr, caption):
    """True if the caption lexically states the attribute (so the prompt is not neutral for it)."""
    r = _RE.get(attr)
    return bool(r and caption and r.search(caption))


def load_caption_map(proposed_biases_path):
    """caption_id (str) -> caption text, from a proposed_biases JSON."""
    with open(proposed_biases_path) as f:
        data = json.load(f)
    return {str(e["caption_id"]): e.get("caption", "") for e in data["bias_proposal"]}


def pair_leakage(caption_ids, attr_a, attr_b, caption_map):
    """Per-pair prompt-quality: fraction of captions that state attr_a and/or attr_b.

    A high value means the joint result for this pair is partly dictated by the prompts, not the
    model -> treat the pair's NMI with caution / exclude leaky captions."""
    n = len(caption_ids)
    if n == 0 or not caption_map:
        return None
    leak_a = leak_b = leak_any = 0
    for cid in caption_ids:
        cap = caption_map.get(str(cid), "")
        a = mentions(attr_a, cap)
        b = mentions(attr_b, cap)
        leak_a += a
        leak_b += b
        leak_any += (a or b)
    return {
        "captions": n,
        "leak_a_rate": round(leak_a / n, 3),
        "leak_b_rate": round(leak_b / n, 3),
        "leak_any_rate": round(leak_any / n, 3),
        "clean_captions": n - leak_any,
    }
