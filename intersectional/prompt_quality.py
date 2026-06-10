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


def build_leak_index(proposed_biases_path, attr_map=None):
    """Open-set leakage: ``{caption_id: {attr: leaked}}`` for EVERY proposed attribute.

    ATTR_WORDS only covers the three demographic attributes; for the all-pairs scan we need a
    check that scales to arbitrary free-text attributes. The proposal entry already carries the
    candidate ``classes`` for each bias, so the generic test is: does the caption lexically
    mention one of the proposed class labels for that attribute ("a man WALKING down the
    street" leaks activity when its classes include "walking")? For the demographic attributes
    the ATTR_WORDS synonym lists are OR-ed in (class labels alone would miss "lady", "grandpa").
    Same word-boundary heuristic as ``mentions``; still lexical, still to be eyeballed.
    """
    with open(proposed_biases_path) as f:
        data = json.load(f)
    index = {}
    for e in data["bias_proposal"]:
        caption = (e.get("caption") or "").lower()
        attrs = {}
        for bias in e.get("proposed_biases", {}).get("bias", []):
            if not isinstance(bias.get("name"), str) or not isinstance(bias.get("classes"), list):
                continue  # a handful of malformed LLM proposals in the full COCO file
            attr = bias["name"].strip().split()[-1].lower()
            attr = (attr_map or {}).get(attr, attr)
            leaked = attrs.get(attr, False) or mentions(attr, caption)
            if not leaked:
                for cls in bias.get("classes", []):
                    if re.search(r"\b" + re.escape(str(cls).lower()) + r"\b", caption):
                        leaked = True
                        break
            attrs[attr] = leaked
        index[str(e["caption_id"])] = attrs
    return index


def leaks(attr, caption_id, leak_index, attr_map=None):
    """Leak lookup for a (possibly raw) attribute name; keys are last-token canonical
    (with the same synonym merging the index was built with)."""
    key = attr.strip().split()[-1].lower()
    key = (attr_map or {}).get(key, key)
    return bool(leak_index.get(str(caption_id), {}).get(key, False))


def load_caption_map(proposed_biases_path):
    """caption_id (str) -> caption text, from a proposed_biases JSON."""
    with open(proposed_biases_path) as f:
        data = json.load(f)
    return {str(e["caption_id"]): e.get("caption", "") for e in data["bias_proposal"]}


def pair_leakage(caption_ids, attr_a, attr_b, caption_map, leak_index=None):
    """Per-pair prompt-quality: fraction of captions that state attr_a and/or attr_b.

    A high value means the joint result for this pair is partly dictated by the prompts, not the
    model -> treat the pair's NMI with caution / exclude leaky captions. With ``leak_index``
    the check covers all open-set attributes (class-label based); otherwise it falls back to
    the demographic ATTR_WORDS lists on ``caption_map``."""
    n = len(caption_ids)
    if n == 0 or (not caption_map and leak_index is None):
        return None
    leak_a = leak_b = leak_any = 0
    for cid in caption_ids:
        if leak_index is not None:
            a = leaks(attr_a, cid, leak_index)
            b = leaks(attr_b, cid, leak_index)
        else:
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
