"""Pre-populate ``utils/synonyms.json`` for every class of a proposals file. CPU, offline.

``utils.filter_caption_generated`` queries ConceptNet (one HTTP call per class missing from
synonyms.json) — api.conceptnet.io answers 502 for most requests, which once burned a whole
GPU job's wall-clock before a single image was generated. The committed synonyms.json only
covered the classes that survived the baseline's ``filter_threshold=0.5``; the open-set run
sets the threshold to 0, exposing ~6k uncovered tail classes (~12k with case variants: the
post-processing lowercases classes before the lookup, so both spellings must be present).

This writes the same minimal entry the pipeline would produce when ConceptNet errors out:
``[class, plural, singular]`` (inflect, guarded — it crashes on odd LLM class strings, and a
cached entry also prevents that crash inside the pipeline). Weaker than ConceptNet's synonym
expansion for the upstream caption filter, but deterministic and offline; the downstream
open-set leak index (prompt_quality.build_leak_index) measures leakage independently anyway.

    python intersectional/complete_synonyms.py proposed_biases/coco/3/coco_train_openset.json
"""
import sys
import json

import inflect


def complete(proposals_path, synonyms_path="utils/synonyms.json"):
    eng = inflect.engine()
    with open(synonyms_path) as f:
        syn = json.load(f)
    with open(proposals_path) as f:
        data = json.load(f)["bias_proposal"]
    variants = set()
    for c in data:
        for b in c["proposed_biases"].get("bias", []):
            if isinstance(b.get("classes"), list):
                for x in b["classes"]:
                    s = str(x)
                    variants.update({s, s.strip(), s.lower(), s.strip().lower()})
    added = bad = 0
    for cls in sorted(variants):
        if cls in syn:
            continue
        words = [cls]
        try:
            words.append(eng.plural(cls))
            singular = eng.singular_noun(cls)
            if singular is not False:
                words.append(singular)
        except Exception:
            bad += 1
        syn[cls] = sorted(set(words))
        added += 1
    with open(synonyms_path, "w") as f:
        json.dump(syn, f, indent=4)
    print(f"added {added} offline entries ({bad} inflect-failures); total {len(syn)}")
    return added


if __name__ == "__main__":
    complete(sys.argv[1] if len(sys.argv) > 1 else
             "proposed_biases/coco/3/coco_train_openset.json")
