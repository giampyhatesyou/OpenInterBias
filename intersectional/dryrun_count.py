"""Dry-run gate: print how many images the CURRENT config would generate, without any GPU.

Used by cluster/run_demo.sh. Also exercises utils.post_processing (the code path that can hang on
ConceptNet), so if this returns quickly the real generation won't hang. CPU-only.
"""
import sys
import os

# this file lives in intersectional/; add the repo root so `import utils...` resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.argv = ["x", "--dataset", "coco", "--generator", "sd-xl"]
import utils.arg_parse as ap  # noqa: E402

opt = ap.argparse_generate_images()
from utils.datasets import Proposed_biases  # noqa: E402

ds = Proposed_biases(
    opt["dataset_setting"]["proposed_biases_path"],
    opt["gen_setting"]["max_prompts_per_bias"],
    opt["gen_setting"]["filter_threshold"],
    opt["gen_setting"]["hard_threshold"],
    opt["gen_setting"]["merge_threshold"],
    opt["dataset_setting"]["valid_bias_fn"],
    opt["dataset_setting"]["filter_caption_fn"],
    opt["dataset_setting"]["all_images"],
)
print("IMAGES=%d" % len(ds.get_data()))
