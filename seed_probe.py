from utils.config import GEN_SETTING
from utils.datasets import Proposed_biases
gs = GEN_SETTING["coco"]
try:
    ds = Proposed_biases(gs["proposed_biases_path"], GEN_SETTING["max_prompts_per_bias"], GEN_SETTING["filter_threshold"], GEN_SETTING["hard_threshold"], GEN_SETTING["merge_threshold"], gs["valid_bias_fn"], gs["filter_caption_fn"], gs["all_images"])
    print("OK len=", len(ds))
except Exception as e:
    print("CRASH", type(e).__name__, str(e)[:70])
