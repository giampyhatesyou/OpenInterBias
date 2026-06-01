import os
from utils.config import VQA_SETTING, GEN_SETTING
from utils.datasets import VQA_dataset
ds = VQA_SETTING["coco"]["generated"]
ds["images_path"] = os.path.join(GEN_SETTING["save_path"], ds["subfolder"], "sd-xl", ds["inner_folder"])
print("images_path =", ds["images_path"])
try:
    d = VQA_dataset(ds, "generated", ds["max_prompts_per_bias"], GEN_SETTING["filter_threshold"], GEN_SETTING["hard_threshold"], GEN_SETTING["merge_threshold"], ds["valid_bias_fn"], ds["filter_caption_fn"])
    print("VQA_DATASET_OK items=", len(d))
except Exception as e:
    import traceback; print("VQA_DATASET_CRASH", type(e).__name__, str(e)[:120])
    traceback.print_exc()
