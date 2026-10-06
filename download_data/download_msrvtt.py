import os
import json
import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "VLM2Vec/MSR-VTT"
OUTPUT_DIR = r"S:\video_retrieval\msr_vtt"

N_TRAIN = 1_000
N_TEST = 1_000
SEED = 67


# --------------------------------------------------
# 1. Load split information
# --------------------------------------------------

print("Loading MSR-VTT split information...")

train = load_dataset(
    REPO,
    "train_7k",
    split="train",
)

test = load_dataset(
    REPO,
    "test_1k",
    split="test",
)


train_ids = sorted(set(train["video_id"]))
test_ids = sorted(set(test["video_id"]))

print("Full train:", len(train_ids))
print("Full test :", len(test_ids))


# --------------------------------------------------
# 2. Select reproducible training subset
# --------------------------------------------------

random.seed(SEED)

train_subset = random.sample(train_ids, N_TRAIN)
test_subset = random.sample(test_ids, N_TEST)

print("Selected train:", len(train_subset))
print("Selected test :", len(test_subset))

# --------------------------------------------------
# 3. Save selected IDs
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(
    os.path.join(OUTPUT_DIR, "subset.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "seed": SEED,
            "train": train_subset,
            "test": test_subset,
        },
        f,
        indent=2,
    )


# --------------------------------------------------
# 4. Download videos
# --------------------------------------------------

all_ids = train_subset + test_subset

failed = []

for video_id in tqdm(all_ids, desc="Downloading videos"):

    filename = f"raw_videos/{video_id}.mp4"

    try:
        hf_hub_download(
            repo_id=REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
        )

    except Exception as e:
        print(f"\nFailed: {video_id}: {e}")
        failed.append(video_id)


# --------------------------------------------------
# 5. Download full caption annotations
# --------------------------------------------------

hf_hub_download(
    repo_id=REPO,
    filename="raw_data/MSRVTT_data.json",
    repo_type="dataset",
    local_dir=OUTPUT_DIR,
)


# --------------------------------------------------
# 6. Save failures
# --------------------------------------------------

with open(
    os.path.join(OUTPUT_DIR, "failed.json"),
    "w",
) as f:
    json.dump(failed, f, indent=2)


print("\nFinished!")
print("Successful:", len(all_ids) - len(failed))
print("Failed:", len(failed))
