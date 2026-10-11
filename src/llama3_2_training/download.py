import json, random, collections
from huggingface_hub import HfApi, snapshot_download

DATASET = "allenai/dolma3_mix-150B-1025"
RATIO = 6 / 157 * 1.2

def shards():
    api = HfApi()
    files = [f for f in api.list_repo_tree(DATASET, repo_type="dataset",
                                           path_in_repo="data", recursive=True)
             if f.path.endswith(".jsonl.zst")]

    by_dir = collections.defaultdict(list)
    for f in files:
        by_dir[f.path.rsplit("/", 1)[0]].append(f)

    random.seed(0)
    selected = []
    for d, fs in sorted(by_dir.items()):
        fs = sorted(fs, key=lambda f: f.path)
        random.shuffle(fs)
        budget = sum(f.size for f in fs) * RATIO
        acc = 0
        for f in fs:
            if acc >= budget:
                break
            selected.append(f.path)
            acc += f.size
    return selected

def main():
    selected = shards()
    with open("selected_shards.json", "w") as fp:
        json.dump(selected, fp, indent=1)
    print(f"{len(selected)} shards")

    snapshot_download(DATASET, repo_type="dataset",
                      allow_patterns=selected,
                      local_dir="dolma3_6b_raw",
                      max_workers=16)

if __name__ == "__main__":
    main()