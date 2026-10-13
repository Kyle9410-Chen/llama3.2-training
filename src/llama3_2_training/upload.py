import argparse
import os

from huggingface_hub import HfApi

REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def get_args():
    p = argparse.ArgumentParser(description="Upload a trained model folder to the Hugging Face Hub")
    p.add_argument("model_dir", nargs="?", default=os.environ.get("UPLOAD_MODEL_PATH"),
                   help="e.g. out/muon-1h-535585/final (default: $UPLOAD_MODEL_PATH)")
    p.add_argument("repo_id", nargs="?", default=os.environ.get("HF_REPO_ID"),
                   help="e.g. your-name/llama1b-muon (default: $HF_REPO_ID)")
    p.add_argument("--message", default=None, help="commit message (default: based on model_dir)")
    args = p.parse_args()
    if not args.model_dir:
        p.error("model_dir is required (pass it as an argument or set UPLOAD_MODEL_PATH)")
    if not args.repo_id:
        p.error("repo_id is required (pass it as an argument or set HF_REPO_ID)")
    return args


def main():
    args = get_args()

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(args.model_dir, f))]
    if missing:
        raise SystemExit(f"{args.model_dir} is missing: {', '.join(missing)}")

    api = HfApi()
    url = api.create_repo(args.repo_id, exist_ok=True)
    print(f"repo: {url}", flush=True)

    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        commit_message=args.message or f"Upload {args.model_dir}",
    )
    print(f"uploaded {args.model_dir} -> {commit.commit_url}", flush=True)


if __name__ == "__main__":
    main()
