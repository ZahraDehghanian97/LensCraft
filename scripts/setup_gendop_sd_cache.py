import os
import shutil
import sys

from dotenv import load_dotenv

load_dotenv()

MIRROR_REPO = "Manojb/stable-diffusion-2-1-base"
TARGET_REPO_DIRNAME = "models--stabilityai--stable-diffusion-2-1-base"


def main() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    target_dir = os.path.join(HF_HUB_CACHE, TARGET_REPO_DIRNAME)
    if os.path.isdir(target_dir):
        print(f"Cache entry already exists at {target_dir}; nothing to do.")
        return

    snap = snapshot_download(
        MIRROR_REPO,
        allow_patterns=[
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
        ],
    )

    te_dir = os.path.join(snap, "text_encoder")
    src = os.path.join(te_dir, "model.fp16.safetensors")
    dst = os.path.join(te_dir, "model.safetensors")
    if not os.path.exists(src):
        sys.exit(
            f"{MIRROR_REPO} is missing text_encoder/model.fp16.safetensors; "
            "check the repo layout."
        )
    if not (os.path.exists(dst) or os.path.islink(dst)):
        os.symlink(os.path.basename(src), dst)

    if not os.path.isdir(os.path.join(snap, "tokenizer")):
        sys.exit(f"{MIRROR_REPO} has no tokenizer/ subfolder; check the repo id.")

    mirror_dir = os.path.join(HF_HUB_CACHE, f"models--{MIRROR_REPO.replace('/', '--')}")
    os.rename(mirror_dir, target_dir)
    shutil.rmtree(
        os.path.join(HF_HUB_CACHE, ".locks", os.path.basename(mirror_dir)),
        ignore_errors=True,
    )
    print(f"Cache entry ready: {target_dir}")


if __name__ == "__main__":
    main()
