import os
import shutil
import sys

from dotenv import load_dotenv

load_dotenv()

MIRROR_REPO = "sd2-community/stable-diffusion-2-1-base"
TARGET_REPO_DIRNAME = "models--stabilityai--stable-diffusion-2-1-base"


def main() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from safetensors.torch import safe_open, save_file

    ckpt_path = os.environ.get("GENDOP_CHECKPOINT_PATH")
    if not ckpt_path or not os.path.exists(ckpt_path):
        sys.exit(
            "GENDOP_CHECKPOINT_PATH must point to the downloaded GenDoP "
            f"checkpoint (got: {ckpt_path!r})."
        )

    target_dir = os.path.join(HF_HUB_CACHE, TARGET_REPO_DIRNAME)
    if os.path.isdir(target_dir):
        print(f"Cache entry already exists at {target_dir}; nothing to do.")
        return

    snap = snapshot_download(
        MIRROR_REPO,
        allow_patterns=[
            "model_index.json",
            "feature_extractor/*",
            "scheduler/*",
            "text_encoder/config.json",
            "tokenizer/*",
            "unet/config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    )

    text_encoder = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith("text_encoder."):
                text_encoder[key[len("text_encoder."):]] = f.get_tensor(key)
    if not text_encoder:
        sys.exit(f"No text_encoder.* tensors found in {ckpt_path}.")
    te_file = os.path.join(snap, "text_encoder", "model.safetensors")
    if os.path.islink(te_file):
        os.remove(te_file)
    save_file(text_encoder, te_file, metadata={"format": "pt"})
    print(f"Wrote text encoder ({len(text_encoder)} tensors) from {ckpt_path}")

    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    unet_dir = os.path.join(snap, "unet")
    for name in os.listdir(unet_dir):
        path = os.path.join(unet_dir, name)
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
    unet.save_pretrained(unet_dir)
    print("Wrote tiny stand-in UNet (GenDoP discards it)")

    mirror_dir = os.path.join(HF_HUB_CACHE, f"models--{MIRROR_REPO.replace('/', '--')}")
    os.rename(mirror_dir, target_dir)
    shutil.rmtree(os.path.join(HF_HUB_CACHE, ".locks", os.path.basename(mirror_dir)), ignore_errors=True)
    print(f"Cache entry ready: {target_dir}")


if __name__ == "__main__":
    main()
