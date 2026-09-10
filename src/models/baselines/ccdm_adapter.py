import os
import importlib.util
import numpy as np
import torch
import logging
import gdown

from utils.paths import third_party

logger = logging.getLogger(__name__)


class _CCDMTextEmbedder:
    """Use the projected CLIP features used to train the released checkpoint."""

    def __init__(self, model_name, device, chunk_size=100):
        import clip

        if model_name not in ("openai/clip-vit-base-patch32", "clip-vit-base-patch32", "ViT-B/32"):
            raise ValueError("The CCDM checkpoint requires CLIP ViT-B/32 text embeddings")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.device = device
        self.chunk_size = chunk_size
        self.tokenize = clip.tokenize
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.model.eval()

    def extract_clip_embeddings(self, texts, return_seq=False):
        if return_seq:
            raise ValueError("CCDM requires pooled, projected text embeddings")
        with torch.no_grad():
            return torch.cat([
                self.model.encode_text(
                    self.tokenize(texts[start:start + self.chunk_size], truncate=True).to(self.device)
                ).float()
                for start in range(0, len(texts), self.chunk_size)
            ])


class CCDMAdapter:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.n_T = 1000
        self.n_feature = 5
        self.n_textemb = 512

        self.seq_len = getattr(config, "seq_len", 300)
        self.tan_half_fov_x = getattr(config, "tan_half_fov_x", 0.3639)
        self.tan_half_fov_y = getattr(config, "tan_half_fov_y", 0.2055)

        self.ddpm, self.clip_embedder, self.mean, self.std = self._load_models()

    def _load_models(self):
        # Load by file: other baselines can also have a module called `main`.
        spec = importlib.util.spec_from_file_location(
            "_lenscraft_ccdm_upstream",
            third_party("Camera-control", "[2024][EG]Text+keyframe", "main.py"),
        )
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)

        transformer = upstream.Transformer(n_feature=self.n_feature, n_textemb=self.n_textemb)
        ddpm = upstream.DDPM(nn_model=transformer, betas=(1e-4, 0.02), n_T=self.n_T, device=self.device)
        ddpm.to(self.device)

        if not os.path.exists(self.config.checkpoint_path):
            logger.info(f"Checkpoint file not found at {self.config.checkpoint_path}, downloading...")
            os.makedirs(os.path.dirname(self.config.checkpoint_path) or ".", exist_ok=True)
            gdown.download(id="136IZeL4PSf9L6FJ4n_jFM6QFLTDjbvr1", output=self.config.checkpoint_path, quiet=False)
            logger.info(f"Downloaded checkpoint to {self.config.checkpoint_path}")

        state_dict = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        ddpm.load_state_dict(state_dict)
        ddpm.eval()

        clip_embedder = _CCDMTextEmbedder(
            model_name=getattr(self.config, "clip_model_name", "openai/clip-vit-base-patch32"),
            device=self.device,
            chunk_size=getattr(self.config, "chunk_size", 100)
        )

        mean_std_path = os.path.join(self.config.data_dir, "Mean_Std.npy")
        data_path = os.path.join(self.config.data_dir, "data.npy")

        if not os.path.exists(mean_std_path) and not os.path.exists(data_path):
            logger.info(f"Data file not found at {data_path}, downloading...")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            gdown.download(id="1VxmGy9szWShOKzWvIxrmgaNEkeqGPLJU", output=data_path, quiet=False)
            logger.info(f"Downloaded data file to {data_path}")

        if not os.path.exists(mean_std_path):
            data = np.load(data_path, allow_pickle=True)[()]
            d = np.concatenate(data["cam"], 0)
            mean, std = np.mean(d, 0), np.std(d, 0)
            np.save(mean_std_path, {"Mean": mean, "Std": std})

        mean_std_data = np.load(mean_std_path, allow_pickle=True)[()]

        mean = torch.tensor(mean_std_data["Mean"], dtype=torch.float32, device=self.device)
        std = torch.tensor(mean_std_data["Std"], dtype=torch.float32, device=self.device)

        return ddpm, clip_embedder, mean, std

    def _smooth_trajectory_batch(self, batch_trajectories, window_size=10, iterations=4):
        B, T, _ = batch_trajectories.shape
        device = batch_trajectories.device

        idx = torch.arange(T, device=device)
        offsets = idx[None, :] - idx[:, None]
        # Match upstream's x[max(0, i-w):min(T, i+w)] (exclusive upper end).
        kernel = (offsets >= -window_size) & (offsets < window_size)
        kernel = kernel.to(dtype=batch_trajectories.dtype)
        kernel /= kernel.sum(dim=1, keepdim=True)
        kernel = kernel.unsqueeze(0).expand(B, -1, -1)

        traj = batch_trajectories

        for _ in range(iterations):
            traj = torch.bmm(kernel, traj)

        return traj

    _PROMPT_REWRITES = (("pushes in", "zooms in"), ("pulls out", "zooms out"))

    def generate_using_text(self, text_prompts, subject_trajectory=None, trajectory=None, padding_mask=None):
        for old, new in self._PROMPT_REWRITES:
            text_prompts = [prompt.replace(old, new) for prompt in text_prompts]

        with torch.no_grad():
            text_embeddings = self.clip_embedder.extract_clip_embeddings(text_prompts, return_seq=False).to(self.device)

            guide_w = float(self.config.get("guidance_scale", 2.0))
            generated = self.ddpm.sample(
                n_sample=len(text_prompts),
                c=text_embeddings,
                size=(self.seq_len, self.n_feature),
                device=self.device,
                guide_w=guide_w
            )

            denormalized = generated * self.std[None, None, :] + self.mean[None, None, :]
            smoothed_batch = self._smooth_trajectory_batch(denormalized)

            return smoothed_batch
