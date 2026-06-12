import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class GenDoPAdapter:
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device

        self.discrete_bins = int(config.get("discrete_bins", 256))
        self.pose_length = int(config.get("pose_length", 30))
        self.cond_mode = str(config.get("cond_mode", "text"))
        self.num_cond_tokens = int(config.get("num_cond_tokens", 77))
        self.target_height = int(config.get("target_height", 512))
        self.target_width = int(config.get("target_width", 512))
        self.max_seq_length = config.get("max_seq_length", None)
        self.test_max_seq_length = config.get("test_max_seq_length", None)

        self._gendop_root = self._resolve_gendop_root()
        self.model, self.opt = self._load_model()

        # Cache the token decoder after sys.path is configured.
        from core.utils import token_to_camera
        self._token_to_camera = token_to_camera

    @staticmethod
    def _resolve_gendop_root() -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "third_parties",
            "GenDoP",
        )

    def _load_model(self):
        if self._gendop_root not in sys.path:
            sys.path.insert(0, self._gendop_root)

        from core.models import LMM
        from core.options import Options
        from core.utils import monkey_patch_transformers

        monkey_patch_transformers()

        opt = Options()
        opt.discrete_bins = self.discrete_bins
        opt.pose_length = self.pose_length
        opt.cond_mode = self.cond_mode
        opt.num_cond_tokens = self.num_cond_tokens
        opt.target_height = self.target_height
        opt.target_width = self.target_width
        if self.max_seq_length is not None:
            opt.max_seq_length = int(self.max_seq_length)
        if self.test_max_seq_length is not None:
            opt.test_max_seq_length = int(self.test_max_seq_length)

        model = LMM(opt)

        ckpt_path = self.config.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"GenDoP checkpoint not found at '{ckpt_path}'. Download the "
                "released weights and point `checkpoint_path` at the resulting file."
            )

        if ckpt_path.endswith("safetensors"):
            ckpt = load_file(ckpt_path, device="cpu")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")

        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.info("[GenDoP] missing keys (%d): %s ...", len(missing), missing[:5])
        if unexpected:
            logger.info("[GenDoP] unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])

        model = model.half().eval().to(self.device)
        logger.info("Loaded GenDoP checkpoint from %s", ckpt_path)
        return model, opt

    def _tokens_to_c2ws(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse-quantize a flat GenDoP token sequence into a tensor of
        camera-to-world matrices ``[T, 4, 4]``."""
        expected_len = self.pose_length * 10

        if tokens.numel() != expected_len:
            fallback = (
                torch.tensor(
                    [256, 128, 128, 128, 128, 128, 128, 36, 64, 60],
                    dtype=torch.float32,
                    device=tokens.device,
                )
                / 256.0
                * self.discrete_bins
            )
            tokens = fallback.repeat(self.pose_length)

        coords = tokens.reshape(-1, 10).float()
        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1]

        temp_traj = coords_traj / (0.5 * self.discrete_bins) - 1.0
        temp_instri = coords_instri / (self.discrete_bins / 10.0)
        scale = torch.exp(coords_scale / self.discrete_bins * 4.0 - 2.0)

        # token_to_camera allocates helper tensors on CPU, so decode there.
        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1).unsqueeze(0).cpu()
        camera_pose = self._token_to_camera(
            camera_tokens, self.target_width, self.target_height
        )

        c2ws_34 = camera_pose[0, :, :12].reshape(-1, 3, 4).cpu().numpy()
        c2ws_34[:, :3, 3] = c2ws_34[:, :3, 3] * float(scale[0].item())

        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=c2ws_34.dtype)
        c2ws = np.stack([np.vstack([m, bottom]) for m in c2ws_34], axis=0)
        return torch.from_numpy(c2ws).float()

    @torch.no_grad()
    def generate_using_text(
        self,
        text_prompts: List[str],
        subject_trajectory: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cond_mode != "text":
            raise NotImplementedError(
                "Only cond_mode='text' is currently routed through the GenDoP "
                f"adapter; received cond_mode='{self.cond_mode}'."
            )

        generations = []
        for prompt in text_prompts:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                tokens = self.model.generate(
                    [prompt],
                    max_new_tokens=self.opt.test_max_seq_length,
                    clean=True,
                )

            token_seq = torch.as_tensor(tokens[0], device=self.device)
            if token_seq.numel() > 0:
                token_seq = token_seq[:-1]

            c2ws = self._tokens_to_c2ws(token_seq).to(self.device)
            generations.append(c2ws)

        return torch.stack(generations, dim=0)
