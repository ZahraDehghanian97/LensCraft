import torch

from data.convertor.utils import handle_single_or_batch
from data.convertor.base_convertor import BaseConvertor


class GenDoPConvertor(BaseConvertor):
    DEFAULT_SUBJECT_VOLUME = (0.5, 1.7, 0.3)

    def __init__(self):
        super().__init__()

    @handle_single_or_batch(arg_specs=[(1, 3), (2, 2)])
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = trajectory.device, trajectory.dtype
        batch_size, seq_len = trajectory.shape[:2]

        transform = trajectory.clone()
        transform[..., :3, 1:3] = -transform[..., :3, 1:3]

        subject_transform = (
            torch.eye(4, device=device, dtype=dtype)
            .expand(batch_size, seq_len, 4, 4)
            .clone()
        )
        subject_transform[..., :3, 3] = 0.0

        if subject_volume is None:
            subject_volume = torch.tensor(
                self.DEFAULT_SUBJECT_VOLUME, dtype=dtype, device=device
            ).expand(batch_size, -1).clone()

        return transform, subject_transform, subject_volume

    @handle_single_or_batch(arg_specs=[(1, 3), (2, 3)])
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        out = transform.clone()
        out[..., :3, 1:3] = -out[..., :3, 1:3]  # OpenCV -> OpenGL
        return out, None, None
