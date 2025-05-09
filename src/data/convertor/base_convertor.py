from abc import ABC, abstractmethod

import torch

class BaseConvertor(ABC):
    """
    Abstract base class for conversion between CCDM and transform formats.
    """
    @abstractmethod
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None,
        *args,
        **kwargs
    ):
        """Convert input to standard transform format."""
        pass

    @abstractmethod
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: torch.Tensor | None = None,
        subject_volume: torch.Tensor | None = None,
        *args,
        **kwargs
    ):
        """Convert from standard transform format to target dataset format."""
        pass
