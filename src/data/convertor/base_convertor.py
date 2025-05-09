from abc import ABC, abstractmethod
from typing import Optional
import torch

class BaseConvertor(ABC):
    """
    Abstract base class for conversion between CCDM and transform formats.
    """
    @abstractmethod
    def to_standard(
        self,
        trajectory: torch.Tensor,
        subject_trajectory: Optional[torch.Tensor] = None,
        subject_volume: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert input to standard transform format."""
        pass

    @abstractmethod
    def from_standard(
        self,
        transform: torch.Tensor,
        subject_trajectory: Optional[torch.Tensor] = None,
        subject_volume: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert from standard transform format to target dataset format."""
        pass
