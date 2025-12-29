import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    """Container for loss outputs."""
    main_loss: torch.Tensor


class BaseLoss:
    """Base class for all loss functions."""

    def weighted_mean(self, term: torch.Tensor, W: torch.Tensor | None) -> torch.Tensor:
        """Computes the mean of a tensor, applying weights W if provided."""
        if W is not None:
            return (term * W).mean()
        return term.mean()

    def __call__(self, u_pred: torch.Tensor, u_gt: torch.Tensor, W: torch.Tensor | None) -> LossOutput:
        raise NotImplementedError("Subclasses must implement __call__ method.")



