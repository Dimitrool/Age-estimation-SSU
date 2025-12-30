import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    """Container for loss outputs."""
    loss: torch.Tensor


class BaseLoss:
    """Base class for all loss functions."""

    def weighted_mean(self, term: torch.Tensor, W: torch.Tensor | None = None) -> torch.Tensor:
        """Computes the mean of a tensor, applying weights W if provided."""
        if W is not None:
            return (term * W).mean()
        return term.mean()

    def __call__(self, age_pred: torch.Tensor, age_true: torch.Tensor, W: torch.Tensor | None = None) -> LossOutput:
        raise NotImplementedError("Subclasses must implement __call__ method.")


class _Elementwise(BaseLoss):
    def _penalty(self, diff: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, age_pred: torch.Tensor, age_true: torch.Tensor, W: torch.Tensor | None = None) -> LossOutput:
        resid = self._penalty(age_pred - age_true)
        loss = self.weighted_mean(resid, W)
        return LossOutput(loss=loss)


class MSELoss(_Elementwise):
    def _penalty(self, diff): return diff.pow(2)


class L1Loss(_Elementwise):
    def _penalty(self, diff): return diff.abs()


class CustomLoss(_Elementwise):
    def _penalty(self, diff): return diff.pow(2)

