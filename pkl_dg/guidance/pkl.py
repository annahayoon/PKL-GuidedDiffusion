import torch
from .base import GuidanceStrategy


class PKLGuidance(GuidanceStrategy):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "ForwardModel",
        t: int,
    ) -> torch.Tensor:
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        ratio = y / (Ax_plus_B + self.epsilon)
        residual = 1.0 - ratio
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient

