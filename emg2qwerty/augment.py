import torch
from torch import nn


class EMGAugment(nn.Module):
    """
    Augment spectrogram-like EMG inputs with shape:
        (T, N, B, C, F)
    where
        T = time steps
        N = batch size
        B = number of bands
        C = electrode channels
        F = frequency bins

    These augmentations are label-preserving enough for CTC training.
    """

    def __init__(
        self,
        p_apply: float = 0.8,
        noise_std: float = 0.01,
        gain_jitter_min: float = 0.9,
        gain_jitter_max: float = 1.1,
        channel_dropout_p: float = 0.1,
        max_time_mask_width: int = 12,
        max_freq_mask_width: int = 4,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> None:
        super().__init__()
        self.p_apply = p_apply
        self.noise_std = noise_std
        self.gain_jitter_min = gain_jitter_min
        self.gain_jitter_max = gain_jitter_max
        self.channel_dropout_p = channel_dropout_p
        self.max_time_mask_width = max_time_mask_width
        self.max_freq_mask_width = max_freq_mask_width
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, B, C, F)
        T = x.size(0)
        if T <= 1 or self.max_time_mask_width <= 0:
            return x

        for _ in range(self.num_time_masks):
            width = torch.randint(
                low=0,
                high=min(self.max_time_mask_width, T) + 1,
                size=(1,),
                device=x.device,
            ).item()
            if width == 0:
                continue
            start = torch.randint(
                low=0,
                high=T - width + 1,
                size=(1,),
                device=x.device,
            ).item()
            x[start:start + width, ...] = 0
        return x

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, B, C, F)
        F = x.size(-1)
        if F <= 1 or self.max_freq_mask_width <= 0:
            return x

        for _ in range(self.num_freq_masks):
            width = torch.randint(
                low=0,
                high=min(self.max_freq_mask_width, F) + 1,
                size=(1,),
                device=x.device,
            ).item()
            if width == 0:
                continue
            start = torch.randint(
                low=0,
                high=F - width + 1,
                size=(1,),
                device=x.device,
            ).item()
            x[..., start:start + width] = 0
        return x

    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, B, C, F)
        if self.channel_dropout_p <= 0:
            return x

        _, N, B, C, _ = x.shape
        keep_mask = (
            torch.rand((N, B, C), device=x.device) > self.channel_dropout_p
        ).float()
        keep_mask = keep_mask.unsqueeze(0).unsqueeze(-1)  # (1, N, B, C, 1)
        return x * keep_mask

    def _gain_jitter(self, x: torch.Tensor) -> torch.Tensor:
        _, N, B, C, _ = x.shape
        gains = torch.empty((N, B, C), device=x.device).uniform_(
            self.gain_jitter_min,
            self.gain_jitter_max,
        )
        gains = gains.unsqueeze(0).unsqueeze(-1)  # (1, N, B, C, 1)
        return x * gains

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return x
        return x + torch.randn_like(x) * self.noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        if torch.rand(1, device=x.device).item() > self.p_apply:
            return x

        x = x.clone()
        x = self._gain_jitter(x)
        x = self._gaussian_noise(x)
        x = self._channel_dropout(x)
        x = self._time_mask(x)
        x = self._freq_mask(x)
        return x
