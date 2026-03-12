from collections.abc import Sequence
from typing import ClassVar

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty.charset import charset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
)


class LSTMEncoder(nn.Module):
    """
    Input:  (T, N, F)
    Output: (T, N, H_out)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # PyTorch only uses LSTM dropout when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, F)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class LSTMCTCModule(pl.LightningModule):
    """
    Recurrent replacement for the repo's TDSConvCTCModule.

    Keeps:
      - same frontend normalization
      - same multi-band MLP frontend
      - same CTC loss / decoder / metrics
    Replaces:
      - TDSConvEncoder -> BiLSTM encoder
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        lstm_hidden_size: int,
        lstm_num_layers: int,
        bidirectional: bool,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler

        frontend_features = self.NUM_BANDS * mlp_features[-1]
        encoder_out_features = lstm_hidden_size * (2 if bidirectional else 1)

        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # -> (T, N, frontend_features)
        )

        self.encoder = LSTMEncoder(
            input_size=frontend_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(encoder_out_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)      # (T, N, F)
        x = self.encoder(x)            # (T, N, H)
        x = self.classifier(x)         # (T, N, C)
        return x

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        emissions = self.forward(inputs)

        # No temporal downsampling here, so lengths stay the same.
        # Keeping this explicit makes it easy to change later.
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,                  # (T, N, C)
            targets=targets.transpose(0, 1),     # (T, N) -> (N, T)
            input_lengths=emission_lengths,      # (N,)
            target_lengths=target_lengths,       # (N,)
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        metrics.update(
            preds=predictions,
            target=targets.detach().cpu().numpy(),
            target_lengths=target_lengths.detach().cpu().numpy(),
        )

        self.log(
            f"{phase}/loss",
            loss,
            prog_bar=(phase != "train"),
            on_step=False,
            on_epoch=True,
            batch_size=len(input_lengths),
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(input_lengths),
        )

        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._step("test", batch)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())

        if self.lr_scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(self.lr_scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
