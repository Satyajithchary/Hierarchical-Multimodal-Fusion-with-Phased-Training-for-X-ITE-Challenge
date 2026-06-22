"""
src/model.py
------------
Hierarchical Multimodal Pain Recognition model.

Architecture overview
─────────────────────
Three specialised unimodal encoders extract 256-d feature vectors:

  • EnhancedPhysioProcessor  — 1D CNN → BiLSTM → Multi-Head Attention
  • EfficientVideoProcessor  — MobileNetV3-Small (chunked, gradient-checkpointed)
                               → per-stream BiLSTM → MLP fusion
  • EnhancedAudioProcessor   — Log-Mel spectrogram → 2D CNN

These features feed a four-level hierarchical fusion:

  1. Unimodal classifiers        (3 heads)
  2. Pairwise fusion classifiers (3 heads)
  3. Full cross-modal attention  (1 head)
  4. Learnable weighted ensemble over all 7 logits

An auxiliary uncertainty head concatenates hidden activations from
all classifiers and outputs a scalar confidence score.

Reference
─────────
Chary et al., "Hierarchical Multimodal Fusion with Phased Training for
Automated Pain Recognition in the X-ITE Challenge," ACIIW 2025.
"""

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.models as models
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Unimodal Encoders
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedPhysioProcessor(nn.Module):
    """
    1D CNN-BiLSTM with Multi-Head Self-Attention for physiological signals.

    Input : [B, 4, 10 000]  (ECG, SCL, EMG-face, EMG-trap at 1 000 Hz)
    Output: [B, feature_dim]
    """

    def __init__(
        self,
        input_channels: int = 4,
        dropout_rate:   float = 0.3,
        feature_dim:    int = 256,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=51, padding=25)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, padding=5)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        self.lstm  = nn.LSTM(
            128, 128, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=dropout_rate,
        )
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.fc        = nn.Linear(256, feature_dim)
        self.dropout   = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        lstm_out, _   = self.lstm(x.transpose(1, 2))
        attn_out, _   = self.attention(lstm_out, lstm_out, lstm_out)
        return F.relu(self.fc(attn_out.mean(dim=1)))


class EfficientVideoProcessor(nn.Module):
    """
    MobileNetV3-Small frame encoder with per-stream BiLSTM temporal modelling.

    Processes four video streams (frontal, side, thermal, body) in memory-
    efficient 60-frame chunks using gradient checkpointing.

    Input : dict[str, Tensor[B, 3, T, 224, 224]]
    Output: Tensor[B, feature_dim]
    """

    STREAMS = ["frontal", "side", "thermal", "body"]

    def __init__(
        self,
        num_streams:      int = 4,
        feature_dim:      int = 256,
        lstm_hidden_dim:  int = 256,
        dropout_rate:     float = 0.3,
        chunk_size:       int = 60,
    ):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.feature_extractor = mobilenet.features
        # Freeze early layers to retain general features
        for param in self.feature_extractor[:5].parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.temporal_models = nn.ModuleDict({
            s: nn.LSTM(feature_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
            for s in self.STREAMS
        })
        self.fusion_layers = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2 * num_streams, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.chunk_size = chunk_size

    def _process_stream(
        self,
        stream_data: torch.Tensor,
        stream_name: str,
    ) -> torch.Tensor | None:
        """Process a single [B, C, T, H, W] stream."""
        B, C, T, H, W = stream_data.shape
        all_features = []

        for i in range(0, T, self.chunk_size):
            chunk = stream_data[:, :, i : i + self.chunk_size]
            b, c, t_chunk, h, w = chunk.shape
            if t_chunk == 0:
                continue

            chunk_flat = chunk.permute(0, 2, 1, 3, 4).reshape(b * t_chunk, c, h, w)

            def _extract(x):
                return self.feature_extractor(x)

            frame_feats = checkpoint(_extract, chunk_flat, use_reentrant=False)
            all_features.append(self.projection(frame_feats))

        if not all_features:
            return None

        seq = torch.cat(all_features, dim=0).view(B, T, -1)
        _, (h_n, _) = self.temporal_models[stream_name](seq)
        return torch.cat((h_n[0], h_n[1]), dim=1)            # [B, lstm_hidden*2]

    def forward(self, video_streams: dict) -> torch.Tensor:
        B = next(iter(video_streams.values())).size(0)
        stream_outputs = []

        for name, data in video_streams.items():
            if torch.all(data == 0):
                stream_outputs.append(
                    torch.zeros(B, 512, device=data.device)
                )
                continue

            feat = self._process_stream(data, name)
            stream_outputs.append(
                feat if feat is not None
                else torch.zeros(B, 512, device=data.device)
            )

        return self.fusion_layers(torch.cat(stream_outputs, dim=1))


class EnhancedAudioProcessor(nn.Module):
    """
    2D CNN on log-Mel spectrograms for ambient vocal expression.

    Input : [B, 160 000]  (audio at 16 kHz, 10 s)
    Output: [B, feature_dim]
    """

    def __init__(
        self,
        sample_rate:        int = 16_000,
        n_mels:             int = 128,
        dropout_rate:       float = 0.3,
        segment_length_sec: float = 10.0,
        feature_dim:        int = 256,
        augmentations=None,
    ):
        super().__init__()
        self.augmentations = augmentations

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=256
        )
        self.conv1 = nn.Conv2d(1, 32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128,3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout_rate)

        n_frames       = int(segment_length_sec * sample_rate / 256) + 1
        final_time_dim = n_frames // (self.pool.kernel_size ** 3)
        self.fc = nn.Linear(128 * (n_mels // 8) * final_time_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel  = self.mel_transform(x)
        log_mel = (mel + 1e-6).log().unsqueeze(1)

        if self.training and self.augmentations:
            log_mel = self.augmentations.augment_audio_spec(log_mel)

        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(log_mel)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))
        return F.relu(self.fc(x.view(x.size(0), -1)))


# ─────────────────────────────────────────────────────────────────────────────
# Full Hierarchical Model
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPainRecognition(nn.Module):
    """
    Hierarchical Multimodal Fusion model for binary pain classification.

    Prediction heads
    ─────────────────
    • 3 unimodal classifiers
    • 3 pairwise fusion classifiers
    • 1 full cross-modal attention classifier
    → 7 logits combined via a learnable weighted ensemble

    Auxiliary output
    ─────────────────
    • Uncertainty scalar (sigmoid) — exposed when ``return_uncertainty=True``.
    """

    MODALITIES = ["physio", "video", "audio"]

    def __init__(
        self,
        num_classes:        int = 2,
        dropout_rate:       float = 0.4,
        feature_dim:        int = 256,
        segment_length_sec: float = 10.0,
        augmentations=None,
    ):
        super().__init__()

        # ── Unimodal encoders ────────────────────────────────────────────────
        self.physio_processor = EnhancedPhysioProcessor(
            dropout_rate=dropout_rate, feature_dim=feature_dim
        )
        self.video_processor = EfficientVideoProcessor(
            feature_dim=feature_dim, lstm_hidden_dim=feature_dim
        )
        self.audio_processor = EnhancedAudioProcessor(
            dropout_rate=dropout_rate,
            segment_length_sec=segment_length_sec,
            feature_dim=feature_dim,
            augmentations=augmentations,
        )

        # ── Classifiers ───────────────────────────────────────────────────────
        self.classifiers = nn.ModuleDict({
            m: self._mlp(feature_dim, 128, num_classes)
            for m in self.MODALITIES
        })
        self.pairwise_fusions = nn.ModuleDict({
            f"{m1}_{m2}": self._mlp(feature_dim * 2, 256, num_classes)
            for m1, m2 in combinations(self.MODALITIES, 2)
        })
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8,
            dropout=dropout_rate, batch_first=True,
        )
        self.full_fusion_classifier = self._mlp(feature_dim * 3, 256, num_classes)

        # ── Ensemble weights (learnable) ──────────────────────────────────────
        n_heads = len(self.classifiers) + len(self.pairwise_fusions) + 1
        self.ensemble_weights = nn.Parameter(torch.ones(n_heads))

        # ── Uncertainty head ──────────────────────────────────────────────────
        total_hidden = (128 * 3) + (256 * 3) + 256
        self.uncertainty_head = nn.Linear(total_hidden, 1)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, out_dim),
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        physio: torch.Tensor,
        video:  dict,
        audio:  torch.Tensor,
        return_uncertainty: bool = False,
    ):
        """
        Parameters
        ----------
        physio : Tensor[B, 4, 10000]
        video  : dict[str, Tensor[B, 3, T, 224, 224]]
        audio  : Tensor[B, 160000]
        return_uncertainty : bool

        Returns
        -------
        logits : dict[str, Tensor[B, num_classes]]
            Keys: 'physio', 'video', 'audio', pairwise keys, 'full', 'ensemble'.
        features : dict[str, Tensor[B, feature_dim]]
        uncertainty : Tensor[B, 1]   (only when return_uncertainty=True)
        """
        # ── Encode ───────────────────────────────────────────────────────────
        physio_exists = physio.abs().sum() > 0
        video_exists  = any(v.abs().sum() > 0 for v in video.values())
        audio_exists  = audio.abs().sum()  > 0

        dev = physio.device
        physio_feat = (self.physio_processor(physio)  if physio_exists
                       else torch.zeros(physio.size(0), 256, device=dev))
        video_feat  = (self.video_processor(video)    if video_exists
                       else torch.zeros(physio.size(0), 256, device=dev))
        audio_feat  = (self.audio_processor(audio)    if audio_exists
                       else torch.zeros(physio.size(0), 256, device=dev))

        features = {"physio": physio_feat, "video": video_feat, "audio": audio_feat}

        # ── Unimodal classifiers ──────────────────────────────────────────────
        logits: dict         = {}
        hidden_features: dict= {}

        for m in self.MODALITIES:
            hidden_features[m] = F.relu(self.classifiers[m][0](features[m]))
            logits[m]          = self.classifiers[m](features[m])

        # ── Pairwise classifiers ──────────────────────────────────────────────
        for m1, m2 in combinations(self.MODALITIES, 2):
            key  = f"{m1}_{m2}"
            pair = torch.cat([features[m1], features[m2]], dim=1)
            hidden_features[key] = F.relu(self.pairwise_fusions[key][0](pair))
            logits[key]          = self.pairwise_fusions[key](pair)

        # ── Full cross-modal attention fusion ─────────────────────────────────
        stacked  = torch.stack(list(features.values()), dim=1)     # [B, 3, D]
        attended, _ = self.cross_modal_attention(stacked, stacked, stacked)
        fused    = attended.flatten(start_dim=1)                    # [B, 3*D]
        hidden_features["full"] = F.relu(self.full_fusion_classifier[0](fused))
        logits["full"]          = self.full_fusion_classifier(fused)

        # ── Learnable ensemble ────────────────────────────────────────────────
        all_logits = list(logits.values())
        weights    = F.softmax(self.ensemble_weights, dim=0)
        stacked_logits   = torch.stack(all_logits, dim=2)          # [B, C, n_heads]
        logits["ensemble"] = (stacked_logits * weights).sum(dim=2)

        if return_uncertainty:
            hidden_cat  = torch.cat(list(hidden_features.values()), dim=1)
            uncertainty = torch.sigmoid(self.uncertainty_head(hidden_cat))
            return logits, features, uncertainty

        return logits, features
