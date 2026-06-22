"""
src/dataset.py
--------------
Dataset class and data augmentation pipelines for the X-ITE Pain
Recognition task. Each sample is a 10-second multimodal window
containing physiological signals, four video streams, and audio.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import librosa

import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class PainDataAugmentation:
    """
    Modality-specific augmentation pipelines.

    Augmentations are only applied during training:
      - Physiological: additive Gaussian noise (std = 1 % of signal std).
      - Video:         random horizontal flip + colour jitter.
      - Audio spectrogram: SpecAugment (time & frequency masking).
    """

    def __init__(self, training: bool = True):
        self.training = training
        if self.training:
            self.video_aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
            self.audio_aug = T.Compose([
                torchaudio.transforms.TimeMasking(time_mask_param=80),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            ])

    def augment_physio(self, physio_tensor: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return physio_tensor
        noise = torch.randn_like(physio_tensor) * 0.01 * torch.std(physio_tensor)
        return physio_tensor + noise

    def augment_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """video_tensor shape: [C, T, H, W]"""
        if not self.training:
            return video_tensor
        video_tensor = video_tensor.permute(1, 0, 2, 3)          # [T, C, H, W]
        augmented = [self.video_aug(frame) for frame in video_tensor]
        return torch.stack(augmented, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]

    def augment_audio_spec(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        return self.audio_aug(spec)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class OptimizedPainRecognitionDataset(Dataset):
    """
    Subject-independent dataset for the X-ITE Pain Challenge.

    Directory layout assumed::

        <data_path>/
          low_pain/
            S*/
              ecg_*.csv   scl_*.csv   emg_face_*.csv   emg_trap_*.csv
              fvf/*.mp4   fvs/*.mp4   ft/*.mp4          bdy/*.mp4
              audio/*.wav
          high_pain/
            ...

    Parameters
    ----------
    data_path : str | Path
        Root directory of the dataset.
    mode : str
        One of ``'train'``, ``'val'``, ``'test'``.
    segment_length_sec : float
        Length of each data window in seconds.
    train_split, val_split : float
        Fraction of subjects for training and validation.
        Test fraction = 1 − train_split − val_split.
    random_seed : int
        Seed for reproducible subject splits.
    augmentations : PainDataAugmentation | None
        Augmentation pipeline; a training-mode instance is created
        automatically when ``None`` and ``mode == 'train'``.
    """

    AUDIO_SR    = 16_000   # Hz
    VIDEO_FPS   = 30       # frames / second
    PHYSIO_SR   = 1_000    # Hz

    STREAM_MAP = {"fvf": "frontal", "fvs": "side", "ft": "thermal", "bdy": "body"}

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        segment_length_sec: float = 10.0,
        train_split: float = 0.70,
        val_split:   float = 0.15,
        random_seed: int = 42,
        augmentations: PainDataAugmentation | None = None,
    ):
        self.data_path          = Path(data_path)
        self.mode               = mode
        self.segment_length_sec = segment_length_sec
        self.augmentations      = augmentations or PainDataAugmentation(training=(mode == "train"))

        self.target_audio_len   = int(segment_length_sec * self.AUDIO_SR)
        self.target_video_frames= int(segment_length_sec * self.VIDEO_FPS)
        self.target_physio_len  = int(segment_length_sec * self.PHYSIO_SR)

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.subject_file_map = self._create_file_map()
        self.samples          = self._create_split_samples(train_split, val_split)

        if not self.samples:
            print(f"WARNING: No samples found for mode='{self.mode}'. "
                  "Check data_path and directory structure.")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _create_file_map(self) -> dict:
        subject_file_map: dict = {}
        for file_path in self.data_path.rglob("S*/*.*"):
            try:
                subject_id  = file_path.parent.name
                segment_key = "_".join(file_path.stem.split("_")[:2])
                pain_level  = 0 if "low_pain"  in file_path.parent.parent.parent.name else 1
                subject_file_map.setdefault(subject_id, {})
                if segment_key not in subject_file_map[subject_id]:
                    subject_file_map[subject_id][segment_key] = {
                        "files": [], "pain_level": pain_level
                    }
                subject_file_map[subject_id][segment_key]["files"].append(file_path)
            except (IndexError, AttributeError):
                continue
        return subject_file_map

    def _create_split_samples(self, train_split: float, val_split: float) -> list:
        if not self.subject_file_map:
            return []

        unique_subjects = sorted(self.subject_file_map.keys())
        np.random.shuffle(unique_subjects)

        n_train = int(len(unique_subjects) * train_split)
        n_val   = int(len(unique_subjects) * val_split)

        splits = {
            "train": set(unique_subjects[:n_train]),
            "val":   set(unique_subjects[n_train : n_train + n_val]),
            "test":  set(unique_subjects[n_train + n_val :]),
        }
        target = splits.get(self.mode, set())

        samples = [
            {
                "subject_id":  sid,
                "segment_key": sk,
                "pain_level":  data["pain_level"],
                "files":       data["files"],
            }
            for sid in target
            if sid in self.subject_file_map
            for sk, data in self.subject_file_map[sid].items()
        ]
        print(
            f"[{self.mode:5s}] subjects={len(target):3d}  samples={len(samples):5d}"
        )
        return samples

    def _pad_or_truncate(self, array: np.ndarray, target_len: int) -> np.ndarray:
        if len(array) >= target_len:
            return array[:target_len]
        return np.pad(array, (0, target_len - len(array)), "constant")

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_physio_segment(self, files: list) -> torch.Tensor:
        physio = torch.zeros(4, self.target_physio_len, dtype=torch.float32)
        sig_paths = {"ecg": None, "scl": None, "emg_face": None, "emg_trap": None}

        for f in files:
            n = f.name.lower()
            if   "ecg"  in n:                     sig_paths["ecg"]      = f
            elif "scl"  in n:                     sig_paths["scl"]      = f
            elif "emg"  in n and "face" in n:     sig_paths["emg_face"] = f
            elif "emg"  in n and "trap" in n:     sig_paths["emg_trap"] = f

        for ch, key in enumerate(["ecg", "scl", "emg_face", "emg_trap"]):
            if sig_paths[key]:
                try:
                    data = (
                        pd.read_csv(sig_paths[key], header=None, on_bad_lines="skip")
                        .iloc[:, 0].values.astype(np.float32)
                    )
                    physio[ch] = torch.from_numpy(
                        self._pad_or_truncate(data, self.target_physio_len)
                    )
                except Exception:
                    continue
        return physio

    def _load_video_segment(self, files: list) -> dict:
        streams = {
            s: torch.zeros(3, self.target_video_frames, 224, 224)
            for s in ["frontal", "side", "thermal", "body"]
        }
        for f in files:
            if f.suffix not in {".mp4", ".avi", ".mov"}:
                continue
            for key, name in self.STREAM_MAP.items():
                if key in str(f.parent.parent.name):
                    try:
                        cap    = cv2.VideoCapture(str(f))
                        frames = []
                        while len(frames) < self.target_video_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_rgb = cv2.cvtColor(
                                cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB
                            )
                            frames.append(
                                torch.from_numpy(frame_rgb).permute(2, 0, 1) / 255.0
                            )
                        cap.release()
                        if frames:
                            while len(frames) < self.target_video_frames:
                                frames.append(frames[-1])
                            streams[name] = torch.stack(frames, dim=1)
                    except Exception:
                        continue
        return streams

    def _load_audio_segment(self, files: list) -> torch.Tensor:
        for f in files:
            if f.suffix in {".wav", ".mp3", ".m4a"}:
                try:
                    audio, _ = librosa.load(
                        str(f), sr=self.AUDIO_SR, duration=self.segment_length_sec
                    )
                    return torch.from_numpy(
                        self._pad_or_truncate(audio, self.target_audio_len)
                    )
                except Exception:
                    break
        return torch.zeros(self.target_audio_len)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        physio = self.augmentations.augment_physio(
            self._load_physio_segment(sample["files"])
        )
        video = {
            k: self.augmentations.augment_video(v)
            for k, v in self._load_video_segment(sample["files"]).items()
        }
        audio = self._load_audio_segment(sample["files"])

        return {
            "physio":     physio,
            "video":      video,
            "audio":      audio,
            "label":      torch.tensor(sample["pain_level"], dtype=torch.long),
            "subject_id": sample["subject_id"],
        }
