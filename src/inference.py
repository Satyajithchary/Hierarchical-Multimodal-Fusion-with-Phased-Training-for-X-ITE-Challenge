"""
src/inference.py
----------------
Inference pipeline for the X-ITE Pain Challenge submission.

Loads a trained checkpoint, runs predictions on the challenge test set,
and writes results to a per-subject Excel file in the format required by
the organisers.

Expected test data layout
─────────────────────────
<TEST_DATA_PATH>/
  audio/<subject_id>/<segment>.wav
  bio/<subject_id>/<segment>.mat
  fvf/<subject_id>/<segment>.mp4
  fvs/<subject_id>/<segment>.mp4
  ft/<subject_id>/<segment>.mp4
  bdy/<subject_id>/<segment>.mp4
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import cv2
import librosa
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from src.model import HierarchicalPainRecognition


# ─────────────────────────────────────────────────────────────────────────────
# Challenge Test Dataset
# ─────────────────────────────────────────────────────────────────────────────

class XITEChallengeTestDataset(Dataset):
    """
    Reads the challenge test splits (subjects S006, S075, S112, S117) from
    the folder structure provided by the organisers.

    Parameters
    ----------
    data_path         : str | Path  Root directory of the challenge test data.
    segment_length_sec: float       Length of each segment window (seconds).
    """

    AUDIO_SR    = 16_000
    VIDEO_FPS   = 30
    PHYSIO_SR   = 1_000
    STREAM_MAP  = {"fvf": "frontal", "fvs": "side", "ft": "thermal", "bdy": "body"}

    def __init__(self, data_path: str, segment_length_sec: float = 10.0):
        self.data_path          = Path(data_path)
        self.segment_length_sec = segment_length_sec
        self.target_audio_len   = int(segment_length_sec * self.AUDIO_SR)
        self.target_video_frames= int(segment_length_sec * self.VIDEO_FPS)
        self.target_physio_len  = int(segment_length_sec * self.PHYSIO_SR)

        self.samples = self._scan()
        if not self.samples:
            raise FileNotFoundError(
                f"No test segments found under '{self.data_path}'.\n"
                "Check that the path and directory structure are correct."
            )
        n_subj = len({s["subject_id"] for s in self.samples})
        print(f"Found {len(self.samples)} test segments across {n_subj} subjects.")

    def _scan(self) -> list:
        samples = []
        for wav in sorted(self.data_path.glob("audio/S*/*.wav")):
            samples.append({
                "subject_id":  wav.parent.name,
                "segment_key": wav.stem,
            })
        return samples

    def __len__(self)  -> int:  return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]
        return {
            "physio":     self._load_physio(info),
            "video":      self._load_video(info),
            "audio":      self._load_audio(info),
            "segment_id": info["segment_key"],
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pad_or_truncate(self, arr: np.ndarray, n: int) -> np.ndarray:
        return arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)))

    def _load_physio(self, info: dict) -> torch.Tensor:
        physio = torch.zeros(4, self.target_physio_len, dtype=torch.float32)
        mat_path = (
            self.data_path / "bio" / info["subject_id"] / f"{info['segment_key']}.mat"
        )
        if mat_path.exists():
            try:
                data = loadmat(mat_path)["data"]
                if data.shape[0] < data.shape[1]:
                    data = data.T
                # Column order in .mat: [emg_face, emg_trap, scl, ecg, ...]
                # Reindex to match training order: [ecg=4, scl=3, emg_face=0, emg_trap=2]
                for ch, col in enumerate([4, 3, 0, 2]):
                    if col < data.shape[1]:
                        physio[ch] = torch.from_numpy(
                            self._pad_or_truncate(
                                data[:, col].astype(np.float32),
                                self.target_physio_len,
                            )
                        )
            except Exception as e:
                print(f"  Warning: could not load bio {mat_path}: {e}")
        return physio

    def _load_video(self, info: dict) -> dict:
        streams = {
            s: torch.zeros(3, self.target_video_frames, 224, 224)
            for s in ["frontal", "side", "thermal", "body"]
        }
        for key, name in self.STREAM_MAP.items():
            vpath = (
                self.data_path / key / info["subject_id"] / f"{info['segment_key']}.mp4"
            )
            if not vpath.exists():
                continue
            try:
                cap, frames = cv2.VideoCapture(str(vpath)), []
                while len(frames) < self.target_video_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(
                        torch.from_numpy(
                            cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
                        ).permute(2, 0, 1) / 255.0
                    )
                cap.release()
                if frames:
                    while len(frames) < self.target_video_frames:
                        frames.append(frames[-1])
                    streams[name] = torch.stack(frames, dim=1)
            except Exception as e:
                print(f"  Warning: could not load video {vpath}: {e}")
        return streams

    def _load_audio(self, info: dict) -> torch.Tensor:
        apath = (
            self.data_path / "audio" / info["subject_id"] / f"{info['segment_key']}.wav"
        )
        if apath.exists():
            try:
                audio, _ = librosa.load(
                    str(apath), sr=self.AUDIO_SR, duration=self.segment_length_sec
                )
                return torch.from_numpy(
                    self._pad_or_truncate(audio, self.target_audio_len)
                )
            except Exception as e:
                print(f"  Warning: could not load audio {apath}: {e}")
        return torch.zeros(self.target_audio_len)


# ─────────────────────────────────────────────────────────────────────────────
# Submission generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_submission(config: dict) -> None:
    """
    Run inference and write the per-subject Excel submission file.

    Parameters
    ----------
    config : dict
        Must contain: MODEL_PATH, TEST_DATA_PATH, TEAM_NAME,
        batch_size, num_workers, segment_length_sec, device.
    """
    print("=" * 60)
    print("X-ITE Pain Challenge — Submission File Generator")
    print("=" * 60)

    model_path = config["MODEL_PATH"]
    if not Path(model_path).exists():
        print(f"FATAL: model checkpoint not found at '{model_path}'")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    device = config["device"]
    ckpt   = torch.load(model_path, map_location=device)
    mcfg   = ckpt.get("config", {})

    model = HierarchicalPainRecognition(
        num_classes=mcfg.get("num_classes", 2),
        dropout_rate=mcfg.get("dropout_rate", 0.4),
        segment_length_sec=mcfg.get("segment_length_sec", 10.0),
        feature_dim=mcfg.get("feature_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Model loaded from: {model_path}")

    # ── Dataset / loader ──────────────────────────────────────────────────────
    test_ds = XITEChallengeTestDataset(
        data_path=config["TEST_DATA_PATH"],
        segment_length_sec=config.get("segment_length_sec", 10.0),
    )
    loader = DataLoader(
        test_ds,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    results = []
    print(f"\nRunning inference on {len(test_ds)} segments...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            seg_id = batch["segment_id"][0]
            print(f"  [{i+1:4d}/{len(loader)}] {seg_id}", end="\r")
            logits = model(
                batch["physio"].to(device),
                {k: v.to(device) for k, v in batch["video"].items()},
                batch["audio"].to(device),
            )
            probs = F.softmax(logits["ensemble"], dim=1).squeeze()
            results.append({
                "Segment ID":      seg_id,
                "Prob_PL_1":       round(probs[0].item(), 4),
                "Prob_PL_2":       round(probs[1].item(), 4),
                "Predicted_Label": "PL_1" if probs[0] > probs[1] else "PL_2",
            })
    print("\nInference complete.")

    # ── Write Excel ───────────────────────────────────────────────────────────
    team_name = config.get("TEAM_NAME", "MINDH_Lab")
    out_path  = f"{team_name}_Results.xlsx"

    df = pd.DataFrame(results)
    df["_seg_num"] = df["Segment ID"].apply(
        lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1
    )
    df = df.sort_values("_seg_num").drop(columns="_seg_num")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for subj in ["S006", "S075", "S112", "S117"]:
            sub_df = df[df["Segment ID"].str.startswith(subj)].copy()
            if sub_df.empty:
                print(f"  Warning: no segments found for subject {subj}.")
                continue
            sub_df["Segment ID"] = sub_df["Segment ID"].apply(
                lambda x: f"{subj}_seg{x.split('_')[-1]}"
            )
            sub_df[["Segment ID", "Prob_PL_1", "Prob_PL_2", "Predicted_Label"]].to_excel(
                writer, sheet_name=subj, index=False
            )
            print(f"  Sheet '{subj}': {len(sub_df)} rows written.")

    print(f"\nSubmission file saved → {out_path}")
    print("=" * 60)
