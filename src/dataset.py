
import os
import random
from typing import Dict

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

from .utils import CLIPS_DIR, TRAIN_TSV, DEV_TSV

class CommonVoicePLDataset(Dataset):
    """Dataset ładujący czyste nagrania z polskiego Common Voice."""

    def __init__(self, tsv_path: str, clips_dir: str = CLIPS_DIR, sample_rate: int = 16000):
        super().__init__()
        self.df = pd.read_csv(tsv_path, sep="\t").reset_index(drop=True)
        self.clips_dir = clips_dir
        self.sample_rate = sample_rate
        self.resampler = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        filename = row["path"]
        text = row.get("sentence", "")
        full_path = os.path.join(self.clips_dir, filename)

        waveform, sr = torchaudio.load(full_path)  # [C, T]

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            if self.resampler is None:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)

        return {
            "audio": waveform.squeeze(0),  # [T]
            "text": text,
            "id": filename,
        }

# Enhancement dataset
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 2.0 
SEGMENT_LEN = int(SAMPLE_RATE * SEGMENT_SECONDS)


def mix_with_white_noise(clean_waveform: torch.Tensor, snr_db: float):
    """Dodaje biały szum o zadanym SNR (w dB)."""
    if clean_waveform.dim() == 2:
        clean_waveform = clean_waveform.squeeze(0)
    clean = clean_waveform.float()

    noise = torch.randn_like(clean)

    rms_clean = (clean**2).mean().sqrt()
    rms_noise = (noise**2).mean().sqrt() + 1e-8

    snr_linear = 10 ** (snr_db / 20.0)
    desired_rms_noise = rms_clean / snr_linear
    noise = noise * (desired_rms_noise / rms_noise)

    noisy = clean + noise
    return noisy, clean

def random_segment(wave: torch.Tensor, segment_len: int = SEGMENT_LEN):
    if wave.dim() == 2:
        wave = wave.squeeze(0)
    length = wave.shape[0]
    if length == segment_len:
        return wave
    if length < segment_len:
        pad_len = segment_len - length
        return torch.cat([wave, torch.zeros(pad_len)], dim=0)
    start = random.randint(0, length - segment_len)
    return wave[start:start+segment_len]

class PolishEnhancementDataset(Dataset):
    """Dataset generujący pary (noisy, clean) z czystych nagrań."""

    def __init__(self, base_dataset: Dataset, segment_len: int = SEGMENT_LEN,
                 snr_min: float = 0.0, snr_max: float = 10.0):
        super().__init__()
        self.base = base_dataset
        self.segment_len = segment_len
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        wave = item["audio"]  # [T]

        wave_seg = random_segment(wave, self.segment_len)

        snr_db = random.uniform(self.snr_min, self.snr_max)
        noisy_seg, clean_seg = mix_with_white_noise(wave_seg, snr_db)

        noisy_seg = noisy_seg.unsqueeze(0)  # [1, T]
        clean_seg = clean_seg.unsqueeze(0)  # [1, T]

        return noisy_seg, clean_seg
