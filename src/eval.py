import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import DEV_TSV, get_device
from .dataset import CommonVoicePLDataset, mix_with_white_noise, SAMPLE_RATE
from .model import UNet1D
from .losses import si_snr_metric


def collate_eval(batch):
    return batch  # batch = list dictów (batch_size=1)


def main():
    device = get_device()
    print("Using device:", device)

    # ----------------------------------------
    # 1) Wczytanie modelu
    # ----------------------------------------
    model_path = os.path.join("models", "unet_pl.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Nie znaleziono {model_path}. Najpierw uruchom trening (python -m src.train)."
        )

    model = UNet1D(in_channels=1, base_channels=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model załadowany z {model_path}")

    # ----------------------------------------
    # 2) Dane walidacyjne (clean)
    # ----------------------------------------
    dev_clean_full = CommonVoicePLDataset(DEV_TSV)

    # --- USTAWIENIE --- 
    # None  → wszystkie próbki
    # 1000 → losowe 1000 próbek (szybciej)
    N_SAMPLES = None

    if N_SAMPLES is None or N_SAMPLES > len(dev_clean_full):
        indices = list(range(len(dev_clean_full)))
    else:
        indices = random.sample(range(len(dev_clean_full)), k=N_SAMPLES)

    print(f"Liczba przykładów do ewaluacji: {len(indices)}\n")

    subset = torch.utils.data.Subset(dev_clean_full, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_eval)

    # ----------------------------------------
    # 3) Ewaluacja z tqdm
    # ----------------------------------------
    total_sisnr_noisy = 0.0
    total_sisnr_enh = 0.0
    count = 0

    progress = tqdm(loader, desc="Ewaluacja SI-SNR", ncols=100)

    for batch in progress:
        item = batch[0]
        clean_full = item["audio"]

        # ograniczamy długość, jeśli nagrania są bardzo długie:
        MAX_SECONDS = 4.0
        max_len = int(MAX_SECONDS * SAMPLE_RATE)
        if clean_full.shape[0] > max_len:
            clean_seg = clean_full[:max_len]
        else:
            clean_seg = clean_full

        noisy, clean_seg = mix_with_white_noise(clean_seg, snr_db=5.0)

        with torch.no_grad():
            est = model(noisy.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0).squeeze(0)

        min_len = min(clean_seg.shape[0], noisy.shape[0], est.shape[0])
        clean_seg = clean_seg[:min_len]
        noisy = noisy[:min_len]
        est = est[:min_len]

        sisnr_noisy = si_snr_metric(
            noisy.unsqueeze(0).unsqueeze(0),
            clean_seg.unsqueeze(0).unsqueeze(0),
        )
        sisnr_enh = si_snr_metric(
            est.unsqueeze(0).unsqueeze(0),
            clean_seg.unsqueeze(0).unsqueeze(0),
        )

        total_sisnr_noisy += sisnr_noisy
        total_sisnr_enh += sisnr_enh
        count += 1

        # aktualizacja opisu paska:
        improvement = (total_sisnr_enh / count) - (total_sisnr_noisy / count)
        progress.set_postfix({
            "avg noisy": f"{total_sisnr_noisy / count:.2f}",
            "avg enh": f"{total_sisnr_enh / count:.2f}",
            "improve": f"{improvement:.2f} dB"
        })

    avg_noisy = total_sisnr_noisy / max(count, 1)
    avg_enh = total_sisnr_enh / max(count, 1)
    improvement = avg_enh - avg_noisy

    print("\n========== RAPORT JAKOŚCI (SI-SNR) ==========")
    print(f"Liczba przykładów: {count}")
    print(f"Średni SI-SNR noisy   : {avg_noisy:.2f} dB")
    print(f"Średni SI-SNR enhanced: {avg_enh:.2f} dB")
    print(f"Średnia poprawa       : {improvement:.2f} dB")
    print("=============================================\n")


if __name__ == "__main__":
    main()
