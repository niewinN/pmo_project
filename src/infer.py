import os
import random
import soundfile as sf

import torch

from .utils import DEV_TSV, OUTPUT_DIR, MODELS_DIR, get_device
from .dataset import CommonVoicePLDataset, mix_with_white_noise, SAMPLE_RATE
from .model import UNet1D
from .losses import si_snr_metric


def save_wav(tensor, path, sr=SAMPLE_RATE):
    tensor = tensor.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, tensor, sr)


def main():
    device = get_device()
    print(f"Using device: {device}")

    #Wczytanie wytrenowanego modelu
    model_path = os.path.join(MODELS_DIR, "unet_pl.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Nie znaleziono {model_path}. Najpierw uruchom trening (python -m src.train)."
        )

    model = UNet1D(in_channels=1, base_channels=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model załadowany z {model_path}")

    #Dane z dev-setu (czyste nagrania)
    dev_clean = CommonVoicePLDataset(DEV_TSV)

    indices = random.sample(range(len(dev_clean)), k=3)

    MAX_SECONDS = 4.0
    MAX_LEN = int(MAX_SECONDS * SAMPLE_RATE)

    for i, idx in enumerate(indices):
        item = dev_clean[idx]
        clean_full = item["audio"]  # pełne nagranie [T]
        sr = SAMPLE_RATE

        if clean_full.shape[0] > MAX_LEN:
            clean_seg = clean_full[:MAX_LEN]
        else:
            clean_seg = clean_full

        #Dodajemy biały szum (SNR ~ 5 dB)
        noisy, clean_seg = mix_with_white_noise(clean_seg, snr_db=5.0)

        with torch.no_grad():
            est = model(noisy.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0).squeeze(0)

        min_len = min(clean_seg.shape[0], noisy.shape[0], est.shape[0])
        clean_seg = clean_seg[:min_len]
        noisy = noisy[:min_len]
        est = est[:min_len]

        #Metryki SI-SNR
        sisnr_noisy = si_snr_metric(
            noisy.unsqueeze(0).unsqueeze(0), clean_seg.unsqueeze(0).unsqueeze(0)
        )
        sisnr_est = si_snr_metric(
            est.unsqueeze(0).unsqueeze(0), clean_seg.unsqueeze(0).unsqueeze(0)
        )

        print(f"Przykład {i}:")
        print(f"  SI-SNR noisy vs clean : {sisnr_noisy:.2f} dB")
        print(f"  SI-SNR model vs clean : {sisnr_est:.2f} dB")

        #Zapis plików WAV
        base = os.path.join(OUTPUT_DIR, f"example_{i}")
        save_wav(clean_seg, base + "_clean.wav", sr)
        save_wav(noisy, base + "_noisy.wav", sr)
        save_wav(est, base + "_enhanced.wav", sr)
        print(f"  Zapisano: {base}_clean/noisy/enhanced.wav")


if __name__ == "__main__":
    main()
