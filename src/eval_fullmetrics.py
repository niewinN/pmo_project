import os
import random
import datetime
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pystoi import stoi

from .utils import DEV_TSV, get_device
from .dataset import CommonVoicePLDataset, mix_with_white_noise, SAMPLE_RATE
from .model import UNet1D
from .losses import si_snr_metric
import pandas as pd



def collate_eval(batch):
    return batch 


def compute_stoi(clean, est, sr):
    """
    clean, est: numpy 1D, ten sam sr
    STOI wg biblioteki pystoi (0..1, im więcej tym lepiej)
    """
    target_sr = 10000
    if sr != target_sr:
        clean_rs = librosa.resample(clean, orig_sr=sr, target_sr=target_sr)
        est_rs = librosa.resample(est, orig_sr=sr, target_sr=target_sr)
        sr_use = target_sr
    else:
        clean_rs = clean
        est_rs = est
        sr_use = sr

    return float(stoi(clean_rs, est_rs, sr_use, extended=False))


def main():
    device = get_device()
    print("Using device:", device)


    model_path = os.path.join("models", "unet_pl.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Nie znaleziono {model_path}. Najpierw uruchom trening (python -m src.train)."
        )

    model = UNet1D(in_channels=1, base_channels=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model załadowany z {model_path}")

 
    dev_clean_full = CommonVoicePLDataset(DEV_TSV)

    N_SAMPLES = 300

    if N_SAMPLES is None or N_SAMPLES > len(dev_clean_full):
        indices = list(range(len(dev_clean_full)))
    else:
        indices = random.sample(range(len(dev_clean_full)), k=N_SAMPLES)

    print(f"Liczba przykładów do ewaluacji: {len(indices)}\n")

    subset = torch.utils.data.Subset(dev_clean_full, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_eval)

    sisnr_noisy_sum = 0.0
    sisnr_enh_sum = 0.0

    stoi_noisy_sum = 0.0
    stoi_enh_sum = 0.0

    # listy do wykresów
    sisnr_noisy_list = []
    sisnr_enh_list = []
    stoi_noisy_list = []
    stoi_enh_list = []


    count = 0

    rows = []  # tu zbieramy metryki per-próbka do CSV


    progress = tqdm(loader, desc="Ewaluacja SI-SNR + STOI", ncols=100)

    for batch in progress:
        item = batch[0]
        clean_full = item["audio"]  # torch [T]

        MAX_SECONDS = 4.0
        max_len = int(MAX_SECONDS * SAMPLE_RATE)
        if clean_full.shape[0] > max_len:
            clean_seg = clean_full[:max_len]
        else:
            clean_seg = clean_full

        # dodajemy szum o SNR=5 dB
        noisy_t, clean_t = mix_with_white_noise(clean_seg, snr_db=5.0)

        with torch.no_grad():
            est_t = model(
                noisy_t.unsqueeze(0).unsqueeze(0).to(device)
            ).squeeze(0).squeeze(0).cpu()

        # wyrównanie długości
        min_len = min(clean_t.shape[0], noisy_t.shape[0], est_t.shape[0])
        clean_t = clean_t[:min_len]
        noisy_t = noisy_t[:min_len]
        est_t = est_t[:min_len]

        #SI-SNR
        sisnr_noisy = si_snr_metric(
            noisy_t.unsqueeze(0).unsqueeze(0),
            clean_t.unsqueeze(0).unsqueeze(0),
        )
        sisnr_enh = si_snr_metric(
            est_t.unsqueeze(0).unsqueeze(0),
            clean_t.unsqueeze(0).unsqueeze(0),
        )

        sisnr_noisy_sum += sisnr_noisy
        sisnr_enh_sum += sisnr_enh

        sisnr_noisy_list.append(sisnr_noisy)
        sisnr_enh_list.append(sisnr_enh)

        #STOI
        clean_np = clean_t.numpy()
        noisy_np = noisy_t.numpy()
        est_np = est_t.numpy()

        stoi_noisy = compute_stoi(clean_np, noisy_np, SAMPLE_RATE)
        stoi_enh = compute_stoi(clean_np, est_np, SAMPLE_RATE)

        stoi_noisy_sum += stoi_noisy
        stoi_enh_sum += stoi_enh

        stoi_noisy_list.append(stoi_noisy)
        stoi_enh_list.append(stoi_enh)


        rows.append({
            "id": item.get("id", str(count)),
            "sisnr_noisy": float(sisnr_noisy),
            "sisnr_enh": float(sisnr_enh),
            "dsisnr": float(sisnr_enh - sisnr_noisy),
            "stoi_noisy": float(stoi_noisy),
            "stoi_enh": float(stoi_enh),
            "dstoi": float(stoi_enh - stoi_noisy),
            "n_samples_used_sec": MAX_SECONDS,
            "snr_db": 5.0,
        })

        MODELS_DIR = "models"
        PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

                # =========================
        # WYKRES 1: Scatter SI-SNR
        # =========================
        plt.figure(figsize=(6, 6))

        plt.scatter(
            sisnr_noisy_list,
            sisnr_enh_list,
            alpha=0.4,
            label="próbki audio"
        )

        lims = [
            min(sisnr_noisy_list + sisnr_enh_list),
            max(sisnr_noisy_list + sisnr_enh_list)
        ]
        plt.plot(lims, lims, "r--", label="brak poprawy (y=x)")

        plt.xlabel("SI-SNR noisy [dB]")
        plt.ylabel("SI-SNR enhanced [dB]")
        plt.title("Porównanie jakości: noisy vs enhanced")
        plt.legend()
        plt.grid(True)

        scatter_path = os.path.join(PLOTS_DIR, "sisnr_scatter_noisy_vs_enhanced.png")
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()

        print(f"📊 Scatter SI-SNR zapisano do: {scatter_path}")

                # =========================
        # WYKRES 2: CDF poprawy SI-SNR
        # =========================
        delta_sisnr = np.array(sisnr_enh_list) - np.array(sisnr_noisy_list)
        delta_sorted = np.sort(delta_sisnr)
        cdf = np.arange(1, len(delta_sorted) + 1) / len(delta_sorted)

        plt.figure(figsize=(6, 4))
        plt.plot(delta_sorted, cdf)

        plt.xlabel("Poprawa ΔSI-SNR [dB]")
        plt.ylabel("Odsetek próbek")
        plt.title("CDF poprawy jakości SI-SNR")
        plt.grid(True)

        cdf_path = os.path.join(PLOTS_DIR, "sisnr_cdf.png")
        plt.tight_layout()
        plt.savefig(cdf_path)
        plt.close()

        print(f"📊 CDF SI-SNR zapisano do: {cdf_path}")


        count += 1

        avg_sisnr_noisy = sisnr_noisy_sum / count
        avg_sisnr_enh = sisnr_enh_sum / count
        avg_stoi_noisy = stoi_noisy_sum / count
        avg_stoi_enh = stoi_enh_sum / count

        progress.set_postfix({
            "ΔSI-SNR": f"{(avg_sisnr_enh - avg_sisnr_noisy):.2f} dB",
            "ΔSTOI": f"{(avg_stoi_enh - avg_stoi_noisy):.3f}",
        })

    avg_sisnr_noisy = sisnr_noisy_sum / max(count, 1)
    avg_sisnr_enh = sisnr_enh_sum / max(count, 1)
    sisnr_improvement = avg_sisnr_enh - avg_sisnr_noisy

    avg_stoi_noisy = stoi_noisy_sum / max(count, 1)
    avg_stoi_enh = stoi_enh_sum / max(count, 1)
    stoi_improvement = avg_stoi_enh - avg_stoi_noisy

    print("\n========== RAPORT JAKOŚCI (SI-SNR / STOI) ==========")
    print(f"Liczba przykładów: {count}")
    print("")
    print(f"Średni SI-SNR noisy    : {avg_sisnr_noisy:.2f} dB")
    print(f"Średni SI-SNR enhanced : {avg_sisnr_enh:.2f} dB")
    print(f"Średnia poprawa SI-SNR : {sisnr_improvement:.2f} dB")
    print("")
    print(f"Średni STOI noisy      : {avg_stoi_noisy:.3f}")
    print(f"Średni STOI enhanced   : {avg_stoi_enh:.3f}")
    print(f"Średnia poprawa STOI   : {stoi_improvement:.3f}")
    print("====================================================\n")

    

    report_path = os.path.join(MODELS_DIR, "eval_fullmetrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========== RAPORT JAKOŚCI (SI-SNR / STOI) ==========\n")
        f.write(f"Data: {datetime.datetime.now()}\n")
        f.write(f"Liczba przykładów: {count}\n\n")
        f.write(f"Średni SI-SNR noisy    : {avg_sisnr_noisy:.2f} dB\n")
        f.write(f"Średni SI-SNR enhanced : {avg_sisnr_enh:.2f} dB\n")
        f.write(f"Średnia poprawa SI-SNR : {sisnr_improvement:.2f} dB\n\n")
        f.write(f"Średni STOI noisy      : {avg_stoi_noisy:.3f}\n")
        f.write(f"Średni STOI enhanced   : {avg_stoi_enh:.3f}\n")
        f.write(f"Średnia poprawa STOI   : {stoi_improvement:.3f}\n")
        f.write("====================================================\n")

    print(f"📁 Raport tekstowy zapisano do: {report_path}")

    csv_path = os.path.join(MODELS_DIR, "eval_fullmetrics_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["avg_sisnr_noisy", avg_sisnr_noisy])
        writer.writerow(["avg_sisnr_enhanced", avg_sisnr_enh])
        writer.writerow(["sisnr_improvement", sisnr_improvement])
        writer.writerow(["avg_stoi_noisy", avg_stoi_noisy])
        writer.writerow(["avg_stoi_enhanced", avg_stoi_enh])
        writer.writerow(["stoi_improvement", stoi_improvement])
        writer.writerow(["n_samples", count])

    print(f"📄 CSV zapisano do: {csv_path}")

    per_sample_csv = os.path.join(MODELS_DIR, "eval_per_sample.csv")
    pd.DataFrame(rows).to_csv(per_sample_csv, index=False, encoding="utf-8")
    print(f"📄 Per-sample CSV zapisano do: {per_sample_csv}")

    # 1) SI-SNR noisy vs enhanced
    plt.figure(figsize=(6, 4))
    plt.bar(["noisy", "enhanced"], [avg_sisnr_noisy, avg_sisnr_enh])
    plt.ylabel("SI-SNR [dB]")
    plt.title("Średni SI-SNR (noisy vs enhanced)")
    for i, v in enumerate([avg_sisnr_noisy, avg_sisnr_enh]):
        plt.text(i, v + 0.2, f"{v:.2f}", ha="center")
    sisnr_plot = os.path.join(PLOTS_DIR, "sisnr_noisy_enhanced.png")
    plt.tight_layout()
    plt.savefig(sisnr_plot)
    plt.close()

    # 2) STOI
    plt.figure(figsize=(6, 4))
    plt.bar(["noisy", "enhanced"], [avg_stoi_noisy, avg_stoi_enh])
    plt.ylabel("STOI")
    plt.ylim(0, 1)
    plt.title("Średni STOI (noisy vs enhanced)")
    for i, v in enumerate([avg_stoi_noisy, avg_stoi_enh]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    stoi_plot = os.path.join(PLOTS_DIR, "stoi_noisy_enhanced.png")
    plt.tight_layout()
    plt.savefig(stoi_plot)
    plt.close()

    print(f"📊 Wykres SI-SNR zapisano do: {sisnr_plot}")
    print(f"📊 Wykres STOI zapisano do: {stoi_plot}")
    print("\nGotowe. Możesz użyć tych plików w raporcie / prezentacji.")

        # =========================
    # WYKRES 3: Przykład waveform + spektrogram
    # =========================
    example = dev_clean_full[indices[0]]
    clean = example["audio"][:max_len]

    noisy, _ = mix_with_white_noise(clean, snr_db=5.0)

    with torch.no_grad():
        enhanced = model(
            noisy.unsqueeze(0).unsqueeze(0).to(device)
        ).squeeze().cpu()

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(noisy.numpy(), label="noisy", alpha=0.6)
    plt.plot(enhanced.numpy(), label="enhanced", alpha=0.8)
    plt.legend()
    plt.title("Waveform (czas)")

    plt.subplot(2, 1, 2)
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(enhanced.numpy())),
        ref=np.max
    )
    librosa.display.specshow(
        S, sr=SAMPLE_RATE, x_axis="time", y_axis="hz"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spektrogram (enhanced)")

    wf_path = os.path.join(PLOTS_DIR, "example_waveform_spectrogram.png")
    plt.tight_layout()
    plt.savefig(wf_path)
    plt.close()

    print(f"📊 Waveform + spektrogram zapisano do: {wf_path}")



if __name__ == "__main__":
    main()
