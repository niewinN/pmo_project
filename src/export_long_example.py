import os
import random
import datetime

import torch
import torchaudio

from .utils import get_device, DEV_TSV
from .dataset import CommonVoicePLDataset, mix_with_white_noise, SAMPLE_RATE
from .model import UNet1D
from .losses import si_snr_metric  # nasza implementacja SI-SNR


# --------- Post-processing do odsłuchu (uczciwe: tylko dla demo) ---------
def rms_normalize(x: torch.Tensor, target_dbfs: float = -20.0, eps: float = 1e-8) -> torch.Tensor:
    # x: [T]
    rms = torch.sqrt(torch.mean(x**2) + eps)
    current_dbfs = 20.0 * torch.log10(rms + eps)
    gain_db = target_dbfs - current_dbfs
    gain = 10.0 ** (gain_db / 20.0)
    return x * gain


def peak_normalize(x: torch.Tensor, peak_dbfs: float = -1.0, eps: float = 1e-8) -> torch.Tensor:
    peak = torch.max(torch.abs(x)) + eps
    target_peak = 10.0 ** (peak_dbfs / 20.0)
    return x * (target_peak / peak)


def soft_limiter(x: torch.Tensor, threshold: float = 0.99) -> torch.Tensor:
    return torch.clamp(x, -threshold, threshold)


# --------- Sklejanie wielu krótkich klipów z DEV do jednego dłuższego ---------
def make_concat_sample(
    dev_ds,
    target_seconds: float = 25.0,
    n_clips_min: int = 10,
    max_each_seconds: float = 4.0,
    min_each_seconds: float = 1.2,
    pause_seconds: float = 0.15,
    max_tries: int = 6000,
):
    target_len = int(target_seconds * SAMPLE_RATE)
    max_each = int(max_each_seconds * SAMPLE_RATE)
    min_each = int(min_each_seconds * SAMPLE_RATE)
    pause = torch.zeros(int(pause_seconds * SAMPLE_RATE))

    chunks = []
    picked_ids = []
    total = 0
    tries = 0

    while (len(picked_ids) < n_clips_min or total < target_len) and tries < max_tries:
        j = random.randint(0, len(dev_ds) - 1)
        item = dev_ds[j]
        a = item["audio"]

        if a.numel() > max_each:
            a = a[:max_each]

        if a.numel() < min_each:
            tries += 1
            continue

        chunks.append(a)
        chunks.append(pause)
        picked_ids.append(item.get("id", f"dev_{j}"))
        total += a.numel() + pause.numel()
        tries += 1

    if len(picked_ids) == 0:
        raise RuntimeError("Nie udało się stworzyć sklejki (spróbuj zmniejszyć min_each_seconds).")

    clean_full = torch.cat(chunks, dim=0)
    clean = clean_full[:target_len] if clean_full.numel() > target_len else clean_full
    return clean, picked_ids


def compute_sisnr(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [T] -> SI-SNR w dB (float)
    with torch.no_grad():
        return float(
            si_snr_metric(
                a.unsqueeze(0).unsqueeze(0),
                b.unsqueeze(0).unsqueeze(0),
            )
        )


def main():
    device = get_device()
    print("Using device:", device)

    model_path = os.path.join("models", "unet_pl.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Brak models/unet_pl.pth. Najpierw uruchom trening (python -m src.train).")

    # base_channels musi pasować do treningu (u Ciebie 16)
    model = UNet1D(in_channels=1, base_channels=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dev = CommonVoicePLDataset(DEV_TSV)

    
    N_CANDIDATES = 30
    TARGET_SECONDS = 25.0     
    N_CLIPS_MIN = 10          
    SNR_DB = 5.0              
   

    best = None  

    try:
        from tqdm import tqdm
        iterator = tqdm(range(N_CANDIDATES), desc="Szukam najlepszego demo", ncols=95)
    except Exception:
        iterator = range(N_CANDIDATES)

    for k in iterator:
        clean = None
        try:
            clean, picked_ids = make_concat_sample(
                dev_ds=dev,
                target_seconds=TARGET_SECONDS,
                n_clips_min=N_CLIPS_MIN,
                max_each_seconds=4.0,
                min_each_seconds=1.2,
                pause_seconds=0.15,
                max_tries=7000,
            )

            noisy, clean_ref = mix_with_white_noise(clean, snr_db=SNR_DB)

            with torch.no_grad():
                enhanced = model(noisy.unsqueeze(0).unsqueeze(0).to(device)).squeeze().cpu()

            # wyrównanie długości
            m = min(clean_ref.numel(), noisy.numel(), enhanced.numel())
            clean_ref = clean_ref[:m]
            noisy = noisy[:m]
            enhanced = enhanced[:m]

            sisnr_noisy = compute_sisnr(noisy, clean_ref)
            sisnr_enh = compute_sisnr(enhanced, clean_ref)
            delta = sisnr_enh - sisnr_noisy

            candidate = {
                "delta": delta,
                "sisnr_noisy": sisnr_noisy,
                "sisnr_enh": sisnr_enh,
                "clean": clean_ref,
                "noisy": noisy,
                "enhanced": enhanced,
                "picked_ids": picked_ids,
                "seconds": m / SAMPLE_RATE,
            }

            if best is None or candidate["delta"] > best["delta"]:
                best = candidate

            # jeśli tqdm jest, pokaż live wynik
            if "tqdm" in globals() or "tqdm" in str(type(iterator)):
                pass

        except Exception as e:
            # pomiń kandydata, idź dalej
            continue

    if best is None:
        raise RuntimeError("Nie udało się znaleźć żadnego poprawnego kandydata. Zmniejsz ograniczenia sklejki.")

    # ------------------- ZAPIS NAJLEPSZEGO DEMO -------------------
    out_dir = os.path.join("outputs", "best_demo")
    os.makedirs(out_dir, exist_ok=True)

    clean_path = os.path.join(out_dir, "clean.wav")
    noisy_path = os.path.join(out_dir, "noisy.wav")
    enh_path = os.path.join(out_dir, "enhanced.wav")

    torchaudio.save(clean_path, best["clean"].unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(noisy_path, best["noisy"].unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(enh_path, best["enhanced"].unsqueeze(0), SAMPLE_RATE)

    # wersje "pod odsłuch" (wyrównanie głośności + limiter + peak)
    clean_listen = peak_normalize(soft_limiter(rms_normalize(best["clean"], -20.0), 0.99), -1.0)
    noisy_listen = peak_normalize(soft_limiter(rms_normalize(best["noisy"], -20.0), 0.99), -1.0)
    enh_listen = peak_normalize(soft_limiter(rms_normalize(best["enhanced"], -20.0), 0.99), -1.0)

    torchaudio.save(os.path.join(out_dir, "clean_listen.wav"), clean_listen.unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, "noisy_listen.wav"), noisy_listen.unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, "enhanced_listen.wav"), enh_listen.unsqueeze(0), SAMPLE_RATE)

    # raport
    report_path = os.path.join(out_dir, "best_demo_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BEST DEMO SUMMARY\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"N_CANDIDATES: {N_CANDIDATES}\n")
        f.write(f"TARGET_SECONDS: {TARGET_SECONDS}\n")
        f.write(f"SNR_DB: {SNR_DB}\n\n")
        f.write(f"Seconds saved: {best['seconds']:.2f}\n")
        f.write(f"SI-SNR noisy: {best['sisnr_noisy']:.2f} dB\n")
        f.write(f"SI-SNR enh  : {best['sisnr_enh']:.2f} dB\n")
        f.write(f"ΔSI-SNR     : {best['delta']:.2f} dB\n\n")
        f.write("First 15 clip IDs used in concat:\n")
        for x in best["picked_ids"][:15]:
            f.write(f" - {x}\n")
        if len(best["picked_ids"]) > 15:
            f.write(" ...\n")

    print("\n✅ Zapisano najlepsze demo do outputs/best_demo/")
    print("Pliki (surowe):")
    print(" -", clean_path)
    print(" -", noisy_path)
    print(" -", enh_path)
    print("Pliki (do odsłuchu, wyrównana głośność):")
    print(" - outputs/best_demo/clean_listen.wav")
    print(" - outputs/best_demo/noisy_listen.wav")
    print(" - outputs/best_demo/enhanced_listen.wav")
    print("\nWynik:")
    print(f"  SI-SNR noisy : {best['sisnr_noisy']:.2f} dB")
    print(f"  SI-SNR enh   : {best['sisnr_enh']:.2f} dB")
    print(f"  ΔSI-SNR      : {best['delta']:.2f} dB")
    print(f"  Długość      : {best['seconds']:.2f} s")
    print(f"  Raport       : {report_path}")


if __name__ == "__main__":
    main()
