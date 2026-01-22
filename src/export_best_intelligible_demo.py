import os
import random
import datetime

import torch
import torchaudio
import librosa
from pystoi import stoi

from .utils import get_device, DEV_TSV
from .dataset import CommonVoicePLDataset, mix_with_white_noise, SAMPLE_RATE
from .model import UNet1D
from .losses import si_snr_metric


# ---- (opcjonalnie) wyrównanie głośności do odsłuchu, nie do metryk ----
def rms_normalize(x: torch.Tensor, target_dbfs: float = -20.0, eps: float = 1e-8) -> torch.Tensor:
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


def compute_sisnr(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        return float(
            si_snr_metric(
                a.unsqueeze(0).unsqueeze(0),
                b.unsqueeze(0).unsqueeze(0),
            )
        )

def compute_stoi(clean_t: torch.Tensor, est_t: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    # STOI w pystoi standardowo liczy się dla 10 kHz
    clean = clean_t.detach().cpu().numpy()
    est = est_t.detach().cpu().numpy()

    target_sr = 10000
    if sr != target_sr:
        clean = librosa.resample(clean, orig_sr=sr, target_sr=target_sr)
        est = librosa.resample(est, orig_sr=sr, target_sr=target_sr)
        sr_use = target_sr
    else:
        sr_use = sr

    return float(stoi(clean, est, sr_use, extended=False))


def build_slow_pool(dev_ds, pool_size=2000, min_sec=1.5, max_sec=5.0, max_chars_per_sec=12.0):
    """
    Budujemy pulę indeksów z DEV, które wyglądają na wolniejsze i sensowne do demo.
    Kryterium "wolno": liczba znaków w transkrypcji / sekundy <= max_chars_per_sec.
    """
    good = []
    tries = 0

    while len(good) < pool_size and tries < pool_size * 20:
        idx = random.randint(0, len(dev_ds) - 1)
        item = dev_ds[idx]
        a = item["audio"]
        text = item.get("text", "")

        dur = a.numel() / SAMPLE_RATE
        if dur < min_sec or dur > max_sec:
            tries += 1
            continue

        # jeśli brak tekstu, pomijamy (bo nie ocenimy tempa)
        if not text or len(text.strip()) < 5:
            tries += 1
            continue

        cps = len(text) / max(dur, 1e-6)  # chars per second
        if cps <= max_chars_per_sec:
            good.append(idx)

        tries += 1

    if len(good) == 0:
        raise RuntimeError("Nie udało się zbudować puli wolniejszych klipów. Zwiększ max_chars_per_sec lub poluzuj min/max_sec.")

    return good


def make_concat_from_pool(dev_ds, pool_indices, target_seconds=25.0, n_clips_min=10,
                          max_each_seconds=4.0, min_each_seconds=1.5, pause_seconds=0.25):
    target_len = int(target_seconds * SAMPLE_RATE)
    max_each = int(max_each_seconds * SAMPLE_RATE)
    min_each = int(min_each_seconds * SAMPLE_RATE)
    pause = torch.zeros(int(pause_seconds * SAMPLE_RATE))

    chunks = []
    picked_ids = []
    total = 0

    # Losujemy z puli “wolniejszych” klipów
    while (len(picked_ids) < n_clips_min or total < target_len):
        idx = random.choice(pool_indices)
        item = dev_ds[idx]
        a = item["audio"]
        uid = item.get("id", f"dev_{idx}")
        text = item.get("text", "")

        # ucinamy do max_each, ale zostawiamy minimum
        if a.numel() > max_each:
            a = a[:max_each]
        if a.numel() < min_each:
            continue

        chunks.append(a)
        chunks.append(pause)
        picked_ids.append((uid, text))

        total += a.numel() + pause.numel()

        # zabezpieczenie: nie rośnij w nieskończoność
        if len(picked_ids) > 50:
            break

    clean_full = torch.cat(chunks, dim=0)
    clean = clean_full[:target_len] if clean_full.numel() > target_len else clean_full
    return clean, picked_ids


def main():
    device = get_device()
    print("Using device:", device)

    model_path = os.path.join("models", "unet_pl.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Brak models/unet_pl.pth. Najpierw uruchom trening (python -m src.train).")

    model = UNet1D(in_channels=1, base_channels=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dev = CommonVoicePLDataset(DEV_TSV)

    # ----------------- PARAMETRY DEMO -----------------
    SNR_DB = 5.0

    # szukamy najbardziej zrozumiałego (STOI) spośród kandydatów
    N_CANDIDATES = 20
    TARGET_SECONDS = 25.0
    N_CLIPS_MIN = 10

    # filtr “wolniejszych” klipów
    POOL_SIZE = 1500
    MIN_SEC = 1.5
    MAX_SEC = 5.0
    MAX_CHARS_PER_SEC = 12.0  # mniejsza liczba = wolniej

    # pauzy między klipami (większe = łatwiej zrozumieć demo)
    PAUSE_SECONDS = 0.25
    # --------------------------------------------------

    print("Buduję pulę wolniejszych klipów (to chwilę potrwa, ale bez treningu)...")
    pool = build_slow_pool(
        dev_ds=dev,
        pool_size=POOL_SIZE,
        min_sec=MIN_SEC,
        max_sec=MAX_SEC,
        max_chars_per_sec=MAX_CHARS_PER_SEC
    )
    print(f"✅ Pula gotowa: {len(pool)} klipów spełnia warunki tempa i długości.\n")

    best = None

    try:
        from tqdm import tqdm
        iterator = tqdm(range(N_CANDIDATES), desc="Szukam najbardziej zrozumiałego demo (STOI)", ncols=95)
    except Exception:
        iterator = range(N_CANDIDATES)

    for _ in iterator:
        clean, picked = make_concat_from_pool(
            dev_ds=dev,
            pool_indices=pool,
            target_seconds=TARGET_SECONDS,
            n_clips_min=N_CLIPS_MIN,
            max_each_seconds=4.0,
            min_each_seconds=MIN_SEC,
            pause_seconds=PAUSE_SECONDS,
        )

        noisy, clean_ref = mix_with_white_noise(clean, snr_db=SNR_DB)

        with torch.no_grad():
            enhanced = model(noisy.unsqueeze(0).unsqueeze(0).to(device)).squeeze().cpu()

        m = min(clean_ref.numel(), noisy.numel(), enhanced.numel())
        clean_ref = clean_ref[:m]
        noisy = noisy[:m]
        enhanced = enhanced[:m]

        # metryki
        sisnr_noisy = compute_sisnr(noisy, clean_ref)
        sisnr_enh = compute_sisnr(enhanced, clean_ref)
        delta_sisnr = sisnr_enh - sisnr_noisy

        stoi_noisy = compute_stoi(clean_ref, noisy, SAMPLE_RATE)
        stoi_enh = compute_stoi(clean_ref, enhanced, SAMPLE_RATE)
        delta_stoi = stoi_enh - stoi_noisy

        # Kryterium wyboru: chcemy wysokie STOI(enh), a nie tylko duże ΔSI-SNR
        score = stoi_enh  # możesz zmienić np. score = stoi_enh + 0.2 * delta_sisnr

        cand = {
            "score": score,
            "stoi_noisy": stoi_noisy,
            "stoi_enh": stoi_enh,
            "delta_stoi": delta_stoi,
            "sisnr_noisy": sisnr_noisy,
            "sisnr_enh": sisnr_enh,
            "delta_sisnr": delta_sisnr,
            "clean": clean_ref,
            "noisy": noisy,
            "enhanced": enhanced,
            "picked": picked,
            "seconds": m / SAMPLE_RATE,
        }

        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        raise RuntimeError("Nie udało się znaleźć żadnego kandydata.")

    out_dir = os.path.join("outputs", "best_intelligible_demo")
    os.makedirs(out_dir, exist_ok=True)

    # zapis surowy
    torchaudio.save(os.path.join(out_dir, "clean.wav"), best["clean"].unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, "noisy.wav"), best["noisy"].unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, "enhanced.wav"), best["enhanced"].unsqueeze(0), SAMPLE_RATE)

    # zapis do odsłuchu (wyrównanie poziomu)
    clean_l = peak_normalize(soft_limiter(rms_normalize(best["clean"], -20.0), 0.99), -1.0)
    noisy_l = peak_normalize(soft_limiter(rms_normalize(best["noisy"], -20.0), 0.99), -1.0)
    enh_l = peak_normalize(soft_limiter(rms_normalize(best["enhanced"], -20.0), 0.99), -1.0)

    torchaudio.save(os.path.join(out_dir, "clean_listen.wav"), clean_l.unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, "noisy_listen.wav"), noisy_l.unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, "enhanced_listen.wav"), enh_l.unsqueeze(0), SAMPLE_RATE)

    # raport tekstowy (żeby było co pokazać)
    report_path = os.path.join(out_dir, "summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BEST INTELLIGIBLE DEMO SUMMARY\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"SNR_DB: {SNR_DB}\n")
        f.write(f"TARGET_SECONDS: {TARGET_SECONDS}\n")
        f.write(f"N_CANDIDATES: {N_CANDIDATES}\n\n")
        f.write(f"Seconds: {best['seconds']:.2f}\n")
        f.write(f"STOI noisy: {best['stoi_noisy']:.3f}\n")
        f.write(f"STOI enh  : {best['stoi_enh']:.3f}\n")
        f.write(f"ΔSTOI     : {best['delta_stoi']:.3f}\n\n")
        f.write(f"SI-SNR noisy: {best['sisnr_noisy']:.2f} dB\n")
        f.write(f"SI-SNR enh  : {best['sisnr_enh']:.2f} dB\n")
        f.write(f"ΔSI-SNR     : {best['delta_sisnr']:.2f} dB\n\n")
        f.write("Clips used (id + text):\n")
        for uid, text in best["picked"][:15]:
            f.write(f"- {uid}: {text}\n")
        if len(best["picked"]) > 15:
            f.write("...\n")

    print("\n✅ Zapisano najbardziej zrozumiałe demo do outputs/best_intelligible_demo/")
    print("Do odsłuchu polecam:")
    print(" - outputs/best_intelligible_demo/noisy_listen.wav")
    print(" - outputs/best_intelligible_demo/enhanced_listen.wav")
    print("\nWyniki:")
    print(f"  STOI noisy: {best['stoi_noisy']:.3f}  | STOI enh: {best['stoi_enh']:.3f}  | ΔSTOI: {best['delta_stoi']:.3f}")
    print(f"  SI-SNR noisy: {best['sisnr_noisy']:.2f} dB | SI-SNR enh: {best['sisnr_enh']:.2f} dB | ΔSI-SNR: {best['delta_sisnr']:.2f} dB")
    print(f"  Raport: {report_path}")


if __name__ == "__main__":
    main()
