import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = os.path.join("models", "eval_per_sample.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Brak {csv_path}. Najpierw uruchom: python -m src.eval_fullmetrics"
        )

    df = pd.read_csv(csv_path)
    os.makedirs(os.path.join("models", "plots"), exist_ok=True)
    plots_dir = os.path.join("models", "plots")

    # --- Podstawowe statystyki ---
    n = len(df)
    pct_sisnr_pos = 100.0 * (df["dsisnr"] > 0).mean()
    pct_stoi_pos = 100.0 * (df["dstoi"] > 0).mean()

    # Bootstrap 95% CI dla średniej poprawy (ładnie wygląda w raporcie)
    def bootstrap_ci(x, iters=2000, alpha=0.05, seed=0):
        rng = np.random.default_rng(seed)
        x = np.asarray(x)
        means = []
        for _ in range(iters):
            samp = rng.choice(x, size=len(x), replace=True)
            means.append(np.mean(samp))
        lo = np.quantile(means, alpha/2)
        hi = np.quantile(means, 1-alpha/2)
        return float(lo), float(hi)

    dsisnr = df["dsisnr"].to_numpy()
    dstoi = df["dstoi"].to_numpy()
    ci_sisnr = bootstrap_ci(dsisnr, seed=1)
    ci_stoi = bootstrap_ci(dstoi, seed=2)

    # --- 1) Wykres DELTA (najbardziej czytelny) ---
    plt.figure(figsize=(7,4))
    delta_means = [df["dsisnr"].mean(), df["dstoi"].mean()]
    plt.bar(["ΔSI-SNR [dB]", "ΔSTOI"], delta_means)
    plt.title(f"Średnia poprawa (N={n})")
    plt.grid(axis="y", alpha=0.3)

    # opisy
    plt.text(0, delta_means[0], f"{delta_means[0]:.2f}\n95% CI [{ci_sisnr[0]:.2f}, {ci_sisnr[1]:.2f}]",
             ha="center", va="bottom")
    plt.text(1, delta_means[1], f"{delta_means[1]:.3f}\n95% CI [{ci_stoi[0]:.3f}, {ci_stoi[1]:.3f}]",
             ha="center", va="bottom")

    out1 = os.path.join(plots_dir, "delta_improvements.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()

    plt.figure(figsize=(7,4))
    plt.hist(df["dsisnr"], bins=30)
    plt.title(f"Rozkład poprawy ΔSI-SNR (N={n}) | %>0 dB: {pct_sisnr_pos:.1f}%")
    plt.xlabel("ΔSI-SNR [dB]")
    plt.ylabel("Liczba próbek")
    plt.grid(alpha=0.3)
    out2 = os.path.join(plots_dir, "hist_dsisnr.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()

    plt.figure(figsize=(7,4))
    plt.boxplot([df["dsisnr"], df["dstoi"]], labels=["ΔSI-SNR [dB]", "ΔSTOI"])
    plt.title(f"Rozkład poprawy (boxplot) | %ΔSI-SNR>0: {pct_sisnr_pos:.1f}% | %ΔSTOI>0: {pct_stoi_pos:.1f}%")
    plt.grid(axis="y", alpha=0.3)
    out3 = os.path.join(plots_dir, "boxplot_deltas.png")
    plt.tight_layout()
    plt.savefig(out3, dpi=160)
    plt.close()

    summary_path = os.path.join("models", "eval_summary_extra.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== DODATKOWE PODSUMOWANIE (per-sample) ===\n")
        f.write(f"Liczba próbek: {n}\n\n")
        f.write(f"Średnia poprawa ΔSI-SNR: {df['dsisnr'].mean():.2f} dB\n")
        f.write(f"95% CI (bootstrap): [{ci_sisnr[0]:.2f}, {ci_sisnr[1]:.2f}] dB\n")
        f.write(f"% próbek z poprawą (ΔSI-SNR > 0): {pct_sisnr_pos:.1f}%\n\n")
        f.write(f"Średnia poprawa ΔSTOI: {df['dstoi'].mean():.3f}\n")
        f.write(f"95% CI (bootstrap): [{ci_stoi[0]:.3f}, {ci_stoi[1]:.3f}]\n")
        f.write(f"% próbek z poprawą (ΔSTOI > 0): {pct_stoi_pos:.1f}%\n")

    print("✅ Zapisano nowe wykresy:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)
    print("📄 Podsumowanie:", summary_path)

if __name__ == "__main__":
    main()
