import os
import datetime
import csv
import matplotlib.pyplot as plt

# Średni SI-SNR noisy    : 5.00 dB
# Średni SI-SNR enhanced : 13.68 dB
# Średnia poprawa        : 8.68 dB
# Liczba przykładów      : 9918

AVG_NOISY = 5.00
AVG_ENH   = 13.68
IMPROVEMENT = 8.68
N_SAMPLES = 9918

MODELS_DIR = "models"
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

report_path = os.path.join(MODELS_DIR, "eval_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("========== RAPORT JAKOŚCI (SI-SNR) ==========\n")
    f.write(f"Data: {datetime.datetime.now()}\n")
    f.write(f"Liczba przykładów: {N_SAMPLES}\n\n")
    f.write(f"Średni SI-SNR noisy   : {AVG_NOISY:.2f} dB\n")
    f.write(f"Średni SI-SNR enhanced: {AVG_ENH:.2f} dB\n")
    f.write(f"Średnia poprawa       : {IMPROVEMENT:.2f} dB\n")
    f.write("=============================================\n")

print(f"📁 Raport tekstowy zapisano do: {report_path}")

csv_path = os.path.join(MODELS_DIR, "eval_results.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["avg_sisnr_noisy",     AVG_NOISY])
    writer.writerow(["avg_sisnr_enhanced",  AVG_ENH])
    writer.writerow(["avg_improvement",     IMPROVEMENT])
    writer.writerow(["n_samples",           N_SAMPLES])

print(f"📄 CSV z wynikami zapisano do: {csv_path}")

plt.figure(figsize=(6, 4))
labels = ["noisy", "enhanced"]
values = [AVG_NOISY, AVG_ENH]

plt.bar(labels, values)
plt.ylabel("SI-SNR [dB]")
plt.title("Średni SI-SNR: noisy vs enhanced")

for i, v in enumerate(values):
    plt.text(i, v + 0.3, f"{v:.2f} dB", ha="center")

plot_path = os.path.join(PLOTS_DIR, "sisnr_bar.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"📊 Wykres słupkowy zapisano do: {plot_path}")

plt.figure(figsize=(4, 4))
plt.bar(["improvement"], [IMPROVEMENT])
plt.ylabel("Poprawa SI-SNR [dB]")
plt.title("Średnia poprawa SI-SNR")

plt.text(0, IMPROVEMENT + 0.3, f"{IMPROVEMENT:.2f} dB", ha="center")
plot_imp_path = os.path.join(PLOTS_DIR, "sisnr_improvement.png")
plt.tight_layout()
plt.savefig(plot_imp_path)
plt.close()

print(f"📊 Wykres poprawy zapisano do: {plot_imp_path}")
print("\nGotowe. Możesz użyć tych plików w raporcie / prezentacji.")
