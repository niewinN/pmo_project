
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset


from .utils import TRAIN_TSV, DEV_TSV, MODELS_DIR, get_device, set_seed
from .dataset import CommonVoicePLDataset, PolishEnhancementDataset
from .model import UNet1D
from .losses import CompositeLoss, si_snr_metric

def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Hyperparametry
    batch_size = 8
    num_epochs = 8 
    lr = 1e-3

    # Dane
    print("Ładowanie datasetów...")
    train_clean_full = CommonVoicePLDataset(TRAIN_TSV)
    dev_clean_full   = CommonVoicePLDataset(DEV_TSV)

   
    max_train = 12000   
    max_dev   = 2000


    # Losowe indeksy
    train_indices = torch.randperm(len(train_clean_full))[:max_train].tolist()
    dev_indices   = torch.randperm(len(dev_clean_full))[:max_dev].tolist()

    # Podzbiory
    train_clean = Subset(train_clean_full, train_indices)
    dev_clean   = Subset(dev_clean_full, dev_indices)

    # Enhancement datasets (noisy, clean)
    train_ds = PolishEnhancementDataset(train_clean, snr_min=-5.0, snr_max=10.0)
    dev_ds = PolishEnhancementDataset(dev_clean,   snr_min=-5.0, snr_max=10.0)



    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train size: {len(train_ds)}  |  Dev size: {len(dev_ds)}")

    # Model i strata
    model = UNet1D(in_channels=1, base_channels=16).to(device)
    criterion = CompositeLoss(l1_weight=1.0, sisnr_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_sisnr = -1e9
    model_path = os.path.join(MODELS_DIR, "unet_pl.pth")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_sisnr = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
        for noisy, clean in pbar:
            noisy = noisy.to(device)   # [B, 1, T]
            clean = clean.to(device)   # [B, 1, T]

            optimizer.zero_grad()
            est = model(noisy)

            loss, l1_val, sisnr_val = criterion(est, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_sisnr += sisnr_val
            count += 1

            pbar.set_postfix({"loss": running_loss / count, "SI-SNR": running_sisnr / count})

        # Walidacja
        model.eval()
        val_sisnr = 0.0
        val_count = 0
        with torch.no_grad():
            for noisy, clean in tqdm(dev_loader, desc=f"Epoch {epoch}/{num_epochs} [valid]"):
                noisy = noisy.to(device)
                clean = clean.to(device)
                est = model(noisy)
                sisnr = si_snr_metric(est, clean)
                val_sisnr += sisnr
                val_count += 1

        val_sisnr /= max(val_count, 1)
        print(f"Epoch {epoch} - Validation SI-SNR: {val_sisnr:.2f} dB")

        # Zapis najlepszego modelu
        if val_sisnr > best_val_sisnr:
            best_val_sisnr = val_sisnr
            torch.save(model.state_dict(), model_path)
            print(f"Nowy najlepszy model zapisany do: {model_path}")

    print("Trening zakończony. Najlepsza walidacyjna SI-SNR: {:.2f} dB".format(best_val_sisnr))

if __name__ == "__main__":
    main()
