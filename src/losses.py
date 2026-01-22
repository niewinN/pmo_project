import torch
import torch.nn as nn
from speechbrain.utils.metric_stats import MetricStats

def si_snr(est, ref, eps: float = 1e-8):
    """
    Scale-Invariant SNR (SI-SNR) w dB.
    est, ref: tensory [B, 1, T] albo [B, T]
    """
    if est.dim() == 3:
        est = est.squeeze(1)  # [B, T]
    if ref.dim() == 3:
        ref = ref.squeeze(1)  # [B, T]

    # Wyzerowanie średniej (usunięcie DC)
    est_zm = est - est.mean(dim=1, keepdim=True)
    ref_zm = ref - ref.mean(dim=1, keepdim=True)

    # Projekcja estymatu na sygnał referencyjny
    dot = (est_zm * ref_zm).sum(dim=1, keepdim=True)         # [B, 1]
    ref_energy = (ref_zm ** 2).sum(dim=1, keepdim=True) + eps
    s_target = dot / ref_energy * ref_zm                     # [B, T]
    e_noise = est_zm - s_target                              # [B, T]

    # SI-SNR w dB
    ratio = (s_target ** 2).sum(dim=1) / ((e_noise ** 2).sum(dim=1) + eps)
    si_snr_val = 10 * torch.log10(ratio + eps)               # [B]
    return si_snr_val


def si_snr_loss(est, ref):
    """
    Funkcja straty oparta o SI-SNR:
    chcemy MAKSYMALIZOWAĆ SI-SNR, więc loss = -mean(SI-SNR).
    """
    return -si_snr(est, ref).mean()


def si_snr_metric(est: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Wygodna wersja do logowania – zwraca SI-SNR w dB (większa = lepsza).
    """
    return float(si_snr(est, ref).mean().item())


class CompositeLoss(nn.Module):
    """
    Strata łącząca:
      - L1 między sygnałem wyjściowym a czystym,
      - SI-SNR (nasza implementacja, inspirowana metryką używaną w SpeechBrain).

    Dodatkowo trzymamy obiekt MetricStats z SpeechBrain,
    żeby łatwo było rozszerzyć projekt o metryki z SpeechBrain.
    """

    def __init__(self, l1_weight: float = 1.0, sisnr_weight: float = 0.3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.sisnr_weight = sisnr_weight

        self.stats = MetricStats(metric=lambda x, y: si_snr_metric(x, y))

    def forward(self, est: torch.Tensor, ref: torch.Tensor):
        """
        est, ref: [B, 1, T]
        Zwraca:
          - loss (skalar),
          - L1 (float),
          - SI-SNR w dB (float, większa = lepsza)
        """
        
        min_len = min(est.shape[-1], ref.shape[-1])
        est = est[:, :, :min_len]
        ref = ref[:, :, :min_len]

        l1_val = self.l1(est, ref)
        sisnr_l = si_snr_loss(est, ref)

        loss = self.l1_weight * l1_val + self.sisnr_weight * sisnr_l

        sisnr_db = si_snr_metric(est, ref)

        return loss, float(l1_val.item()), sisnr_db
