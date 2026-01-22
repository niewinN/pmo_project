
# Projekt: Poprawa jakości sygnału mowy (PL) z użyciem SpeechBrain i U-Net 1D

Ten projekt realizuje zadanie poprawy jakości sygnału mowy w języku polskim, z wykorzystaniem:
- korpusu Mozilla Common Voice (polski),
- toolkitu **SpeechBrain** (metryka SI-SNR, opcjonalnie baseline MetricGAN+),
- autorskiego modelu **U-Net 1D** działającego w dziedzinie czasu.

## Struktura projektu

```text
speech_enhancement_pl/
├── data/
│   └── common_voice/
│       ├── clips/           # tutaj umieść pliki audio z Common Voice (mp3)
│       ├── train.tsv
│       └── dev.tsv
├── models/
│   └── unet_pl.pth          # zapisany wytrenowany model (po treningu)
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
├── notebooks/
│   └── analysis.ipynb       # opcjonalna eksploracja w Jupyter/VS Code
├── requirements.txt
└── README.md
```

## Wymagania (wersje kompatybilne ze SpeechBrain)

Zalecane środowisko (np. conda):

```bash
conda create -n speechbrain python=3.9
conda activate speechbrain

pip install -r requirements.txt
```

W pliku `requirements.txt` ustawiono wersje:

- `torch==1.12.1`
- `torchaudio==0.12.1`
- `speechbrain==0.5.14`

To konfiguracja sprawdzona jako kompatybilna z SpeechBrain.

## Dane (Common Voice PL)

Pobierz polski Common Voice (np. wersja 17 lub 23), rozpakuj i skopiuj:
- `train.tsv` do `data/common_voice/train.tsv`
- `dev.tsv`   do `data/common_voice/dev.tsv`
- katalog `clips/` z plikami mp3 również do `data/common_voice/clips/`

Jeśli masz inną ścieżkę, zmień `DATA_ROOT` w `src/utils.py`.

## Uruchamianie treningu

W folderze `speech_enhancement_pl` uruchom:

```bash
conda activate speechbrain   # jeśli używasz conda
pip install -r requirements.txt

python src/train.py
```

Po treningu zapisany model pojawi się jako `models/unet_pl.pth`.

## Inference / przykłady odsłuchu

Uruchom:

```bash
python src/infer.py
```

Skrypt:
- wczyta wytrenowany model,
- pobierze kilka przykładów z dev setu,
- wygeneruje zaszumione wersje,
- odszumi je modelem,
- zapisze wyniki w `outputs/` (clean / noisy / enhanced) oraz wypisze SI-SNR.
