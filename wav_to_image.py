import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.cm as mpl_cm
from PIL import Image

def make_and_save_sharp_100px_spectrogram(
    wav_path: str,
    low_f: float,
    high_f: float,
    output_path: str,
    nperseg: int = 512,
    noverlap: int = 508,
    nfft: int = 2048,
    cmap: str = 'magma'
):
    """
    Lit le WAV, calcule un spectrogramme, coupe la bande [low_f, high_f],
    applique un colormap, redimensionne à 100×100, et sauve en PNG.
    Ajuste dynamiquement nperseg/noverlap si le signal est plus court.
    Ignore les fichiers pour lesquels la bande freq est vide.
    """
    # 1) Lecture & mono-convert
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # 2) Ajustement dynamique
    L = data.shape[0]
    nperseg_eff  = min(nperseg, L)
    noverlap_eff = min(noverlap, nperseg_eff - 1) if nperseg_eff > 1 else 0

    # 3) Spectrogramme
    f, t, Sxx = spectrogram(
        data,
        fs=sr,
        nperseg=nperseg_eff,
        noverlap=noverlap_eff,
        nfft=nfft,
        scaling='density'
    )

    # 4) Assurez-vous que Sxx est 2D
    if Sxx.ndim == 1:
        Sxx = Sxx[:, np.newaxis]

    # 5) Découpe fréquentielle
    mask = (f >= low_f) & (f <= high_f)
    if not mask.any():
        print(f"[WARN] Aucune fréquence entre {low_f}–{high_f} Hz pour {wav_path}")
        return  # on skip ce fichier

    Sxx_crop = Sxx[mask, :]

    # 6) Normalisation
    mn, mx = Sxx_crop.min(), Sxx_crop.max()
    if mn == mx:
        print(f"[WARN] Signal constant dans la bande pour {wav_path}")
        return  # on skip aussi

    Sxx_norm = (Sxx_crop - mn) / (mx - mn)

    # 7) Colormap → RGB
    cmap_obj = mpl_cm.get_cmap(cmap)
    rgba     = cmap_obj(Sxx_norm)               # (H, W, 4)
    rgb      = (rgba[..., :3] * 255).astype('uint8')

    # 8) Resize 100×100
    img = Image.fromarray(rgb)
    img = img.resize((100, 100), resample=Image.LANCZOS)

    # 9) Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"[OK] Saved spectrogram → {output_path}")

if __name__ == "__main__":
    # === À ADAPTER ===
    base_dir      = "/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/val_2"
    ann_dir       = os.path.join(base_dir, "annotations")
    segments_root = os.path.join(base_dir, "audio")       # vos WAV (tf_cropped_wav)
    out_root      = os.path.join(base_dir, "spectrograms_100x100")
    wav_col       = "tf_cropped_wav"                      # colonne CSV contenant le nom complet du .wav
    # ==================

    for csv_path in glob.glob(os.path.join(ann_dir, "*.csv")):
        dataset = os.path.splitext(os.path.basename(csv_path))[0]
        df      = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            cls    = row['annotation']
            low_f  = float(row['low_frequency'])
            high_f = float(row['high_frequency'])
            wav_fn = row[wav_col]   # ex: "2015-02-04T03-27-32_053000_21.9-28.4Hz.wav"

            if not isinstance(wav_fn, str) or not wav_fn.endswith(".wav"):
                print(f"[WARN] nom invalide dans '{wav_col}': {wav_fn!r}")
                continue

            wav_path = os.path.join(segments_root, dataset, cls, wav_fn)
            if not os.path.isfile(wav_path):
                print(f"[WARN] absent: {wav_path}")
                continue

            png_name = wav_fn.replace(".wav", ".png")
            out_path = os.path.join(out_root, dataset, cls, png_name)

            make_and_save_sharp_100px_spectrogram(
                wav_path=wav_path,
                low_f=low_f,
                high_f=high_f,
                output_path=out_path
            )
