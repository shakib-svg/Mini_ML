import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

def sanitize_datetime_string(dt_str: str) -> str:
    """
    "2015-02-16T11:54:10.436000+00:00" → "2015-02-16T11-54-10_436000"
    """
    # on retire le suffixe "+00:00"
    core = dt_str[:-6]
    return core.replace(":", "-").replace(".", "_")

def tf_crop_and_invert(
    csv_path: str,
    trimmed_root: str,
    output_root: str,
    updated_csv_path: str,
    n_fft: int = 1024,
    hop_length: int = 512,
    target_sr: int = None
):
    """
    - csv_path         : votre CSV original (avec low/high_frequency et annotation et start_datetime)
    - trimmed_root     : dossier de vos WAV déjà découpés en temps, organisé par classe :
                         trimmed_root/<annotation>/<timestamp>.wav
    - output_root      : où sauver les WAV time‑freq cropped (même arborescence par classe)
    - updated_csv_path : chemin du CSV de sortie, avec la colonne tf_cropped_wav
    - n_fft, hop_length: paramètres de STFT/ISTFT
    - target_sr        : si None, on conserve le SR d'origine du fichier
    """
    df       = pd.read_csv(csv_path)
    new_names = []

    for idx, row in df.iterrows():
        cls       = row['annotation']
        low_hz    = float(row['low_frequency'])
        high_hz   = float(row['high_frequency'])
        label     = sanitize_datetime_string(row['start_datetime'])
        filename  = f"{label}.wav"

        in_path = os.path.join(trimmed_root, cls, filename)
        if not os.path.isfile(in_path):
            print(f"[WARN] introuvable : {in_path}")
            new_names.append("")
            continue

        # 1) chargement
        y, sr = librosa.load(in_path, sr=target_sr)
        # 2) STFT
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        # 3) masque fréquentiel
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mask  = (freqs >= low_hz) & (freqs <= high_hz)
        S_masked = np.zeros_like(S)
        S_masked[mask, :] = S[mask, :]

        # 4) ISTFT
        y_crop = librosa.istft(S_masked, hop_length=hop_length, length=len(y))

        # 5) sauvegarde
        out_dir = os.path.join(output_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{label}_{low_hz:.1f}-{high_hz:.1f}Hz.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, y_crop, sr)

        print(f"[OK] {out_path}")
        new_names.append(out_name)

    # 6) mise à jour du CSV
    df['tf_cropped_wav'] = new_names
    df.to_csv(updated_csv_path, index=False)
    print(f"\n✅ CSV mis à jour écrit dans : {updated_csv_path}")



if __name__ == "__main__":
    # === À adapter ===
    csv_file         = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train/annotations/rosssea2014.csv'
    trimmed_root       = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train_1/rosssea2014'
    output_root    = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train_2/rosssea2014'
    updated_csv_output = "/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train_2/anotat/rosssea2014.csv"
    # ==================

    tf_crop_and_invert(
        csv_path=csv_file,
        trimmed_root=trimmed_root,
        output_root=output_root,
        updated_csv_path=updated_csv_output,
        n_fft=1024,
        hop_length=512,
        target_sr=None
    )
