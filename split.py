import os
import pandas as pd
from pydub import AudioSegment
from datetime import datetime, timezone

def parse_wav_start_datetime(wav_filename):
    base = os.path.splitext(os.path.basename(wav_filename))[0]
    # Suppose que la partie .wav est comme "2015-04-05T03-00-00_000"
    if base.endswith('_000'):
        base = base[:-4]
    fmt = '%Y-%m-%dT%H-%M-%S'
    wav_start = datetime.strptime(base, fmt)
    wav_start = wav_start.replace(tzinfo=timezone.utc)
    return wav_start

def get_seconds_offset(wav_start_dt, annotation_dt_str):
    fmt_in = '%Y-%m-%dT%H:%M:%S.%f%z'
    annotation_dt = datetime.strptime(annotation_dt_str, fmt_in)
    return (annotation_dt - wav_start_dt).total_seconds()

def sanitize_datetime_string(dt_str):
    fmt_in = '%Y-%m-%dT%H:%M:%S.%f%z'
    dt = datetime.strptime(dt_str, fmt_in)
    fmt_out = '%Y-%m-%dT%H-%M-%S_%f'
    return dt.strftime(fmt_out)

def extract_segments_from_csv(csv_path, audio_root, output_root):
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)

    for idx, row in df.iterrows():
        wav_filename   = row['filename']   # ex: "ballenyislands2015/2015-04-05T03-00-00_000.wav"
        annotation_cls = row['annotation']
        start_dt_str   = row['start_datetime']
        end_dt_str     = row['end_datetime']

        # Construit le chemin complet du .wav
        full_wav_path = os.path.join(audio_root, wav_filename)

        if not os.path.isfile(full_wav_path):
            print(f"[WARN] Fichier introuvable: {full_wav_path}")
            continue

        # Parse la date/heure de début à partir du nom du .wav (on ne prend que la partie basename)
        wav_start_dt = parse_wav_start_datetime(wav_filename)

        start_offset_s = get_seconds_offset(wav_start_dt, start_dt_str)
        end_offset_s   = get_seconds_offset(wav_start_dt, end_dt_str)

        audio = AudioSegment.from_wav(full_wav_path)
        start_ms = int(start_offset_s * 1000)
        end_ms   = int(end_offset_s * 1000)

        if start_ms < 0 or end_ms < 0:
            print(f"[WARN] Offset négatif (ligne {idx}) -> ignoré.")
            continue
        if end_ms <= start_ms:
            print(f"[WARN] end <= start (ligne {idx}) -> ignoré.")
            continue

        segment = audio[start_ms:end_ms]
        clip_start_label = sanitize_datetime_string(start_dt_str)

        # Dossier de sortie, ex: .../out/ballenyislands2015/bma/
        # Dans ce script on extrait "ballenyislands2015" du path
        dataset_part = os.path.dirname(wav_filename)  # = "ballenyislands2015"
        out_dir = os.path.join(output_root, dataset_part, annotation_cls)
        os.makedirs(out_dir, exist_ok=True)

        out_filename = f"{clip_start_label}.wav"
        out_path = os.path.join(out_dir, out_filename)
        segment.export(out_path, format="wav")
        print(f"[OK] {out_path} sauvegardé")


if __name__ == "__main__":
    # Exemple d’utilisation :
    
    # 1) Chemin vers le CSV (annotations)
    csv_file = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train/annotations/ballenyislands2015.csv'
    
    # 2) Dossier racine où se trouvent les .wav
    #    Par ex: ".../biodcase_development_set/train/audio/"
    audio_root = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train/audio/ballenyislands2015'
    
    # 3) Dossier de sortie pour les extraits
    output_root = '/home/shakib/Desktop/S4/machine_learning/mini_projet/biodcase_development_set/train_1'
    
    # Appel de la fonction principale
    extract_segments_from_csv(csv_file, audio_root, output_root)

    
