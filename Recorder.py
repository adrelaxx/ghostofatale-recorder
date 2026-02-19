import whisper
import sys
import os
import torch
from datetime import timedelta
from moviepy import VideoFileClip

def format_timestamp(seconds):
    """Formate les secondes en HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python Recorder.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print("Erreur : fichier vidéo introuvable.")
        sys.exit(1)

    # 1. Dossier de sortie
    output_dir = "transcript"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # 2. Configuration CPU & Modèle Small
    device = "cpu"
    print(f"--- Mode: CPU | Modèle: Small ---")
    print("Chargement du modèle Whisper...")
    model = whisper.load_model("small", device=device)

    # 3. Chargement vidéo
    print("Analyse de la durée...")
    video = VideoFileClip(video_path)
    duration = video.duration
    chunk_size = 1800  # 30 minutes

    print(f"Début de la transcription vers : {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        current_time = 0

        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            temp_audio = f"temp_segment_{int(current_time)}.wav"

            print(f"\nTraitement : {format_timestamp(current_time)} -> {format_timestamp(end_time)}")

            # MoviePy 2.0+
            segment = video.subclipped(current_time, end_time)

            # Export audio (compatible v2.0+)
            segment.audio.write_audiofile(
                temp_audio,
                codec="pcm_s16le"
            )

            # Transcription
            result = model.transcribe(
                temp_audio,
                language="fr",
                fp16=False
            )

            # Écriture avec timestamps globaux
            for seg in result["segments"]:
                start_glob = format_timestamp(seg["start"] + current_time)
                end_glob = format_timestamp(seg["end"] + current_time)
                text = seg["text"].strip()
                f.write(f"[{start_glob} - {end_glob}] {text}\n")

            f.flush()

            # Nettoyage
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            current_time += chunk_size
            progress = min(round((current_time / duration) * 100, 1), 100)
            print(f">>> Progression : {progress}%")

    video.close()
    print(f"\nFini ! Résultat : {output_file}")


if __name__ == "__main__":
    main()
