import stable_whisper
import sys
import os
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

    output_dir = "transcript"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # 1. Utilisation de stable-ts (plus précis sur les coupures de voix)
    print(f"--- Mode: CPU | Modèle: Small (Stable-TS) ---")
    print("Chargement du modèle...")
    model = stable_whisper.load_model("small", device="cpu")

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

            segment_video = video.subclipped(current_time, end_time)
            segment_video.audio.write_audiofile(temp_audio, codec="pcm_s16le")

            # 2. Transcription optimisée pour plusieurs voix
            # condition_on_previous_text=False empêche le modèle de s'auto-influencer
            result = model.transcribe(
                temp_audio, 
                language="fr", 
                fp16=False,
                condition_on_previous_text=False,
                regroup=True # Regroupe les mots de façon logique
            )

            # 3. Écriture avec sauts de ligne marqués
            for seg in result.segments:
                start_glob = format_timestamp(seg.start + current_time)
                end_glob = format_timestamp(seg.end + current_time)
                text = seg.text.strip()
                
                # On ajoute un saut de ligne supplémentaire pour séparer visuellement les blocs
                f.write(f"[{start_glob} - {end_glob}] {text}\n\n")

            f.flush()

            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            current_time += chunk_size
            progress = min(round((current_time / duration) * 100, 1), 100)
            print(f">>> Progression : {progress}%")

    video.close()
    print(f"\nTranscription terminée ! Fichier : {output_file}")

if __name__ == "__main__":
    main()
