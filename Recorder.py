import whisper
import sys
import os
import torch
from datetime import timedelta
from moviepy import VideoFileClip

def format_timestamp(seconds):
    """Formatte les secondes en [HH:MM:SS]"""
    return str(timedelta(seconds=int(seconds)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python Recorder.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    
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

    # 3. Traitement Vidéo
    print("Analyse de la durée...")
    video = VideoFileClip(video_path)
    duration = video.duration
    chunk_size = 1800  # Segments de 30 minutes
    
    print(f"Début de la transcription vers : {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        current_time = 0
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            temp_audio = f"temp_segment_{int(current_time)}.wav"
            
            print(f"\nTraitement : {format_timestamp(current_time)} -> {format_timestamp(end_time)}")
            
            # --- CORRECTION MOVIEPY V2.0+ ---
            segment = video.subclipped(current_time, end_time)
            
            # Note: On a enlevé 'verbose' et 'logger' qui font planter MoviePy 2.0
            segment.audio.write_audiofile(temp_audio, codec='pcm_s16le')
            
            # Transcription (Langue française forcée)
            result = model.transcribe(temp_audio, language="fr", fp16=False)
            
            # Écriture des timestamps
            for seg in result["segments"]:
                start_glob = format_timestamp(seg["start"] + current_time)
                end_glob = format_timestamp(seg["end"] + current_time)
                f.write(f"[{start_glob} - {end_glob}] {seg['text'].strip()}\n")
            
            f.flush() 
            
            # Nettoyage
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            current_time += chunk_size
            print(f">>> Progression : {round((current_time/duration)*100, 1)}%")

    video.close()
    print(f"\nFini ! Résultat : {output_file}")

if __name__ == "__main__":
    main()
