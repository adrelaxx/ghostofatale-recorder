import whisper
import sys
import os
import torch
from datetime import timedelta
from moviepy import VideoFileClip

def format_timestamp(seconds):
    """Convertit les secondes en format [HH:MM:SS]"""
    return str(timedelta(seconds=int(seconds)))

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcribe_cpu.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    
    # 1. Gestion des répertoires
    output_dir = "transcript"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # 2. Configuration CPU & Modèle
    # On force le CPU même si un GPU est présent pour répondre à ta demande
    device = "cpu"
    print(f"--- Mode: CPU uniquement | Modèle: Small ---")
    
    print("Chargement du modèle Whisper (patience...)")
    model = whisper.load_model("small", device=device)

    # 3. Analyse de la vidéo
    print("Analyse de la durée du fichier...")
    video = VideoFileClip(video_path)
    duration = video.duration
    
    # Découpage par tranches de 30 minutes (1800s) pour la RAM
    chunk_size = 1800 
    
    print(f"Début de la transcription vers : {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        current_time = 0
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            temp_audio = "temp_segment_audio.wav"
            
            print(f"\nTraitement du segment : {format_timestamp(current_time)} -> {format_timestamp(end_time)}")
            
            # Extraction audio via MoviePy (évite les erreurs audioop de Python 3.13)
            subclip = video.subclip(current_time, end_time)
            subclip.audio.write_audiofile(temp_audio, codec='pcm_s16le', verbose=False, logger=None)
            
            # Transcription du segment (langue forcée FR)
            # fp16=False est obligatoire sur CPU
            result = model.transcribe(temp_audio, language="fr", fp16=False)
            
            # Écriture avec calcul des timestamps globaux
            for segment in result["segments"]:
                start_glob = format_timestamp(segment["start"] + current_time)
                end_glob = format_timestamp(segment["end"] + current_time)
                text = segment["text"].strip()
                f.write(f"[{start_glob} - {end_glob}] {text}\n")
            
            # Sauvegarde immédiate sur le disque
            f.flush()
            
            # Nettoyage du fichier temporaire
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            current_time += chunk_size
            progression = round((current_time / duration) * 100, 1)
            print(f">>> Progression globale : {min(progression, 100.0)}%")

    video.close()
    print(f"\nFélicitations ! Transcription terminée dans : {output_file}")

if __name__ == "__main__":
    main()
