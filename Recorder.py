import whisper
import sys
import os
import torch
from datetime import timedelta
from moviepy import VideoFileClip

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_cpu.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = "transcript"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    # Force CPU et chargement du modèle
    device = "cpu"
    print(f"--- Mode: CPU | Modèle: Small ---")
    model = whisper.load_model("small", device=device)

    print("Analyse du fichier vidéo...")
    video = VideoFileClip(video_path)
    duration = video.duration
    chunk_size = 1800  # 30 minutes
    
    print(f"Transcription en cours vers : {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        current_time = 0
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            temp_audio = f"temp_segment_{current_time}.wav"
            
            print(f"\nTraitement : {format_timestamp(current_time)} -> {format_timestamp(end_time)}")
            
            # --- CORRECTION ICI POUR MOVIEPY v2.0+ ---
            # On utilise .sub() au lieu de .subclip()
            segment = video.sub(current_time, end_time)
            segment.audio.write_audiofile(temp_audio, codec='pcm_s16le', verbose=False, logger=None)
            
            # Transcription
            result = model.transcribe(temp_audio, language="fr", fp16=False)
            
            for seg in result["segments"]:
                start_glob = format_timestamp(seg["start"] + current_time)
                end_glob = format_timestamp(seg["end"] + current_time)
                f.write(f"[{start_glob} - {end_glob}] {seg['text'].strip()}\n")
            
            f.flush()
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            current_time += chunk_size
            print(f">>> Progression : {round((current_time/duration)*100, 1)}%")

    video.close()
    print(f"\nTerminé ! Résultat dans : {output_file}")

if __name__ == "__main__":
    main()
