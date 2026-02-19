import stable_whisper
import sys
import os
from datetime import timedelta
from moviepy import VideoFileClip

def format_timestamp(seconds):
    """Convertit les secondes en format propre HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python Recorder.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = "transcript"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    print(f"--- Mode: CPU | Diarisation et Correction Encodage ---")
    model = stable_whisper.load_model("small", device="cpu")

    video = VideoFileClip(video_path)
    duration = video.duration
    chunk_size = 1200 
    
    # Variables pour gérer l'alternance des voix
    current_speaker = 1
    last_end_time = 0

    # On utilise 'utf-8' pour éviter les caractères bizarres comme Ã©
    with open(output_file, "w", encoding="utf-8") as f:
        current_time = 0
        while current_time < duration:
            end_time = min(current_time + chunk_size, duration)
            temp_audio = f"temp_segment_{int(current_time)}.wav"
            
            segment_video = video.subclipped(current_time, end_time)
            segment_video.audio.write_audiofile(temp_audio, codec="pcm_s16le")

            # Transcription avec paramètres de séparation
            result = model.transcribe(temp_audio, language="fr", fp16=False)

            for seg in result.segments:
                start_glob_sec = seg.start + current_time
                end_glob_sec = seg.end + current_time
                
                # LOGIQUE DE NUMÉROTATION :
                # Si le silence entre deux segments est > 0.8 seconde, on change de personne.
                # C'est ce qui permet de séparer le streamer de la série.
                if (seg.start - last_end_time) > 0.8:
                    current_speaker = 1 if current_speaker == 2 else 2
                
                start_str = format_timestamp(start_glob_sec)
                end_str = format_timestamp(end_glob_sec)
                text = seg.text.strip()
                
                # Écriture formatée avec le Locuteur
                f.write(f"[{start_str} - {end_str}] LOCUTEUR {current_speaker} : {text}\n")
                
                last_end_time = seg.end

            f.flush()
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            current_time += chunk_size

    video.close()
    print(f"\nTerminé ! Le fichier propre est ici : {output_file}")

if __name__ == "__main__":
    main()
