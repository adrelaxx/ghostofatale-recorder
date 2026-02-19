import os
import sys
import whisper
from pydub import AudioSegment
from pydub.utils import make_chunks

def transcribe_large_file(file_path):
    # 1. Création du dossier de sortie
    output_dir = "transcript"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Nom de base pour le fichier de sortie
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}.txt")

    print(f"--- Chargement du modèle Whisper ---")
    model = whisper.load_model("base") # Vous pouvez utiliser 'small' ou 'medium' pour plus de précision

    print(f"--- Chargement de l'audio (cela peut prendre du temps pour 10h) ---")
    audio = AudioSegment.from_file(file_path)

    # 2. Découpage en morceaux de 20 minutes (1200000 ms)
    # C'est plus sûr pour la mémoire vive
    chunk_length_ms = 1200000 
    chunks = make_chunks(audio, chunk_length_ms)

    print(f"--- Début de la transcription ({len(chunks)} segments) ---")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            print(f"Transcription du segment {i+1}/{len(chunks)}...")
            
            # Export temporaire du morceau
            temp_chunk_path = "temp_chunk.wav"
            chunk.export(temp_chunk_path, format="wav")

            # Transcription
            result = model.transcribe(temp_chunk_path, fp16=False)
            
            # Écriture immédiate dans le fichier
            f.write(result["text"] + " ")
            f.flush() # Force l'écriture sur le disque

            # Nettoyage
            os.remove(temp_chunk_path)

    print(f"--- Terminé ! Le transcript est disponible ici : {output_file} ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py nom_du_fichier.mp4")
    else:
        transcribe_large_file(sys.argv[1])
