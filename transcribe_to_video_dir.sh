#!/bin/bash

VIDEO="ma_video.mp4"

VIDEO_DIR=$(dirname "$VIDEO")
VIDEO_BASENAME=$(basename "$VIDEO" .mp4)

PROJECT_DIR="$HOME/whisper_project"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" || exit

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install git+https://github.com/openai/whisper.git

ffmpeg -i "$VIDEO" -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav

mkdir -p audio_parts
ffmpeg -i audio.wav -f segment -segment_time 1800 -c copy audio_parts/part_%03d.wav

mkdir -p transcriptions
for f in audio_parts/*.wav; do
    echo "Transcription de $f..."
    whisper "$f" --language French --model medium --output_format txt --output_dir transcriptions
done

FINAL_TXT="$VIDEO_DIR/${VIDEO_BASENAME}_transcription.txt"
cat transcriptions/*.txt > "$FINAL_TXT"

echo "Fichier final : $FINAL_TXT"
