#!/bin/bash

# --- CONFIGURATION ---
INPUT_ROOT="./runs/train/exp1/visualizations"
OUTPUT_ROOT="./video"
# Quality Setting (Lower is better quality. 17-23 is High Def standard)
CRF=17
# ---------------------

mkdir -p "$OUTPUT_ROOT"
echo "Looking for sequences in: $INPUT_ROOT"

for seq_dir in "$INPUT_ROOT"/*; do    
    if [ -d "$seq_dir" ]; then        
        seq_name=$(basename "$seq_dir")        
        echo "Processing sequence: $seq_name..."

        # Run FFmpeg
        # -framerate 30: Set FPS
        # -pattern_type glob -i "*.png": Grab all PNGs in alphanumeric order
        # -c:v libx264: Use H.264 codec (High Definition)
        # -crf $CRF: Visual quality control
        # -pix_fmt yuv420p: Ensures video plays on all media players (QuickTime, Windows Media, etc.)
        
        ffmpeg -y -framerate 30 -pattern_type glob -i "$seq_dir/*.png" \
        -c:v libx264 -crf "$CRF" -pix_fmt yuv420p \
        "$OUTPUT_ROOT/${seq_name}.mp4"

        echo "Saved: $OUTPUT_ROOT/${seq_name}.mp4"
    fi
done

echo "All videos processed!"