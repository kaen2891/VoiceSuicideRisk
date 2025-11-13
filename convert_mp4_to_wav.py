import os
import ffmpeg
from tqdm import tqdm

root_dir = "./dataset"

# serach all mp4 files
mp4_files = []
for root, _, files in os.walk(root_dir):
    for f in files:
        if f.endswith(".mp4"):
            mp4_files.append(os.path.join(root, f))

print(f"{len(mp4_files)} mp4 files.")

# convert
for mp4_path in tqdm(mp4_files, desc="Converting"):
    wav_path = os.path.splitext(mp4_path)[0] + ".wav"
    if os.path.exists(wav_path):
        continue
    try:
        (
            ffmpeg
            .input(mp4_path)
            .output(wav_path, acodec='pcm_s16le', ac=1, ar='16000')  # mono, 16kHz
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        print(f"failed converting: {mp4_path}")
        print(e)

print(" all samples were transformed")
