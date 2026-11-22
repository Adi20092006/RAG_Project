import os
import subprocess

video_files=os.listdir("videos")

i=1
for v_file in video_files:
    clip_no=v_file.split(" (")[1].split(")")[0]
    ep_no=v_file.split("1E")[1].split(" (")[0]
    print(clip_no)
    audio_file=subprocess.run(["ffmpeg", "-i", f"videos/{v_file}", f"audios/audioE{ep_no}-{clip_no}.mp3"])

