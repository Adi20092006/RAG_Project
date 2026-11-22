import whisper
import json

model = whisper.load_model("large-v2")
result = model.transcribe(audio="audios/audioE11-10.mp3", language="Japanese", task="translate", word_timestamps=False)
chunk=[]

for segment in result["segments"]:
    chunk.append({"id":segment["id"], "start":segment["start"], "end":segment["end"], "text":segment["text"]})

with open("output.json", "w") as f:
    json.dump(chunk, f)