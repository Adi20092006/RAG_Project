import whisper
import json
import os

audios=os.listdir("audios")
model = whisper.load_model("large-v2")

for audio in audios:

    result = model.transcribe(audio=f"audios/{audio}", 
                              language="Japanese", 
                              task="translate", 
                              word_timestamps=False)
    
    chunk=[]

    for segment in result["segments"]:
        chunk.append({"id":segment["id"], "start":segment["start"], "end":segment["end"], "text":segment["text"]})

    name=audio.split(".")[0]
    chunk_with_text={"Title":name, "text":result["text"], "Chunks":chunk}

    with open(f"audio_jsons/{name}.json", "w") as f:
        json.dump(chunk_with_text, f)