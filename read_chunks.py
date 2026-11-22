import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embeddings(text_list):
    r=requests.post("http://localhost:11434/api/embed",
                    json={"model":"bge-m3",
                          "input":text_list}
                    )
    embedding=r.json()["embeddings"]
    return embedding

jsons=os.listdir("audio_jsons") #list of json files

json_dict=[]

for js in jsons:
    with open(f"audio_jsons/{js}", "r") as f: #opening files one by one
        j=json.load(f)
        list_of_text=[a["text"] for a in j["Chunks"]] 
        embeds=create_embeddings(list_of_text)

    for i, chunk in enumerate(j["Chunks"]):
        chunk["title"]=j["Title"]
        chunk["embedding"]=embeds[i]
        json_dict.append(chunk)

embedding_df=pd.DataFrame.from_records(json_dict)
joblib.dump(embedding_df, "embeddings.joblib")

# print(np.vstack(embedding_df["embedding"].values))
# print(np.vstack(embedding_df["embedding"]).shape)

