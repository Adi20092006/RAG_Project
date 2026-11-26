import numpy as np
from chunks_to_embeddings import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

embeds_df=joblib.load("embeddings.joblib")

query=input("What you want: ")
embeded_question=create_embeddings(query)[0]

similarities=cosine_similarity(np.vstack(embeds_df['embedding'].values),[embeded_question]).flatten()
# print(similarities)

top_matchups=3
highly_matched=similarities.argsort()[::-1][:top_matchups]
highly_matched_df=embeds_df.iloc[highly_matched]


print(highly_matched_df)

# prompt for llama3.2 to get a good output
prompt=f"""In some series of videos that are stored user wants to know where to the thing he is looking for, so the here is information of the videos that highly matches the user question below,so i am providing the subtitled text of the videos, videos time stamp(start and end), video title and id:
{highly_matched_df[["id","start","end","text","title"]].to_json()}

-------------------------------------
'{query}'
Based on the question user is asking, tell the user where he can find the answer(at what time stamp and the videos title) he is looking for in the series of the videos.Don't give it in a regular format, give it in a human readable format.
"""
# To see how overall looks like :
# with open("prompt.txt", "w") as f:
#     f.write(prompt)

def genrate_response(prompt):
    r=requests.post("http://localhost:11434/api/generate",
                    json={"model":"llama3.2",
                          "prompt":prompt,
                          "stream":False}
                    )
    response=r.json()
    return response

with open("response.txt", "w") as f:
    f.write(genrate_response(prompt)["response"])