import numpy as np
from read_chunks import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

embeds_df=joblib.load("embeddings.joblib")

query=input("What you want: ")
embeded_question=create_embeddings(query)[0]

similarities=cosine_similarity(np.vstack(embeds_df['embedding'].values),[embeded_question])
print(similarities)

highly_matched=np.argmax(similarities)
result=embeds_df.iloc[highly_matched]
print(result)