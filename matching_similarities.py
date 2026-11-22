import numpy as np
from read_chunks import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import joblib

embeds_df=joblib.load("embeddings.joblib")

query=input("What you want: ")
embeded_question=create_embeddings(query)[0]

similarities=cosine_similarity(np.vstack(embeds_df['embedding'].values),[embeded_question]).flatten()
print(similarities)

top_matchups=3
highly_matched=similarities.argsort()[::-1][:top_matchups]

print(embeds_df.iloc[highly_matched])

