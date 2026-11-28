# How use this RAG assistance to find at what time frame in the video is the section you needed :

## Step 1: Collect the videos you needed
Move all your videos to the videos folder 

## Step 2: Convert videos to mp3(audio)
Convert all the videos to mp3(audio) and store it to audios folder by running process_video.py file

## Step 3: Convert mp3(audios) to json chunks
Convert all the audios to text and store it with the details like start, end, id, title in json format in audio_jsons folder by running create_chunks.py file

## Step 4: Convert text to vectors
Convert all the text in json files of audio_jsons folder into embeddings using ollama's bge-m3 model by running chunks_to_embeddings.py file and storing all of it into a joblib folder

## Step 5: Getting response from LLM
Get the final response in the response.txt file by running the final_process.py file. In that file the embeddings of the query(question asked by user) are compared with the embeddings of the text and the highest cosine_similarities are given as response

### NOTE : In final_process.py file change the prompt as per requirement if necessary