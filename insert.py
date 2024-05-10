import chromadb
import numpy as np
import time

#constant variables
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(name="my_collection")


def get_vectors(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Create a 2D array
    vectors = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        line = line[1:-1]  # Remove square brackets
        values = line.split(',')  # Split values by comma
        values = [float(value) for value in values]  # Convert values to float
        vectors.append(values)

    return (vectors)

def insert_embeddings_in_batch(batch_embeddings_data, batch_id_data):
    collection.add(
        embeddings=batch_embeddings_data,
        ids=batch_id_data
    )


#Get all the vectors from the file
vectors = get_vectors("./embeddings.txt")
#Define batch_size
batch_size = 2
#Initialize the batch_data list
batch_embeddings_data = []
batch_id_data = []
# Initialize list to store time taken for each batch
batch_time = []
line_number = 0

for i in range(len(vectors)):
    batch_embeddings_data.append(vectors[i])
    batch_id_data.append(str(line_number))
    line_number += 1
    if len(batch_embeddings_data) == batch_size:
        batch_start_time = time.time()
        insert_embeddings_in_batch(batch_embeddings_data, batch_id_data)
        batch_time.append(time.time() - batch_start_time)
        batch_embeddings_data = []
        batch_id_data = []

total_time = sum(batch_time)
# Print the time taken
print("List of durations for batch insertion: ", batch_time)
print(f"Time taken to complete the insertion: {total_time} seconds")
        
